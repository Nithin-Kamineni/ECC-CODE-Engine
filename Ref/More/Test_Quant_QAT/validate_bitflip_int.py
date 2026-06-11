# validate_bitflip_int.py
# Bit-flip validation experiment, INTEGER-SIDE variant.
#
# Difference from validate_bitflip.py:
#   - For INT8 / INT4 formats, the bit flip operates on the actual STORED
#     INTEGER value of the weight, not on its dequantized float.
#   - For format=float32, this script falls back to float-side flip
#     (since there is no integer storage there).
#
# Models attackers who can perturb the stored integer weights in true
# INT8 / INT4 deployments such as TensorRT, ONNX Runtime native INT8,
# or hardware accelerators that operate on packed integer kernels.
#
# Procedure for INT8 / INT4:
#   1) Quantize the model with test-Quantizer's per-channel/per-tensor PTQ.
#      Save (q_int, scale_info) for every weight tensor.
#   2) Dequantize once to populate the model's float weights for inference.
#   3) Bit-flip a chosen weight by:
#        a) flipping a bit (default sign bit) of its INT q value
#        b) re-dequantizing using the SAME scale factor for that channel
#        c) overwriting the model's float weight at that position
#   4) Evaluate accuracy.
#   5) Restore the original q (and corresponding float) before next attack.
#
# Sign-bit semantics for n-bit two's complement:
#   bit position = n-1
#   flip via XOR with 1<<(n-1) on the unsigned wrap, then re-interpret as signed
#   For symmetric quantizers used here, original q in [-2^(n-1)+1, +2^(n-1)-1].
#
# Author: Habibur Rahaman, University of Florida, ECE Department

from __future__ import annotations
import os, csv, time, argparse, importlib.util, sys, struct
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def _load_test_quantizer():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [os.path.join(here, "test-Quantizer.py"),
                  os.path.join(here, "test_quantizer.py"),
                  "test-Quantizer.py", "test_quantizer.py"]
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("test_quantizer", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["test_quantizer"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("Could not locate test-Quantizer.py")

TQ = _load_test_quantizer()


# =========================================================
# Helpers
# =========================================================

def _iter_weight_params(model):
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        lname = name.lower()
        if lname.endswith(".bias") or lname.endswith("_bias"): continue
        if "bn" in lname or ".norm" in lname or "running_" in lname: continue
        if p.ndim in (2, 4) and p.dtype.is_floating_point:
            yield name, p


def name_to_param(model):
    return {n: p for n, p in _iter_weight_params(model)}


@torch.no_grad()
def evaluate_top1(model, loader, device, max_batches=None):
    model.eval()
    correct = total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches: break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
    return 100.0 * correct / max(1, total)


@torch.no_grad()
def avg_loss(model, loader, device, max_batches=2, criterion=None):
    """Average loss over a few batches for empirical scoring."""
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches: break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tot += criterion(model(x), y).item()
        n += 1
    return tot / max(1, n)


# =========================================================
# Float-side flip (fallback when format=float32)
# =========================================================

def _flip_float32_bit(value: float, bit_pos) -> float:
    raw = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    if bit_pos == "sign":      bit = 31
    elif bit_pos == "msb_exp": bit = 30
    elif bit_pos == "lsb_mant": bit = 0
    else: bit = int(bit_pos)
    raw ^= (1 << bit)
    return struct.unpack(">f", struct.pack(">I", raw))[0]


# =========================================================
# Integer-side flip
# =========================================================

def flip_int_bit_signed(q: int, bit_pos, num_bits: int) -> int:
    """
    Flip one bit of an n-bit two's-complement integer.
    Returns the new value as a signed Python int.
    """
    if bit_pos == "sign":      bit = num_bits - 1
    elif bit_pos == "msb_exp": bit = max(0, num_bits - 2)
    elif bit_pos == "lsb_mant": bit = 0
    else: bit = int(bit_pos)
    if bit >= num_bits: bit = num_bits - 1

    mask = (1 << num_bits) - 1
    u = q & mask
    u ^= (1 << bit)
    if u & (1 << (num_bits - 1)):
        return u - (1 << num_bits)
    return u


# =========================================================
# QuantView: keeps integer storage so bit-flips are physical
# =========================================================

class QuantView:
    """
    Holds (q_int_tensor, scale_info, shape) for every weight tensor of the
    model, and applies dequantized floats back into the model so inference
    runs on the PTQ baseline.

    Then `flip_int(layer, flat_idx, bit_pos)` flips a bit on the stored q,
    re-dequantizes that one element, patches the float weight in the model.
    `undo()` reverses a list of those flips.
    """
    def __init__(self, model, num_bits: int, precomputed_scales: Optional[Dict[str, torch.Tensor]] = None):
        self.num_bits = num_bits
        self.entries: Dict[str, dict] = {}
        # Identify which parameters are weights of nn.Conv2d / nn.Linear.
        # Everything else (BatchNorm running_mean/var, biases, etc.) must
        # NOT be quantized, or the model will collapse.
        quantizable_names = set()
        for module_name, module in model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                quantizable_names.add(f"{module_name}.weight" if module_name else "weight")
        with torch.no_grad():
            for name, p in model.state_dict().items():
                if not p.dtype.is_floating_point:
                    continue
                if name not in quantizable_names:
                    continue  # skip BN buffers, biases, etc.
                if precomputed_scales is not None and name in precomputed_scales:
                    # Use the exact training-time scales (from QAT) and
                    # recover integers by dividing. Round to nearest integer;
                    # since weights are grid-aligned, rounding is lossless.
                    s = precomputed_scales[name].to(p.device).to(torch.float32)
                    s_b = s
                    while s_b.ndim < p.ndim: s_b = s_b.unsqueeze(-1)
                    q_int = torch.round(p / s_b).to(torch.int64)
                    qmax = (1 << (num_bits - 1)) - 1
                    q_int = q_int.clamp(-qmax - 1, qmax)
                    sinfo = {"type": "per_channel", "scales": s.detach().cpu().clone(),
                             "axis": 0, "num_bits": num_bits}
                    q_tensor = q_int
                else:
                    q_tensor, sinfo = TQ.quantize_param_smart(name, p, num_bits)
                if sinfo is None:
                    continue
                # widen to int64 so per-element Python int arithmetic is safe
                self.entries[name] = {
                    "q_int": q_tensor.to(torch.int64).cpu().contiguous(),
                    "sinfo": sinfo,
                    "shape": tuple(p.shape),
                }
        self._apply_all_to_model(model)

    # ---- baseline: write dequantized floats into model ----
    def _apply_all_to_model(self, model):
        sd = model.state_dict(); new_sd = {}
        with torch.no_grad():
            for k, p in sd.items():
                if k in self.entries and p.dtype.is_floating_point:
                    sinfo = self.entries[k]["sinfo"]
                    q = self.entries[k]["q_int"]
                    if sinfo["type"] == "per_tensor":
                        w_dq = q.to(torch.float32) * float(sinfo["scale"])
                    elif sinfo["type"] == "per_channel":
                        s = sinfo["scales"].to(torch.float32)
                        while s.ndim < q.ndim: s = s.unsqueeze(-1)
                        w_dq = q.to(torch.float32) * s
                    else:
                        w_dq = q.to(torch.float32)
                    new_sd[k] = w_dq.to(p.device).to(p.dtype)
                else:
                    new_sd[k] = p
            model.load_state_dict(new_sd, strict=True)

    def _scale_for(self, name: str, multi_idx: tuple) -> float:
        sinfo = self.entries[name]["sinfo"]
        if sinfo["type"] == "per_tensor":
            return float(sinfo["scale"])
        return float(sinfo["scales"][int(multi_idx[0])].item())

    def flip_int(self, model, name: str, flat_idx: int, bit_pos="sign"):
        ent = self.entries[name]
        q_flat = ent["q_int"].view(-1)
        old_q = int(q_flat[flat_idx].item())
        new_q = flip_int_bit_signed(old_q, bit_pos, self.num_bits)
        q_flat[flat_idx] = new_q

        multi = np.unravel_index(flat_idx, ent["shape"])
        scale = self._scale_for(name, multi)
        new_dq = float(new_q) * scale

        with torch.no_grad():
            p = dict(model.named_parameters())[name]
            old_dq = float(p.data.view(-1)[flat_idx].item())
            p.data.view(-1)[flat_idx] = new_dq
        return (name, flat_idx, old_q, old_dq)

    def undo(self, model, undos):
        with torch.no_grad():
            n2p = dict(model.named_parameters())
            for name, flat_idx, old_q, old_dq in undos:
                self.entries[name]["q_int"].view(-1)[flat_idx] = old_q
                if name in n2p:
                    n2p[name].data.view(-1)[flat_idx] = old_dq


# =========================================================
# CSV reading + selection
# =========================================================

def read_top_weights(csv_path: str) -> List[dict]:
    rows = []
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            row["taylor"]    = float(row["taylor"])
            row["fisher"]    = float(row["fisher"])
            row["magnitude"] = float(row["magnitude"])
            row["w"]         = float(row["w"])
            row["flat_idx"]  = int(row["flat_idx"])
            rows.append(row)
    return rows


def pick_top_k(rows, k, by="taylor"):
    return sorted(rows, key=lambda r: r[by], reverse=True)[:k]


def pick_random_k(model, k, rng):
    pool = []
    for name, p in _iter_weight_params(model):
        pool.append((name, p.numel()))
    sizes = np.array([n for _, n in pool], dtype=np.int64)
    cum = np.cumsum(sizes)
    chosen = []
    picks = rng.integers(low=0, high=int(sizes.sum()), size=k)
    for g in picks:
        layer_idx = int(np.searchsorted(cum, g, side="right"))
        local = int(g - (cum[layer_idx - 1] if layer_idx > 0 else 0))
        chosen.append((pool[layer_idx][0], local))
    return chosen


# =========================================================
# Float-side fallback (only used when format=float32)
# =========================================================

def apply_flips_float(model, picks, bit_pos="sign"):
    n2p = name_to_param(model)
    undos = []
    for lname, idx in picks:
        if lname not in n2p: continue
        p = n2p[lname]
        flat = p.data.view(-1)
        old = float(flat[idx].item())
        new = _flip_float32_bit(old, bit_pos)
        flat[idx] = new
        undos.append((lname, idx, old))
    return undos

def undo_flips_float(model, undos):
    n2p = name_to_param(model)
    for lname, idx, old in undos:
        if lname in n2p:
            n2p[lname].data.view(-1)[idx] = old


# =========================================================
# Per (arch, format) experiment
# =========================================================

def run_experiment_one(arch: str, fmt: str, args):
    device = TQ.pick_device(args.device, local_rank=0)
    base_tag = f"{args.dataset.lower()}_{arch.lower()}"
    csv_path = os.path.join(
        args.in_dir,
        f"layer_then_weight_{base_tag}_{fmt}_L{args.top_layers}xN{args.top_per_layer}_{args.layer_metric}.csv"
    )
    if not os.path.isfile(csv_path):
        print(f"  [skip] missing {csv_path}"); return []
    rows = read_top_weights(csv_path)
    if not rows:
        print(f"  [skip] empty {csv_path}"); return []
    print(f"  [read] {csv_path}  ({len(rows)} candidate weights)")

    _, _, nc = TQ.get_datasets(args.dataset, args.data_root,
                                args.imagenet_root or None)
    _, test_loader, _, _ = TQ.get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers,
        dist_mode=False, imagenet_root=args.imagenet_root or None)

    model = TQ.build_model(arch, nc, use_pretrained=bool(args.use_pretrained)).to(device)
    qat_scales: Optional[Dict[str, torch.Tensor]] = None
    if args.weights:
        payload = torch.load(args.weights, map_location=device)
        sd = payload.get("state_dict", payload)
        sd = TQ.strip_prefix_from_state_dict(sd)
        model.load_state_dict(sd, strict=True)
    elif args.qtag == "qat" and fmt.startswith("int"):
        # Load a QAT-trained checkpoint produced by qat_train.py.
        # Two possible formats:
        #   1. NEW format (qat_baked=True): plain state_dict containing
        #      already-fake-quantized float weights, plus the EXACT
        #      per-channel scales used by fake_quant() during training.
        #      We hand those scales to QuantView so it does not
        #      recompute them (which would drift on depthwise nets).
        #   2. OLD format: PTQ-style payload {qstate_dict, meta:{scales}}.
        nb = int(fmt[3:])
        ck = TQ.ckpt_path(args.dataset, arch, f"int{nb}_qat{args.qtag_suffix}")
        if os.path.exists(ck):
            payload = torch.load(ck, map_location=device)
            if "state_dict" in payload and payload.get("meta", {}).get("qat_baked", False):
                sd = TQ.strip_prefix_from_state_dict(payload["state_dict"])
                model.load_state_dict(sd, strict=True)
                qat_scales = payload["meta"].get("scales", None)
                print(f"  [weights] source=artifact_int{nb}_qat (baked-float format, "
                      f"with {len(qat_scales) if qat_scales else 0} exact scales)")
            else:
                TQ.load_quantized_into_model(
                    model, args.dataset, arch, nb, qtag="qat", map_location=device,
                )
                print(f"  [weights] source=artifact_int{nb}_qat (legacy PTQ-payload format)")
        else:
            print(f"  [warn] no int{nb}_qat checkpoint found, falling back to float32")
            TQ.maybe_load_float32_or_pretrained(
                model, args.dataset, arch,
                use_pretrained=bool(args.use_pretrained), map_location=device)
    else:
        TQ.maybe_load_float32_or_pretrained(
            model, args.dataset, arch,
            use_pretrained=bool(args.use_pretrained), map_location=device)

    flip_domain = "float"
    qview: Optional[QuantView] = None
    if fmt.startswith("int"):
        nbits = int(fmt[3:])
        qview = QuantView(model, num_bits=nbits, precomputed_scales=qat_scales)
        qview = QuantView(model, num_bits=nbits)
        flip_domain = f"int{nbits}"
    print(f"  [flip-domain] {flip_domain}")

    print(f"  [eval clean] ...")
    acc_clean = evaluate_top1(model, test_loader, device,
                               max_batches=args.eval_max_batches)
    print(f"  acc_clean = {acc_clean:.2f}%")

    def attack_then_eval(picks):
        if qview is not None:
            undos = []
            for lname, idx in picks:
                if lname in qview.entries:
                    undos.append(qview.flip_int(model, lname, idx,
                                                bit_pos=args.bit_position))
            acc = evaluate_top1(model, test_loader, device,
                                max_batches=args.eval_max_batches)
            qview.undo(model, undos)
        else:
            undos = apply_flips_float(model, picks, bit_pos=args.bit_position)
            acc = evaluate_top1(model, test_loader, device,
                                max_batches=args.eval_max_batches)
            undo_flips_float(model, undos)
        return acc

    def empirical_one(lname, idx):
        """Flip ONE weight, return delta-loss vs base, restore."""
        if qview is not None:
            if lname not in qview.entries:
                return 0.0
            undo = qview.flip_int(model, lname, idx, bit_pos=args.bit_position)
            new_loss = avg_loss(model, test_loader, device,
                                max_batches=args.empirical_max_batches)
            qview.undo(model, [undo])
        else:
            undos = apply_flips_float(model, [(lname, idx)], bit_pos=args.bit_position)
            new_loss = avg_loss(model, test_loader, device,
                                max_batches=args.empirical_max_batches)
            undo_flips_float(model, undos)
        return new_loss - base_loss_for_emp[0]

    base_loss_for_emp = [None]  # mutable closure cell

    if args.empirical:
        base_loss_for_emp[0] = avg_loss(model, test_loader, device,
                                        max_batches=args.empirical_max_batches)
        print(f"  [empirical] scoring {len(rows)} candidates by leave-one-out flip "
              f"(max_batches={args.empirical_max_batches}) ...")
        for r in rows:
            r["empirical"] = float(empirical_one(r["layer"], r["flat_idx"]))
        print(f"  [empirical] done.  top-3 scores: "
              f"{sorted([r['empirical'] for r in rows], reverse=True)[:3]}")

    rng = np.random.default_rng(args.seed)
    out_rows = []
    for K in args.k_list:
        if K > len(rows): K = len(rows)

        picks_t = [(r["layer"], r["flat_idx"])
                   for r in pick_top_k(rows, K, by="taylor")]
        acc_t = attack_then_eval(picks_t)

        picks_m = [(r["layer"], r["flat_idx"])
                   for r in pick_top_k(rows, K, by="magnitude")]
        acc_m = attack_then_eval(picks_m)

        # ---- FISHER attack ----
        picks_f = [(r["layer"], r["flat_idx"])
                   for r in pick_top_k(rows, K, by="fisher")]
        acc_f = attack_then_eval(picks_f)

        # ---- EMPIRICAL attack ----
        if args.empirical:
            picks_e = [(r["layer"], r["flat_idx"])
                       for r in pick_top_k(rows, K, by="empirical")]
            acc_e = attack_then_eval(picks_e)
        else:
            acc_e = None

        rand_accs = []
        for _ in range(args.random_trials):
            picks_r = pick_random_k(model, K, rng)
            rand_accs.append(attack_then_eval(picks_r))
        acc_r_mean = float(np.mean(rand_accs))
        acc_r_std  = float(np.std(rand_accs))

        row = {
            "arch": arch, "format": fmt, "K": K,
            "bit_position": args.bit_position,
            "flip_domain": flip_domain,
            "acc_clean": acc_clean,
            "acc_taylor":    acc_t,
            "acc_magnitude": acc_m,
            "acc_fisher":    acc_f,
            "acc_random_mean": acc_r_mean, "acc_random_std": acc_r_std,
            "drop_taylor":      acc_clean - acc_t,
            "drop_magnitude":   acc_clean - acc_m,
            "drop_fisher":      acc_clean - acc_f,
            "drop_random_mean": acc_clean - acc_r_mean,
        }
        if args.empirical:
            row["acc_empirical"]  = acc_e
            row["drop_empirical"] = acc_clean - acc_e
        out_rows.append(row)
        emp_msg = f"  emp={acc_e:6.2f}%" if args.empirical else ""
        print(f"  K={K:>5}  taylor={acc_t:6.2f}%  "
              f"mag={acc_m:6.2f}%  fisher={acc_f:6.2f}%{emp_msg}  "
              f"random={acc_r_mean:6.2f}±{acc_r_std:.2f}%   "
              f"|  taylor-drop={row['drop_taylor']:6.2f}pp  "
              f"fisher-drop={row['drop_fisher']:6.2f}pp  "
              f"random-drop={row['drop_random_mean']:6.2f}pp")
    return out_rows


# =========================================================
# CLI
# =========================================================

def main():
    p = argparse.ArgumentParser("Bit-flip validation (INTEGER-SIDE flip for INT8/INT4)")
    p.add_argument("--dataset", required=True,
                   choices=["CIFAR10","CIFAR100","MNIST","IMAGENET"])
    p.add_argument("--archs", nargs="+", required=True)
    p.add_argument("--formats", nargs="+", default=["float32","int8","int4"])
    p.add_argument("--data-root", default="./data")
    p.add_argument("--imagenet-root", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--use-pretrained", type=int, default=0)
    p.add_argument("--weights", default="")
    p.add_argument("--qtag", default="ptq", choices=["ptq", "qat"],
                   help="Which quantization checkpoint variant to load when "
                        "fmt is int8/int4. 'ptq' (default) quantizes float32 "
                        "weights on-the-fly. 'qat' loads the int{N}_qat "
                        "checkpoint produced by qat_train.py.")
    p.add_argument("--qtag-suffix", default="",
                   help="Optional suffix on the QAT checkpoint name, matching "
                        "qat_train.py's --qtag-suffix. E.g. --qtag-suffix _lsq "
                        "loads model_int4_qat_lsq.pth instead of "
                        "model_int4_qat.pth. Default empty.")
    p.add_argument("--in-dir", default="artifacts/sensitivity")
    p.add_argument("--top-layers", type=int, default=5)
    p.add_argument("--top-per-layer", type=int, default=200)
    p.add_argument("--layer-metric", default="grad_norm")
    p.add_argument("--k-list", nargs="+", type=int,
                   default=[1, 5, 10, 50, 100, 500, 1000])
    p.add_argument("--bit-position", default="sign")
    p.add_argument("--random-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-max-batches", type=int, default=None)
    p.add_argument("--empirical", type=int, default=0,
                   help="If 1, also run an EMPIRICAL attack: score each "
                        "candidate weight by actually flipping it and "
                        "measuring the loss change. Costs ~1000 forward "
                        "passes per (arch, format).")
    p.add_argument("--empirical-max-batches", type=int, default=2,
                   help="Number of batches per empirical scoring step.")
    p.add_argument("--out-dir", default="artifacts/sensitivity")
    args = p.parse_args()
    args.dataset = args.dataset.upper()

    print(f"[device]    {args.device}")
    print(f"[archs]     {args.archs}")
    print(f"[formats]   {args.formats}")
    print(f"[k-list]    {args.k_list}")
    print(f"[bit]       {args.bit_position}  (INTEGER-SIDE flip for INT formats)")
    print(f"[random×]   {args.random_trials} trials")

    all_rows = []
    t0 = time.time()
    for arch in args.archs:
        for fmt in args.formats:
            print(f"\n=== {arch}  {fmt} ===")
            try:
                all_rows.extend(run_experiment_one(arch, fmt, args))
            except Exception as e:
                print(f"  [!! error] {type(e).__name__}: {e}")
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")

    if not all_rows:
        print("[validate] no results"); return

    out_csv = os.path.join(
        args.out_dir,
        f"validate_bitflip_INT_{args.dataset.lower()}_{args.bit_position}_L{args.top_layers}xN{args.top_per_layer}_{args.qtag}.csv"
    )
    cols = ["arch","format","K","bit_position","flip_domain",
            "acc_clean",
            "acc_taylor","acc_magnitude","acc_fisher",
            "acc_random_mean","acc_random_std",
            "drop_taylor","drop_magnitude","drop_fisher","drop_random_mean"]
    has_emp = any("acc_empirical" in r for r in all_rows)
    if has_emp:
        cols += ["acc_empirical","drop_empirical"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in all_rows: w.writerow([r.get(c, "") for c in cols])
    print(f"[saved] {out_csv}")

    print("\n" + "=" * 110)
    print("VALIDATION SUMMARY  (INTEGER-SIDE flip for INT formats; acc drop in pp)")
    print("=" * 110)
    if has_emp:
        print(f"{'arch':<18} {'fmt':<8} {'domain':<8} {'K':>6}  "
              f"{'taylor':>10}  {'magnitude':>11}  {'fisher':>10}  "
              f"{'empirical':>11}  {'random':>10}")
    else:
        print(f"{'arch':<18} {'fmt':<8} {'domain':<8} {'K':>6}  "
              f"{'taylor':>10}  {'magnitude':>11}  {'fisher':>10}  {'random':>10}")
    print("-" * 110)
    for r in all_rows:
        if has_emp and "drop_empirical" in r:
            print(f"{r['arch']:<18} {r['format']:<8} {r['flip_domain']:<8} {r['K']:>6}  "
                  f"{r['drop_taylor']:>9.2f}pp  "
                  f"{r['drop_magnitude']:>10.2f}pp  "
                  f"{r['drop_fisher']:>9.2f}pp  "
                  f"{r['drop_empirical']:>10.2f}pp  "
                  f"{r['drop_random_mean']:>9.2f}pp")
        else:
            print(f"{r['arch']:<18} {r['format']:<8} {r['flip_domain']:<8} {r['K']:>6}  "
                  f"{r['drop_taylor']:>9.2f}pp  "
                  f"{r['drop_magnitude']:>10.2f}pp  "
                  f"{r['drop_fisher']:>9.2f}pp  "
                  f"{r['drop_random_mean']:>9.2f}pp")
    print("=" * 110)


if __name__ == "__main__":
    main()
