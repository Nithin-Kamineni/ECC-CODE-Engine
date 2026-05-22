# validate_bitflip.py
# Bit-flip validation experiment: take the top weights selected by
# layer_then_weight.py and actually flip bits on them, measure the
# accuracy drop, and compare against random-baseline.
#
# Inputs : the layer_then_weight_*.csv files produced by layer_then_weight.py
# Outputs: a validation CSV per (arch, format) with rows
#          K, accuracy_clean, accuracy_taylor, accuracy_random, drop_taylor,
#          drop_random
#          and the final summary table.
#
# Author: Habibur Rahaman, University of Florida, ECE Department
#
# Examples:
#   # CIFAR-10 ResNet-18, validate float / INT8 / INT4 versions:
#   python validate_bitflip.py --dataset CIFAR10 --archs resnet18 \
#       --formats float32 int8 int4 \
#       --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
#       --k-list 1 5 10 50 100 500 1000 \
#       --bit-position sign \
#       --random-trials 3
#
#   # ImageNet 5-arch sweep, ALL formats, all K's:
#   python validate_bitflip.py --dataset IMAGENET \
#       --imagenet-root ./imagenet-val --use-pretrained 1 \
#       --archs resnet18 resnet50 vgg16 mobilenet_v2 efficientnet_b0 \
#       --formats float32 int8 int4 \
#       --top-layers 5 --top-per-layer 200 --layer-metric grad_norm \
#       --k-list 1 10 100 500 1000 \
#       --bit-position sign --random-trials 3 --max-batches 8

from __future__ import annotations
import os, csv, time, argparse, importlib.util, sys, struct, random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# Bring in test-Quantizer.py
# =========================================================
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
# Bit-flip primitive
# =========================================================

def _flip_float32_bit(value: float, bit_pos) -> float:
    """
    bit_pos can be:
      - "sign"     : flip the sign bit (bit 31)
      - "msb_exp"  : flip the most significant exponent bit (bit 30)
      - "lsb_mant" : flip the least significant mantissa bit (bit 0)
      - integer 0..31 : flip that specific bit
    Returns the new float value after a single bit flip.
    """
    raw = struct.unpack(">I", struct.pack(">f", float(value)))[0]
    if bit_pos == "sign":
        bit = 31
    elif bit_pos == "msb_exp":
        bit = 30
    elif bit_pos == "lsb_mant":
        bit = 0
    else:
        bit = int(bit_pos)
    raw ^= (1 << bit)
    return struct.unpack(">f", struct.pack(">I", raw))[0]


def flip_weight_inplace(p: torch.Tensor, flat_idx: int, bit_pos="sign"):
    """Flip one bit in p.data at the given flat index. Returns (old_val, new_val)."""
    flat = p.data.view(-1)
    old = float(flat[flat_idx].item())
    new = _flip_float32_bit(old, bit_pos)
    flat[flat_idx] = new
    return old, new


def restore_weight_inplace(p: torch.Tensor, flat_idx: int, old_val: float):
    p.data.view(-1)[flat_idx] = old_val


# =========================================================
# Helpers
# =========================================================

def _iter_weight_params(model: nn.Module):
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        lname = name.lower()
        if lname.endswith(".bias") or lname.endswith("_bias"): continue
        if "bn" in lname or ".norm" in lname or "running_" in lname: continue
        if p.ndim in (2, 4) and p.dtype.is_floating_point:
            yield name, p


def name_to_param(model):
    return {n: p for n, p in _iter_weight_params(model)}


def quantize_model_inplace(model: nn.Module, num_bits: int):
    sd = model.state_dict()
    new_sd = {}
    with torch.no_grad():
        for k, p in sd.items():
            if not p.dtype.is_floating_point:
                new_sd[k] = p; continue
            q_tensor, sinfo = TQ.quantize_param_smart(k, p, num_bits)
            if sinfo is None:
                new_sd[k] = p; continue
            qtype = sinfo.get("type", None)
            if qtype == "per_tensor":
                w_dq = TQ.dequantize_tensor(q_tensor, sinfo["scale"]) \
                          .to(p.device).to(p.dtype)
            elif qtype == "per_channel":
                w_dq = TQ.dequantize_per_channel_conv(
                    q_tensor, sinfo["scales"].to(p.device)
                ).to(p.device).to(p.dtype)
            else:
                w_dq = p
            new_sd[k] = w_dq
    model.load_state_dict(new_sd, strict=True)


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


# =========================================================
# Read selections from layer_then_weight CSV
# =========================================================

def read_top_weights(csv_path: str) -> List[dict]:
    """
    Returns rows in CSV order. Caller decides how many to flip (top-K of these).
    layer_then_weight.py orders them by layer_rank ASC, in_layer_rank ASC.
    For "top-K most sensitive globally" we re-sort by taylor desc.
    """
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


def pick_top_k(rows: List[dict], k: int, by: str = "taylor") -> List[dict]:
    sorted_rows = sorted(rows, key=lambda r: r[by], reverse=True)
    return sorted_rows[:k]


def pick_random_k(model, k: int, rng: np.random.Generator) -> List[Tuple[str, int]]:
    """Pick k random (layer_name, flat_idx) pairs uniformly from all weight params."""
    pool = []
    for name, p in _iter_weight_params(model):
        pool.append((name, p.numel()))
    sizes = np.array([n for _, n in pool], dtype=np.int64)
    total = sizes.sum()
    cum = np.cumsum(sizes)
    chosen = []
    picks = rng.integers(low=0, high=total, size=k)
    for g in picks:
        layer_idx = int(np.searchsorted(cum, g, side="right"))
        local = int(g - (cum[layer_idx - 1] if layer_idx > 0 else 0))
        chosen.append((pool[layer_idx][0], local))
    return chosen


# =========================================================
# Apply / undo a bit-flip attack
# =========================================================

def apply_flips(model, picks: List[Tuple[str, int]], bit_pos="sign"):
    """picks = list of (layer_name, flat_idx). Returns undo info."""
    n2p = name_to_param(model)
    undos = []
    for lname, idx in picks:
        if lname not in n2p:
            continue
        old, _ = flip_weight_inplace(n2p[lname], idx, bit_pos=bit_pos)
        undos.append((lname, idx, old))
    return undos


def undo_flips(model, undos):
    n2p = name_to_param(model)
    for lname, idx, old in undos:
        if lname in n2p:
            restore_weight_inplace(n2p[lname], idx, old)


# =========================================================
# Per (arch, format) experiment
# =========================================================

def run_experiment_one(arch: str, fmt: str, args) -> List[dict]:
    """
    Returns list of result rows for one (arch, format).
    Each row: K, acc_clean, acc_taylor, acc_random_mean, acc_random_std,
              drop_taylor, drop_random_mean.
    """
    device = TQ.pick_device(args.device, local_rank=0)

    # --- locate the layer_then_weight CSV ---
    base_tag = f"{args.dataset.lower()}_{arch.lower()}"
    csv_path = os.path.join(
        args.in_dir,
        f"layer_then_weight_{base_tag}_{fmt}_L{args.top_layers}xN{args.top_per_layer}_{args.layer_metric}.csv"
    )
    if not os.path.isfile(csv_path):
        print(f"  [skip] missing {csv_path}")
        return []
    rows = read_top_weights(csv_path)
    if not rows:
        print(f"  [skip] empty {csv_path}")
        return []
    print(f"  [read] {csv_path}  ({len(rows)} candidate weights)")

    # --- build dataset / loader ---
    _, _, nc = TQ.get_datasets(args.dataset, args.data_root,
                                args.imagenet_root or None)
    _, test_loader, _, _ = TQ.get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers,
        dist_mode=False, imagenet_root=args.imagenet_root or None)

    # --- build model and load weights ---
    model = TQ.build_model(arch, nc, use_pretrained=bool(args.use_pretrained)).to(device)
    if args.weights:
        payload = torch.load(args.weights, map_location=device)
        sd = payload.get("state_dict", payload)
        sd = TQ.strip_prefix_from_state_dict(sd)
        model.load_state_dict(sd, strict=True)
    else:
        TQ.maybe_load_float32_or_pretrained(
            model, args.dataset, arch,
            use_pretrained=bool(args.use_pretrained), map_location=device)

    if fmt.startswith("int"):
        nbits = int(fmt[3:])
        quantize_model_inplace(model, num_bits=nbits)

    # --- baseline accuracy ---
    print(f"  [eval clean] ...")
    acc_clean = evaluate_top1(model, test_loader, device,
                               max_batches=args.eval_max_batches)
    print(f"  acc_clean = {acc_clean:.2f}%")

    # --- sweep K ---
    rng = np.random.default_rng(args.seed)
    out_rows = []
    for K in args.k_list:
        if K > len(rows):
            K = len(rows)

        # ---- TAYLOR attack ----
        picks_t = [(r["layer"], r["flat_idx"])
                   for r in pick_top_k(rows, K, by="taylor")]
        undos = apply_flips(model, picks_t, bit_pos=args.bit_position)
        acc_t = evaluate_top1(model, test_loader, device,
                              max_batches=args.eval_max_batches)
        undo_flips(model, undos)

        # ---- MAGNITUDE attack (additional baseline) ----
        picks_m = [(r["layer"], r["flat_idx"])
                   for r in pick_top_k(rows, K, by="magnitude")]
        undos = apply_flips(model, picks_m, bit_pos=args.bit_position)
        acc_m = evaluate_top1(model, test_loader, device,
                              max_batches=args.eval_max_batches)
        undo_flips(model, undos)

        # ---- RANDOM attack (averaged over trials) ----
        rand_accs = []
        for _ in range(args.random_trials):
            picks_r = pick_random_k(model, K, rng)
            undos = apply_flips(model, picks_r, bit_pos=args.bit_position)
            ar = evaluate_top1(model, test_loader, device,
                               max_batches=args.eval_max_batches)
            undo_flips(model, undos)
            rand_accs.append(ar)
        acc_r_mean = float(np.mean(rand_accs))
        acc_r_std  = float(np.std(rand_accs))

        row = {
            "arch": arch, "format": fmt, "K": K,
            "bit_position": args.bit_position,
            "acc_clean":         acc_clean,
            "acc_taylor":        acc_t,
            "acc_magnitude":     acc_m,
            "acc_random_mean":   acc_r_mean,
            "acc_random_std":    acc_r_std,
            "drop_taylor":       acc_clean - acc_t,
            "drop_magnitude":    acc_clean - acc_m,
            "drop_random_mean":  acc_clean - acc_r_mean,
        }
        out_rows.append(row)
        print(f"  K={K:>5}  taylor={acc_t:6.2f}%  "
              f"mag={acc_m:6.2f}%  random={acc_r_mean:6.2f}±{acc_r_std:.2f}%   "
              f"|  taylor-drop={row['drop_taylor']:6.2f}pp  "
              f"random-drop={row['drop_random_mean']:6.2f}pp")
    return out_rows


# =========================================================
# CLI
# =========================================================

def main():
    p = argparse.ArgumentParser("Bit-flip validation of layer-then-weight selections")
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

    # which CSVs to consume
    p.add_argument("--in-dir", default="artifacts/sensitivity")
    p.add_argument("--top-layers", type=int, default=5,
                   help="Must match what layer_then_weight.py was run with.")
    p.add_argument("--top-per-layer", type=int, default=200,
                   help="Must match what layer_then_weight.py was run with.")
    p.add_argument("--layer-metric", default="grad_norm")

    # attack settings
    p.add_argument("--k-list", nargs="+", type=int,
                   default=[1, 5, 10, 50, 100, 500, 1000])
    p.add_argument("--bit-position", default="sign",
                   help="Which bit to flip per weight: sign / msb_exp / lsb_mant / 0..31")
    p.add_argument("--random-trials", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval-max-batches", type=int, default=None,
                   help="Cap eval batches for speed (None = full test set).")

    p.add_argument("--out-dir", default="artifacts/sensitivity")
    args = p.parse_args()
    args.dataset = args.dataset.upper()

    print(f"[device]    {args.device}")
    print(f"[archs]     {args.archs}")
    print(f"[formats]   {args.formats}")
    print(f"[k-list]    {args.k_list}")
    print(f"[bit]       {args.bit_position}")
    print(f"[random×]   {args.random_trials} trials")

    all_rows = []
    t0 = time.time()
    for arch in args.archs:
        for fmt in args.formats:
            print(f"\n=== {arch}  {fmt} ===")
            try:
                rows = run_experiment_one(arch, fmt, args)
                all_rows.extend(rows)
            except Exception as e:
                print(f"  [!! error] {type(e).__name__}: {e}")
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")

    if not all_rows:
        print("[validate] no results")
        return

    out_csv = os.path.join(
        args.out_dir,
        f"validate_bitflip_{args.dataset.lower()}_{args.bit_position}_L{args.top_layers}xN{args.top_per_layer}.csv"
    )
    cols = ["arch","format","K","bit_position",
            "acc_clean","acc_taylor","acc_magnitude",
            "acc_random_mean","acc_random_std",
            "drop_taylor","drop_magnitude","drop_random_mean"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for r in all_rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"[saved] {out_csv}")

    # ----- final summary table -----
    print("\n" + "=" * 110)
    print("VALIDATION SUMMARY  (acc drop in percentage points;  larger = stronger attack)")
    print("=" * 110)
    print(f"{'arch':<18} {'fmt':<8} {'K':>6}  "
          f"{'taylor':>10}  {'magnitude':>11}  {'random':>10}")
    print("-" * 110)
    for r in all_rows:
        print(f"{r['arch']:<18} {r['format']:<8} {r['K']:>6}  "
              f"{r['drop_taylor']:>9.2f}pp  "
              f"{r['drop_magnitude']:>10.2f}pp  "
              f"{r['drop_random_mean']:>9.2f}pp")
    print("=" * 110)


if __name__ == "__main__":
    main()
