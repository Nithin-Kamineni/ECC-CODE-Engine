# sensitivity.py
# Weight sensitivity analysis for neural networks.
# Companion to test-Quantizer.py
#
# Author: Habibur Rahaman, University of Florida, ECE Department
#
#   1) Loads a (float32) checkpoint.
#   2) Optionally also produces INT8 / INT4 versions of the same model
#      via test-Quantizer.py's PTQ pipeline.
#   3) For each version, computes per-weight sensitivity via:
#        magnitude   : |w|
#        grad_abs    : |dL/dw|
#        taylor      : |w * dL/dw|     (Molchanov)
#        fisher      : E[(dL/dw)^2]    (diagonal Fisher)
#        hessian     : |diag(H)|       (Hutchinson)
#   4) Optionally computes per-weight quantization error
#        quant_err_<bits> = |w_float - w_dequant|
#      and joins it into the dump.
#   5) Aggregates per-layer concentration metrics:
#        gini, normalized entropy, top-{0.1,1,5,10}% coverage.
#   6) Optionally dumps individual weights (3 modes):
#        --dump-weights                     -> every weight (huge)
#        --dump-weights --top-k N           -> top-N weights GLOBALLY
#        --dump-weights --top-layers K --top-per-layer M
#                                            -> top-K layers, top-M weights each
#
# Examples:
#   # Float-only basic run:
#   python sensitivity.py --dataset CIFAR10 --arch resnet18 \
#       --methods magnitude grad_abs taylor fisher --max-batches 8
#
#   # Float + INT8 + INT4, with quant-error joined into per-weight dump:
#   python sensitivity.py --dataset CIFAR10 --arch resnet18 \
#       --methods magnitude grad_abs taylor fisher \
#       --quantize-bits 8 4 --quant-error \
#       --dump-weights --top-layers 5 --top-per-layer 200 \
#       --rank-by taylor --layer-rank-by mean

from __future__ import annotations
import os, math, json, time, argparse, importlib.util, sys, csv, copy
from typing import Optional, Dict, Any, List, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# Bring in test-Quantizer.py despite the dash in the filename
# =========================================================
def _load_test_quantizer():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "test-Quantizer.py"),
        os.path.join(here, "test_quantizer.py"),
        "test-Quantizer.py",
        "test_quantizer.py",
    ]
    for path in candidates:
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("test_quantizer", path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules["test_quantizer"] = mod
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError(
        "Could not locate test-Quantizer.py next to sensitivity.py."
    )

TQ = _load_test_quantizer()


# =========================================================
# Helpers
# =========================================================

def _iter_weight_params(model: nn.Module):
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        lname = name.lower()
        if lname.endswith(".bias") or lname.endswith("_bias"):
            continue
        if "bn" in lname or ".norm" in lname or "running_" in lname:
            continue
        if p.ndim in (2, 4) and p.dtype.is_floating_point:
            yield name, p


def _zero_grads(model: nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()


# =========================================================
# Quantization helpers (delegating to test-Quantizer)
# =========================================================

def quantize_model_inplace(model: nn.Module, num_bits: int):
    """
    Apply per-channel/per-tensor PTQ to all weights of `model` IN PLACE,
    using test-Quantizer's quantize_param_smart + dequantize routines.
    Returns dict mapping param-name -> quantization error tensor (|w - w_dq|).
    Biases / BN stats are left untouched.
    """
    errors: Dict[str, torch.Tensor] = {}
    sd = model.state_dict()
    new_sd = {}
    with torch.no_grad():
        for k, p in sd.items():
            if not p.dtype.is_floating_point:
                new_sd[k] = p
                continue
            q_tensor, sinfo = TQ.quantize_param_smart(k, p, num_bits)
            if sinfo is None:
                new_sd[k] = p
                continue
            qtype = sinfo.get("type", None)
            if qtype == "per_tensor":
                w_dq = TQ.dequantize_tensor(q_tensor, sinfo["scale"]).to(p.device).to(p.dtype)
            elif qtype == "per_channel":
                w_dq = TQ.dequantize_per_channel_conv(
                    q_tensor, sinfo["scales"].to(p.device)
                ).to(p.device).to(p.dtype)
            else:
                w_dq = p
            new_sd[k] = w_dq
            errors[k] = (p - w_dq).abs().detach().clone()
    model.load_state_dict(new_sd, strict=True)
    return errors


# =========================================================
# Sensitivity scoring methods
# =========================================================

def grad_pass(model, loader, device, max_batches=8, criterion=None) -> int:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    _zero_grads(model)
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.enable_grad():
            criterion(model(x), y).backward()
        n += 1
    if n > 0:
        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(n)
    return n


def fisher_diagonal(model, loader, device, max_batches=8, criterion=None
                    ) -> Dict[str, torch.Tensor]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    fisher: Dict[str, torch.Tensor] = {}
    n = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        _zero_grads(model)
        with torch.enable_grad():
            criterion(model(x), y).backward()
        for name, p in _iter_weight_params(model):
            if p.grad is None: continue
            g2 = p.grad.detach() ** 2
            fisher[name] = g2.clone() if name not in fisher else fisher[name] + g2
        n += 1
    for name in fisher:
        fisher[name] /= max(1, n)
    return fisher


def hessian_diag_hutchinson(model, loader, device, max_batches=2, n_samples=1,
                            criterion=None) -> Dict[str, torch.Tensor]:
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    pairs = list(_iter_weight_params(model))
    names  = [n for n, _ in pairs]
    params = [p for _, p in pairs]
    accum = {n: torch.zeros_like(p) for n, p in pairs}
    n_total = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        for _ in range(n_samples):
            _zero_grads(model)
            with torch.enable_grad():
                loss = criterion(model(x), y)
                grads = torch.autograd.grad(loss, params, create_graph=True)
                vs = [torch.randint_like(p, high=2, dtype=p.dtype) * 2 - 1 for p in params]
                gv = sum((g * v).sum() for g, v in zip(grads, vs))
                Hv = torch.autograd.grad(gv, params, retain_graph=False)
                for name, v, hv in zip(names, vs, Hv):
                    accum[name] += (v * hv).detach()
            n_total += 1
    for name in accum:
        accum[name] = (accum[name] / max(1, n_total)).abs()
    return accum


# =========================================================
# Concentration metrics
# =========================================================

def gini_coefficient(x: torch.Tensor) -> float:
    v = x.detach().abs().flatten().cpu().numpy().astype(np.float64)
    if v.size == 0 or v.sum() == 0: return 0.0
    v.sort()
    n = v.size
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float(((2 * idx - n - 1) * v).sum() / (n * v.sum()))


def normalized_entropy(x: torch.Tensor) -> float:
    v = x.detach().abs().flatten().cpu().numpy().astype(np.float64)
    s = v.sum()
    if s == 0 or v.size <= 1: return 0.0
    p = v / s; p = p[p > 0]
    H = -(p * np.log(p)).sum()
    Hmax = math.log(v.size)
    return float(H / Hmax) if Hmax > 0 else 0.0


def topk_coverage(x: torch.Tensor, fractions=(0.001, 0.01, 0.05, 0.10)
                  ) -> Dict[str, float]:
    v = x.detach().abs().flatten()
    total = v.sum().item()
    out = {}
    for f in fractions:
        k = max(1, int(v.numel() * f))
        topk = torch.topk(v, k).values.sum().item()
        out[f"top_{f*100:g}pct"] = topk / total if total > 0 else 0.0
    return out


# =========================================================
# Driver
# =========================================================

def compute_all_sensitivity(model, loader, device, methods, max_batches=8
                            ) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    if "magnitude" in methods:
        out["magnitude"] = {n: p.detach().abs().clone()
                            for n, p in _iter_weight_params(model)}
    if "grad_abs" in methods or "taylor" in methods:
        n_used = grad_pass(model, loader, device, max_batches=max_batches)
        print(f"  [grad_pass] used {n_used} batches")
        if "grad_abs" in methods:
            out["grad_abs"] = {n: p.grad.detach().abs().clone()
                               for n, p in _iter_weight_params(model)
                               if p.grad is not None}
        if "taylor" in methods:
            out["taylor"] = {n: (p.detach() * p.grad.detach()).abs().clone()
                             for n, p in _iter_weight_params(model)
                             if p.grad is not None}
    if "fisher" in methods:
        out["fisher"] = fisher_diagonal(model, loader, device, max_batches=max_batches)
        print(f"  [fisher] done")
    if "hessian" in methods:
        out["hessian"] = hessian_diag_hutchinson(
            model, loader, device,
            max_batches=min(2, max_batches), n_samples=1)
        print(f"  [hessian] done")
    return out


def summarize(scores_per_method, extra_per_layer=None
              ) -> "OrderedDict[str, Dict[str, float]]":
    summary: "OrderedDict[str, Dict[str, float]]" = OrderedDict()
    for method, layers in scores_per_method.items():
        for layer, s in layers.items():
            v = s.detach().flatten().cpu()
            row = summary.setdefault(layer, {})
            row["numel"] = int(v.numel())
            row[f"{method}_mean"] = float(v.mean())
            row[f"{method}_max"]  = float(v.max())
            row[f"{method}_sum"]  = float(v.sum())
            row[f"{method}_std"]  = float(v.std())
            row[f"{method}_gini"] = gini_coefficient(s)
            row[f"{method}_ent"]  = normalized_entropy(s)
            for k, val in topk_coverage(s).items():
                row[f"{method}_{k}"] = val
    if extra_per_layer:
        for layer, extras in extra_per_layer.items():
            row = summary.setdefault(layer, {})
            for k, v in extras.items():
                row[k] = v
    return summary


def save_summary(summary, out_dir, tag) -> Tuple[str, str]:
    os.makedirs(out_dir, exist_ok=True)
    js_path  = os.path.join(out_dir, f"sensitivity_{tag}.json")
    csv_path = os.path.join(out_dir, f"sensitivity_{tag}.csv")
    with open(js_path, "w") as f:
        json.dump(summary, f, indent=2)
    if summary:
        all_keys = set()
        for row in summary.values():
            all_keys.update(row.keys())
        keys = sorted(all_keys)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer"] + keys)
            for layer, row in summary.items():
                w.writerow([layer] + [row.get(k, "") for k in keys])
    return js_path, csv_path


# =========================================================
# Per-weight dump (3 modes)
# =========================================================

def _gather_layer_data(model, scores_per_method, extra_per_weight=None):
    """
    Returns list of (name, w_flat_cpu, {method:score_flat_cpu}, shape).
    `extra_per_weight` is an extra dict-of-dicts: extra[name][col]=tensor.
    Those columns are merged into per_m so they become extra columns in CSV/NPZ.
    """
    methods = list(scores_per_method.keys())
    if extra_per_weight:
        for name in extra_per_weight:
            for col in extra_per_weight[name]:
                if col not in methods:
                    methods.append(col)
    layers = []
    for name, p in _iter_weight_params(model):
        per_m = {}
        for m in scores_per_method:
            if name in scores_per_method[m]:
                per_m[m] = scores_per_method[m][name].detach().cpu().flatten()
        if extra_per_weight and name in extra_per_weight:
            for col, t in extra_per_weight[name].items():
                per_m[col] = t.detach().cpu().flatten()
        layers.append((name, p.detach().cpu().flatten(), per_m, tuple(p.shape)))
    return layers, methods


def _layer_aggregate(score_flat: torch.Tensor, agg: str) -> float:
    if agg == "mean": return float(score_flat.mean())
    if agg == "max":  return float(score_flat.max())
    if agg == "sum":  return float(score_flat.sum())
    raise ValueError(f"unknown layer aggregate: {agg}")


def dump_per_weight(model, scores_per_method, out_dir, tag, *,
                    mode: str,
                    top_k_global=None, top_layers=None, top_per_layer=None,
                    rank_by="taylor", layer_rank_by="mean",
                    extra_per_weight=None,
                    write_csv=True, write_npz=True) -> List[str]:
    os.makedirs(out_dir, exist_ok=True)
    created: List[str] = []
    layers, methods = _gather_layer_data(model, scores_per_method, extra_per_weight)
    if rank_by not in methods:
        raise ValueError(f"--rank-by={rank_by} not among computed columns {methods}")
    print(f"[dump-weights] mode={mode}  rank_by={rank_by}  cols={methods}")

    # ---------- ALL ----------
    if mode == "all":
        total = sum(t.numel() for _, t, _, _ in layers)
        print(f"[dump-weights] writing ALL {total:,} weights -- this may be large")
        if write_csv:
            csv_path = os.path.join(out_dir, f"perweight_{tag}_full.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["layer", "flat_idx", "multi_idx", "w"] + methods)
                for lname, w_flat, per_m, shape in layers:
                    n = w_flat.numel(); w_np = w_flat.numpy()
                    arr = {m: per_m[m].numpy() for m in methods if m in per_m}
                    for i in range(n):
                        multi = np.unravel_index(i, shape)
                        row = [lname, i, "(" + ",".join(map(str, multi)) + ")",
                               float(w_np[i])]
                        row += [float(arr[m][i]) if m in arr else "" for m in methods]
                        w.writerow(row)
            print(f"[dump-weights] wrote {csv_path}"); created.append(csv_path)
        if write_npz:
            npz_path = os.path.join(out_dir, f"perweight_{tag}_full.npz")
            payload = {}
            for lname, w_flat, per_m, shape in layers:
                key = lname.replace(".", "_")
                payload[f"{key}__w"] = w_flat.numpy()
                payload[f"{key}__shape"] = np.array(shape, dtype=np.int64)
                for m, s in per_m.items():
                    payload[f"{key}__{m}"] = s.numpy()
            np.savez(npz_path, **payload)
            print(f"[dump-weights] wrote {npz_path}"); created.append(npz_path)
        return created

    # ---------- GLOBAL TOP-K ----------
    if mode == "global_topk":
        if top_k_global is None:
            raise ValueError("global_topk mode requires top_k_global")
        all_scores=[]; all_origins=[]
        for li, (_, _, per_m, _) in enumerate(layers):
            s = per_m[rank_by].numpy()
            all_scores.append(s)
            all_origins.append(np.full(s.shape, li, dtype=np.int32))
        flat_scores = np.concatenate(all_scores)
        flat_layer  = np.concatenate(all_origins)
        flat_local  = np.concatenate([np.arange(s.size, dtype=np.int64)
                                      for s in all_scores])
        K = min(top_k_global, flat_scores.size)
        part = np.argpartition(-flat_scores, K - 1)[:K]
        order = part[np.argsort(-flat_scores[part])]
        keep_layer = flat_layer[order]; keep_local = flat_local[order]
        print(f"[dump-weights] keeping top-{K} weights GLOBALLY by {rank_by}")
        if write_csv:
            csv_path = os.path.join(out_dir,
                f"perweight_{tag}_global_top{K}_by_{rank_by}.csv")
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["global_rank","layer","flat_idx","multi_idx","w"]+methods)
                for rank, (li, loc) in enumerate(zip(keep_layer.tolist(),
                                                     keep_local.tolist())):
                    lname, w_flat, per_m, shape = layers[li]
                    multi = np.unravel_index(int(loc), shape)
                    row = [rank, lname, int(loc),
                           "(" + ",".join(map(str, multi)) + ")",
                           float(w_flat[int(loc)])]
                    row += [float(per_m[m][int(loc)]) if m in per_m else ""
                            for m in methods]
                    w.writerow(row)
            print(f"[dump-weights] wrote {csv_path}"); created.append(csv_path)
        if write_npz:
            npz_path = os.path.join(out_dir,
                f"perweight_{tag}_global_top{K}_by_{rank_by}.npz")
            kept_w = np.zeros(K, dtype=np.float64)
            kept_layer_name = np.empty(K, dtype=object)
            kept_flat = np.zeros(K, dtype=np.int64)
            kept_methods = {m: np.zeros(K, dtype=np.float64) for m in methods}
            for rank, (li, loc) in enumerate(zip(keep_layer.tolist(),
                                                 keep_local.tolist())):
                lname, w_flat, per_m, _shape = layers[li]
                kept_w[rank] = float(w_flat[int(loc)])
                kept_layer_name[rank] = lname
                kept_flat[rank] = int(loc)
                for m in methods:
                    if m in per_m:
                        kept_methods[m][rank] = float(per_m[m][int(loc)])
            np.savez(npz_path, w=kept_w, layer=kept_layer_name,
                     flat_idx=kept_flat, **kept_methods)
            print(f"[dump-weights] wrote {npz_path}"); created.append(npz_path)
        return created

    # ---------- PER-LAYER TOP-K ----------
    if mode == "per_layer_topk":
        if top_layers is None or top_per_layer is None:
            raise ValueError("per_layer_topk mode needs --top-layers and --top-per-layer")
        layer_scores = []
        for li, (lname, _, per_m, _) in enumerate(layers):
            agg = _layer_aggregate(per_m[rank_by], layer_rank_by)
            layer_scores.append((agg, li, lname))
        layer_scores.sort(reverse=True, key=lambda t: t[0])
        if top_layers > len(layer_scores):
            print(f"[dump-weights] top_layers={top_layers} exceeds layer count={len(layer_scores)}; selecting all {len(layer_scores)} layers")
        chosen = layer_scores[:top_layers]
        print(f"[dump-weights] top-{top_layers} layers by {layer_rank_by}({rank_by}):")
        for rk, (agg, li, lname) in enumerate(chosen):
            print(f"   layer_rank={rk}  {lname:48s}  {layer_rank_by}({rank_by})={agg:.3e}")
        rows = []; npz_per_layer = {}
        for layer_rank, (_agg, li, lname) in enumerate(chosen):
            w_flat = layers[li][1]; per_m = layers[li][2]; shape = layers[li][3]
            scores = per_m[rank_by]
            if top_per_layer > scores.numel():
                print(f"[dump-weights] top_per_layer={top_per_layer} exceeds weight count={scores.numel()} in {lname}; selecting all weights")
            M = min(top_per_layer, scores.numel())
            top_vals, top_idx = torch.topk(scores, M, largest=True, sorted=True)
            for in_layer_rank, loc in enumerate(top_idx.tolist()):
                multi = np.unravel_index(int(loc), shape)
                row = {"layer_rank": layer_rank, "layer": lname,
                       "in_layer_rank": in_layer_rank, "flat_idx": int(loc),
                       "multi_idx": "(" + ",".join(map(str, multi)) + ")",
                       "w": float(w_flat[int(loc)])}
                for m in methods:
                    row[m] = float(per_m[m][int(loc)]) if m in per_m else ""
                rows.append(row)
            key = lname.replace(".", "_")
            block = {"flat_idx": top_idx.cpu().numpy().astype(np.int64),
                     "w": w_flat[top_idx].cpu().numpy()}
            for m in methods:
                if m in per_m:
                    block[m] = per_m[m][top_idx].cpu().numpy()
            npz_per_layer[key] = block
        if write_csv:
            csv_path = os.path.join(out_dir,
                f"perweight_{tag}_layers{top_layers}x{top_per_layer}_by_{rank_by}_{layer_rank_by}.csv")
            with open(csv_path, "w", newline="") as f:
                hdr = ["layer_rank","layer","in_layer_rank","flat_idx",
                       "multi_idx","w"] + methods
                w = csv.writer(f); w.writerow(hdr)
                for r in rows:
                    w.writerow([r[h] for h in hdr])
            print(f"[dump-weights] wrote {csv_path}"); created.append(csv_path)
        if write_npz:
            npz_path = os.path.join(out_dir,
                f"perweight_{tag}_layers{top_layers}x{top_per_layer}_by_{rank_by}_{layer_rank_by}.npz")
            payload = {}
            for key, block in npz_per_layer.items():
                for sub, arr in block.items():
                    payload[f"{key}__{sub}"] = arr
            np.savez(npz_path, **payload)
            print(f"[dump-weights] wrote {npz_path}"); created.append(npz_path)
        return created

    raise ValueError(f"unknown dump mode: {mode}")


def print_summary_table(summary, methods):
    print("\n" + "=" * 110)
    print("PER-LAYER SUMMARY  (low gini = spread / good ; high top_1pct = clustered)")
    print("=" * 110)
    method_for_table = ("taylor" if "taylor" in methods
                        else ("fisher" if "fisher" in methods else methods[0]))
    header = f"{'layer':<48} | {'numel':>9} | {method_for_table+'_mean':>13} | " \
             f"{method_for_table+'_gini':>11} | {method_for_table+'_top1pct':>14}"
    print(header); print("-" * len(header))
    for layer, row in summary.items():
        m = f"{method_for_table}_mean"; g = f"{method_for_table}_gini"
        t1 = f"{method_for_table}_top_1pct"
        if m in row:
            print(f"{layer:<48} | {row['numel']:>9d} | "
                  f"{row[m]:>13.3e} | {row[g]:>11.4f} | "
                  f"{row.get(t1, float('nan')):>14.4f}")
    print("=" * 110)
    print("Heuristic: gini < ~0.4 and top_1pct < ~0.10 => well spread.")
    print("           gini > ~0.7 and top_1pct > ~0.30 => clustered.")


# =========================================================
# Per-version pipeline
# =========================================================

def run_one_version(model, loader, device, methods, max_batches,
                    out_dir, tag,
                    extra_per_weight=None,
                    dump_args=None):
    print(f"\n[Sensitivity] === version: {tag} ===")
    t0 = time.time()
    scores = compute_all_sensitivity(model, loader, device,
                                     methods=set(methods),
                                     max_batches=max_batches)
    print(f"[Sensitivity] computed in {time.time() - t0:.1f}s")

    extra_per_layer = None
    if extra_per_weight:
        extra_per_layer = {}
        for name, cols in extra_per_weight.items():
            extra_per_layer[name] = {}
            for col, t in cols.items():
                v = t.detach().flatten().cpu()
                extra_per_layer[name][f"{col}_mean"] = float(v.mean())
                extra_per_layer[name][f"{col}_max"]  = float(v.max())
                extra_per_layer[name][f"{col}_sum"]  = float(v.sum())

    summary = summarize(scores, extra_per_layer=extra_per_layer)
    js_path, csv_path = save_summary(summary, out_dir, tag)
    print(f"[Saved] {js_path}")
    print(f"[Saved] {csv_path}")

    if dump_args is not None and dump_args.get("dump_weights"):
        if dump_args["mode"] == "global_topk" and dump_args["top_k_global"] is None:
            print("[dump-weights] global_topk requested but no --top-k; skipping dump")
        else:
            dump_per_weight(
                model, scores, out_dir, tag,
                mode=dump_args["mode"],
                top_k_global=dump_args["top_k_global"],
                top_layers=dump_args["top_layers"],
                top_per_layer=dump_args["top_per_layer"],
                rank_by=dump_args["rank_by"],
                layer_rank_by=dump_args["layer_rank_by"],
                extra_per_weight=extra_per_weight,
                write_csv=dump_args["write_csv"],
                write_npz=dump_args["write_npz"],
            )

    print_summary_table(summary, methods)


# =========================================================
# CLI
# =========================================================

def main():
    p = argparse.ArgumentParser("Per-weight & per-layer sensitivity (with quantization)")
    p.add_argument("--dataset", required=True,
                   choices=["CIFAR10", "CIFAR100", "MNIST", "IMAGENET"])
    p.add_argument("--arch", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--imagenet-root", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--use-pretrained", type=int, default=0)
    p.add_argument("--weights", default="")
    p.add_argument("--methods", nargs="+",
                   default=["magnitude","grad_abs","taylor","fisher"],
                   choices=["magnitude","grad_abs","taylor","fisher","hessian"])
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--out-dir", default="artifacts/sensitivity")

    # Quantization options
    p.add_argument("--quantize-bits", nargs="*", type=int, default=[],
                   help="Also run sensitivity on quantized versions. "
                        "e.g. --quantize-bits 8 4")
    p.add_argument("--quant-error", action="store_true",
                   help="Add per-weight |w_float - w_dequant| as a column.")

    # Per-weight dump options
    p.add_argument("--dump-weights", action="store_true")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--top-layers", type=int, default=None)
    p.add_argument("--top-per-layer", type=int, default=None)
    p.add_argument("--rank-by", default="taylor",
                   choices=["magnitude","grad_abs","taylor","fisher","hessian"])
    p.add_argument("--layer-rank-by", default="mean",
                   choices=["mean","max","sum"])
    p.add_argument("--no-csv", action="store_true")
    p.add_argument("--no-npz", action="store_true")
    args = p.parse_args()
    args.dataset = args.dataset.upper()

    device = TQ.pick_device(args.device, local_rank=0)
    print(f"[Device] {device}")

    _, _, nc = TQ.get_datasets(args.dataset, args.data_root,
                               args.imagenet_root or None)

    def fresh_model():
        m = TQ.build_model(args.arch, nc,
                           use_pretrained=bool(args.use_pretrained)).to(device)
        if args.weights:
            payload = torch.load(args.weights, map_location=device)
            sd = payload.get("state_dict", payload)
            sd = TQ.strip_prefix_from_state_dict(sd)
            m.load_state_dict(sd, strict=True)
            print(f"[Loaded] explicit checkpoint {args.weights}")
        else:
            src, sz = TQ.maybe_load_float32_or_pretrained(
                m, args.dataset, args.arch,
                use_pretrained=bool(args.use_pretrained),
                map_location=device)
            print(f"[Weights] source={src}, size_mb={sz:.2f}")
        return m

    _, test_loader, _, _ = TQ.get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers,
        dist_mode=False, imagenet_root=args.imagenet_root or None)

    # ---- decide dump mode ----
    if args.dump_weights:
        if args.top_layers is not None or args.top_per_layer is not None:
            if args.top_layers is None or args.top_per_layer is None:
                raise SystemExit("Hierarchical mode needs BOTH --top-layers and --top-per-layer.")
            dump_mode = "per_layer_topk"
        elif args.top_k is not None:
            dump_mode = "global_topk"
        else:
            dump_mode = "all"
    else:
        dump_mode = None

    if args.dump_weights and args.rank_by not in args.methods:
        print(f"[dump-weights] WARN: rank_by={args.rank_by} not in methods; "
              f"using {args.methods[0]}")
        args.rank_by = args.methods[0]

    dump_args = {
        "dump_weights": args.dump_weights,
        "mode": dump_mode,
        "top_k_global": args.top_k,
        "top_layers": args.top_layers,
        "top_per_layer": args.top_per_layer,
        "rank_by": args.rank_by,
        "layer_rank_by": args.layer_rank_by,
        "write_csv": not args.no_csv,
        "write_npz": not args.no_npz,
    } if args.dump_weights else None

    base_tag = f"{args.dataset.lower()}_{args.arch.lower()}"

    # =====================================================
    # 1) FLOAT32 run
    # =====================================================
    model = fresh_model()
    run_one_version(model, test_loader, device,
                    methods=args.methods, max_batches=args.max_batches,
                    out_dir=args.out_dir, tag=f"{base_tag}_float32",
                    extra_per_weight=None, dump_args=dump_args)

    # =====================================================
    # 2) Quantized runs (if requested)
    # =====================================================
    for nbits in args.quantize_bits:
        print(f"\n[Quantize] producing INT{nbits} version ...")
        m_q = fresh_model()
        qerr = quantize_model_inplace(m_q, num_bits=nbits)
        # only keep error tensors for the params we'll score
        qerr_filtered = {n: {f"quant_err_int{nbits}": qerr[n]}
                         for n, _ in _iter_weight_params(m_q) if n in qerr}
        extra = qerr_filtered if args.quant_error else None
        run_one_version(m_q, test_loader, device,
                        methods=args.methods, max_batches=args.max_batches,
                        out_dir=args.out_dir,
                        tag=f"{base_tag}_int{nbits}",
                        extra_per_weight=extra, dump_args=dump_args)


if __name__ == "__main__":
    main()
