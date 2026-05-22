# layer_then_weight.py
# Hierarchical sensitivity analysis: score layers first, pick top-K layers,
# then score individual weights inside those layers and pick top-N per layer.
# Supports float32 / INT8 / INT4 deployments, multiple architectures, and
# saves everything into one combined CSV per (arch, format).
#
# Pipeline (per architecture, per format):
#   1) Load model (or torchvision pretrained for ImageNet).
#   2) Optionally quantize (INT8 or INT4) via test-Quantizer's PTQ.
#   3) Score every layer's sensitivity using a layer-level metric:
#        - "grad_norm"   : ||dL/dW||_F^2 / numel    (default; one backward pass)
#        - "taylor_mean" : mean of |w * dL/dw|
#        - "fisher_mean" : mean of (dL/dw)^2
#        - "perturb"     : empirical -- add Gaussian noise to layer, measure ?Loss
#   4) Rank layers, keep top-K.
#   5) Inside each chosen layer, score every weight with Taylor + Fisher.
#   6) Keep top-N weights per layer.
#   7) Write one CSV per (arch, format) containing all K x N selected weights
#      with full info: layer rank, layer score, weight idx, weight value,
#      taylor, fisher, magnitude, and (when quantized) per-weight quant error.
#
# Author: Habibur Rahaman, University of Florida, ECE Department
#
# Examples:
#   # CIFAR-10 ResNet-18, float + INT8 + INT4:
#   python layer_then_weight.py --dataset CIFAR10 --archs resnet18 \
#       --quantize-bits 8 4 --top-layers 5 --top-per-layer 200 \
#       --layer-metric grad_norm --max-batches 8
#
#   # ImageNet sweep across 5 architectures, no training needed:
#   python layer_then_weight.py --dataset IMAGENET \
#       --imagenet-root ./imagenet-val --use-pretrained 1 \
#       --archs resnet18 resnet50 vgg16 mobilenet_v2 efficientnet_b0 \
#       --quantize-bits 8 4 --top-layers 5 --top-per-layer 200 \
#       --max-batches 4

from __future__ import annotations
import os, csv, json, time, argparse, importlib.util, sys
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn


# =========================================================
# Bring in test-Quantizer.py
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
    raise FileNotFoundError("Could not locate test-Quantizer.py")

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


@torch.no_grad()
def _avg_loss(model, loader, device, max_batches, criterion) -> float:
    model.eval()
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        tot += criterion(model(x), y).item()
        n += 1
    return tot / max(1, n)


# =========================================================
# Quantization helper (delegates to test-Quantizer)
# =========================================================

def quantize_model_inplace(model: nn.Module, num_bits: int) -> Dict[str, torch.Tensor]:
    """
    Apply per-channel/per-tensor PTQ to the model in place.
    Returns dict: name -> per-weight quantization error |w - w_dq|.
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
                w_dq = TQ.dequantize_tensor(q_tensor, sinfo["scale"]) \
                          .to(p.device).to(p.dtype)
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
# Per-weight scoring (Taylor + Fisher in one pass)
# =========================================================

def compute_taylor_fisher(model, loader, device,
                           max_batches=8, criterion=None
                           ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Single pass over batches:
      - Accumulate gradients (averaged) -> Taylor = |w * mean_grad|.
      - Accumulate squared gradients per batch -> Fisher = mean(grad^2).
    """
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    model.eval()

    sum_grad: Dict[str, torch.Tensor] = {}
    sum_grad2: Dict[str, torch.Tensor] = {}
    n_batches = 0

    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        _zero_grads(model)
        with torch.enable_grad():
            criterion(model(x), y).backward()

        for name, p in _iter_weight_params(model):
            if p.grad is None:
                continue
            g = p.grad.detach()
            if name not in sum_grad:
                sum_grad[name] = g.clone()
                sum_grad2[name] = (g ** 2).clone()
            else:
                sum_grad[name] += g
                sum_grad2[name] += g ** 2
        n_batches += 1

    mean_grad: Dict[str, torch.Tensor] = {}
    fisher: Dict[str, torch.Tensor] = {}
    taylor: Dict[str, torch.Tensor] = {}
    for name, p in _iter_weight_params(model):
        if name not in sum_grad:
            continue
        mg = sum_grad[name] / max(1, n_batches)
        fi = sum_grad2[name] / max(1, n_batches)
        mean_grad[name] = mg
        fisher[name]    = fi
        taylor[name]    = (p.detach() * mg).abs()

    return taylor, fisher


# =========================================================
# Layer-level sensitivity scoring  (Step 1 of the pipeline)
# =========================================================

def layer_score_grad_norm(model, loader, device, max_batches=8) -> Dict[str, float]:
    """||dL/dW||_F^2 / numel  per layer.  One backward pass per batch."""
    crit = nn.CrossEntropyLoss()
    model.eval()
    accum: Dict[str, float] = {}
    n_batches = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        _zero_grads(model)
        with torch.enable_grad():
            crit(model(x), y).backward()
        for name, p in _iter_weight_params(model):
            if p.grad is None:
                continue
            v = (p.grad.detach() ** 2).sum().item() / p.numel()
            accum[name] = accum.get(name, 0.0) + v
        n_batches += 1
    for k in accum:
        accum[k] /= max(1, n_batches)
    return accum


def layer_score_from_per_weight(per_weight: Dict[str, torch.Tensor], agg: str
                                ) -> Dict[str, float]:
    out = {}
    for name, t in per_weight.items():
        v = t.detach().flatten()
        if agg == "mean": out[name] = float(v.mean())
        elif agg == "max": out[name] = float(v.max())
        elif agg == "sum": out[name] = float(v.sum())
        else: raise ValueError(agg)
    return out


def layer_score_perturb(model, loader, device, max_batches=4,
                         sigma=0.01, n_trials=3) -> Dict[str, float]:
    """
    Empirical layer sensitivity: add Gaussian noise to each layer in turn,
    measure  Loss on a few batches.  More expensive but most direct.
    """
    crit = nn.CrossEntropyLoss()
    base = _avg_loss(model, loader, device, max_batches, crit)
    out: Dict[str, float] = {}
    for name, p in _iter_weight_params(model):
        std = sigma * p.detach().abs().mean().item()
        deltas = []
        for _ in range(n_trials):
            noise = torch.randn_like(p) * std
            p.data.add_(noise)
            l = _avg_loss(model, loader, device, max_batches, crit)
            p.data.sub_(noise)
            deltas.append(l - base)
        out[name] = float(sum(deltas) / max(1, len(deltas)))
    return out


# =========================================================
# The hierarchical pipeline (Steps 1->6 for one model/format)
# =========================================================

def run_hierarchical(model, loader, device, *,
                     layer_metric: str,
                     top_layers: int,
                     top_per_layer: int,
                     max_batches: int,
                     extra_per_weight: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
                     ) -> Tuple[List[dict], Dict[str, float], Dict[str, Dict[str, torch.Tensor]]]:
    """
    Returns:
      rows           : list of per-weight selected dicts  (the K*N output rows)
      layer_scores   : full layer-level scores (all layers, not just top-K)
      per_weight     : {'taylor': {layer: tensor}, 'fisher': {...}}
                       (the per-weight Taylor / Fisher tensors used for selection)
    """
    print(f"  [step1] computing per-weight Taylor + Fisher ...")
    taylor, fisher = compute_taylor_fisher(model, loader, device,
                                           max_batches=max_batches)
    per_weight = {"taylor": taylor, "fisher": fisher}

    print(f"  [step1b] computing layer-level scores via '{layer_metric}' ...")
    if layer_metric == "grad_norm":
        layer_scores = layer_score_grad_norm(model, loader, device,
                                             max_batches=max_batches)
    elif layer_metric == "taylor_mean":
        layer_scores = layer_score_from_per_weight(taylor, "mean")
    elif layer_metric == "fisher_mean":
        layer_scores = layer_score_from_per_weight(fisher, "mean")
    elif layer_metric == "perturb":
        layer_scores = layer_score_perturb(model, loader, device,
                                            max_batches=min(4, max_batches))
    else:
        raise ValueError(f"unknown layer-metric {layer_metric}")

    # Step 2: rank layers, keep top-K
    ranked = sorted(layer_scores.items(), key=lambda kv: kv[1], reverse=True)
    if top_layers > len(ranked):
        print(f"  [step2] top_layers={top_layers} exceeds layer count={len(ranked)}; selecting all {len(ranked)} layers")
    chosen = ranked[:top_layers]
    print(f"  [step2] top-{top_layers} layers by {layer_metric}:")
    for rk, (lname, sc) in enumerate(chosen):
        print(f"      layer_rank={rk}  {lname:48s}  score={sc:.4e}")

    # Step 3 + 4: within each chosen layer, take top-N weights by Taylor
    rows: List[dict] = []
    name_to_param = dict(_iter_weight_params(model))
    for layer_rank, (lname, layer_score) in enumerate(chosen):
        if lname not in taylor:
            continue
        p = name_to_param[lname]
        shape = tuple(p.shape)
        scores_w = taylor[lname].flatten()
        if top_per_layer > scores_w.numel():
            print(f"  [step3] top_per_layer={top_per_layer} exceeds weight count={scores_w.numel()} in {lname}; selecting all weights")
        N = min(top_per_layer, scores_w.numel())
        top_vals, top_idx = torch.topk(scores_w, N, largest=True, sorted=True)

        w_flat = p.detach().cpu().flatten()
        tay_flat = taylor[lname].detach().cpu().flatten()
        fis_flat = fisher[lname].detach().cpu().flatten()
        mag_flat = p.detach().cpu().flatten().abs()

        # extra columns (e.g. quant error) for this layer
        extra_flat = {}
        if extra_per_weight and lname in extra_per_weight:
            for col, t in extra_per_weight[lname].items():
                extra_flat[col] = t.detach().cpu().flatten()

        for in_layer_rank, loc in enumerate(top_idx.tolist()):
            multi = np.unravel_index(int(loc), shape)
            row = {
                "layer_rank":      layer_rank,
                "layer":           lname,
                "layer_score":     layer_score,
                "in_layer_rank":   in_layer_rank,
                "flat_idx":        int(loc),
                "multi_idx":       "(" + ",".join(map(str, multi)) + ")",
                "w":               float(w_flat[int(loc)]),
                "magnitude":       float(mag_flat[int(loc)]),
                "taylor":          float(tay_flat[int(loc)]),
                "fisher":          float(fis_flat[int(loc)]),
            }
            for col, t in extra_flat.items():
                row[col] = float(t[int(loc)])
            rows.append(row)

    return rows, layer_scores, per_weight


def write_rows_csv(rows: List[dict], out_csv: str, extra_cols: List[str]):
    if not rows:
        return
    base_cols = ["layer_rank","layer","layer_score","in_layer_rank",
                 "flat_idx","multi_idx","w","magnitude","taylor","fisher"]
    cols = base_cols + extra_cols
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r.get(c, "") for c in cols])
    print(f"  [saved] {out_csv}  ({len(rows)} rows)")


def write_layer_summary(layer_scores: Dict[str, float],
                        out_csv: str, layer_metric: str):
    cols = ["layer_rank", "layer", f"{layer_metric}_score"]
    ranked = sorted(layer_scores.items(), key=lambda kv: kv[1], reverse=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols)
        for rk, (lname, sc) in enumerate(ranked):
            w.writerow([rk, lname, sc])
    print(f"  [saved] {out_csv}")


# =========================================================
# Per-arch orchestration: float + (optional) INT8/INT4
# =========================================================

def fresh_model(arch, dataset, num_classes, use_pretrained,
                weights_path, device):
    m = TQ.build_model(arch, num_classes, use_pretrained=bool(use_pretrained)).to(device)
    if weights_path:
        payload = torch.load(weights_path, map_location=device)
        sd = payload.get("state_dict", payload)
        sd = TQ.strip_prefix_from_state_dict(sd)
        m.load_state_dict(sd, strict=True)
        print(f"  [weights] explicit checkpoint {weights_path}")
    else:
        src, sz = TQ.maybe_load_float32_or_pretrained(
            m, dataset, arch,
            use_pretrained=bool(use_pretrained),
            map_location=device)
        print(f"  [weights] source={src}, size_mb={sz:.2f}")
    return m


def run_one_arch(arch: str, args):
    device = TQ.pick_device(args.device, local_rank=0)

    _, _, nc = TQ.get_datasets(args.dataset, args.data_root,
                                args.imagenet_root or None)
    _, test_loader, _, _ = TQ.get_dataloaders(
        args.dataset, args.data_root, args.batch_size, args.workers,
        dist_mode=False, imagenet_root=args.imagenet_root or None)

    base_tag = f"{args.dataset.lower()}_{arch.lower()}"
    out_dir  = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    formats = ["float32"] + [f"int{b}" for b in args.quantize_bits]

    for fmt in formats:
        print(f"\n=== [{arch}] format={fmt} ===")
        m = fresh_model(arch, args.dataset, nc,
                        args.use_pretrained, args.weights, device)
        extra = None
        if fmt.startswith("int"):
            nbits = int(fmt[3:])
            qerr = quantize_model_inplace(m, num_bits=nbits)
            # turn into per-weight extra column
            extra = {}
            for name, _ in _iter_weight_params(m):
                if name in qerr:
                    extra[name] = {f"quant_err_int{nbits}": qerr[name]}

        rows, layer_scores, _ = run_hierarchical(
            m, test_loader, device,
            layer_metric=args.layer_metric,
            top_layers=args.top_layers,
            top_per_layer=args.top_per_layer,
            max_batches=args.max_batches,
            extra_per_weight=extra,
        )

        # determine extra columns present in rows
        extra_cols = []
        if rows:
            sample = rows[0]
            for k in sample.keys():
                if k.startswith("quant_err_") and k not in extra_cols:
                    extra_cols.append(k)

        out_csv = os.path.join(
            out_dir,
            f"layer_then_weight_{base_tag}_{fmt}_L{args.top_layers}xN{args.top_per_layer}_{args.layer_metric}.csv"
        )
        write_rows_csv(rows, out_csv, extra_cols)

        layer_csv = os.path.join(
            out_dir,
            f"layer_summary_{base_tag}_{fmt}_{args.layer_metric}.csv"
        )
        write_layer_summary(layer_scores, layer_csv, args.layer_metric)

        # also dump JSON of layer ranking for quick reuse
        ranked = sorted(layer_scores.items(), key=lambda kv: kv[1], reverse=True)
        with open(os.path.join(
                out_dir,
                f"layer_summary_{base_tag}_{fmt}_{args.layer_metric}.json"
            ), "w") as f:
            json.dump([{"rank": i, "layer": k, "score": v}
                       for i, (k, v) in enumerate(ranked)], f, indent=2)


# =========================================================
# CLI
# =========================================================

def main():
    p = argparse.ArgumentParser(
        "Hierarchical layer-then-weight sensitivity (with quantization support)"
    )
    p.add_argument("--dataset", required=True,
                   choices=["CIFAR10","CIFAR100","MNIST","IMAGENET"])
    p.add_argument("--archs", nargs="+", required=True,
                   help="One or more architectures, e.g. resnet18 resnet50 vgg16")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--imagenet-root", default="")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--use-pretrained", type=int, default=0)
    p.add_argument("--weights", default="",
                   help="Optional explicit checkpoint path. Used for ALL archs.")

    p.add_argument("--layer-metric",
                   default="grad_norm",
                   choices=["grad_norm", "taylor_mean", "fisher_mean", "perturb"],
                   help="How to score each layer for the layer-ranking step.")
    p.add_argument("--top-layers", type=int, default=5)
    p.add_argument("--top-per-layer", type=int, default=200)
    p.add_argument("--max-batches", type=int, default=8)

    p.add_argument("--quantize-bits", nargs="*", type=int, default=[],
                   help="Also run on quantized formats, e.g. --quantize-bits 8 4")

    p.add_argument("--out-dir", default="artifacts/sensitivity")
    args = p.parse_args()
    args.dataset = args.dataset.upper()

    print(f"[device] {args.device}")
    print(f"[archs ] {args.archs}")
    print(f"[fmts  ] float32 + {args.quantize_bits}")
    print(f"[layers] top-{args.top_layers} by {args.layer_metric}")
    print(f"[weights/layer] top-{args.top_per_layer}")

    t0 = time.time()
    for arch in args.archs:
        try:
            run_one_arch(arch, args)
        except Exception as e:
            print(f"[!! arch={arch}] failed: {type(e).__name__}: {e}")
    print(f"\n[done] elapsed {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
