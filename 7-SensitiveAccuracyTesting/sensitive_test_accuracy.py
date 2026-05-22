"""
sensitive_test_accuracy.py — Stage 7: Accuracy evaluation with layer protection.

Same as stage 6 (test_accuracy.py), but before inference selectively replaces
ECC-embedded weights with the original unmodified int8 weights for layers that
are both small (numel < numel_threshold) and highly sensitive (grad_norm_score
in the top (1 - score_percentile) fraction), subject to a hard cap on the
total fraction of weights replaced (max_protect).

The goal is to minimise accuracy drop by protecting the most critical small
layers, while keeping the overhead below 2% of total network weights.

Usage (via run.sh):
    python3 sensitive_test_accuracy.py \
        --dataset CIFAR10 --arch resnet18 --quant-bits 8 \
        --t-value 1 --approach no --codeword 63 \
        --ecc-dir /path/to/embeddedECC \
        --models-dir /path/to/models \
        --sensitivity-dir /path/to/sensitivity \
        --results-dir /path/to/accuracy_results \
        --imagenet-root /path/to/imagenet-val
"""

import os, sys, json, argparse
import torch
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.eval_functions import (
    pick_device, get_dataloaders, build_model, evaluate,
    dequantize_tensor, dequantize_per_channel_conv, strip_prefix_from_state_dict,
)


def _select_protected_layers(sens_json: dict, layer_df: pd.DataFrame,
                              numel_threshold: int, score_percentile: float,
                              max_protect: float):
    """
    Return the list of layer names to protect (replace ECC weights with originals).

    Selection criteria (all must hold):
      1. numel < numel_threshold           (small layer — cheap to protect)
      2. grad_norm_score >= score_threshold (top (1-score_percentile) most sensitive)
      3. cumulative protected weights < max_protect * total_weights

    Returns (protected_layers, protect_count, total_weights).
    """
    total_weights = sum(v["numel"] for v in sens_json.values())

    # High threshold: only the most sensitive layers qualify
    score_threshold = layer_df["grad_norm_score"].quantile(score_percentile)

    df_sorted = layer_df.sort_values("grad_norm_score", ascending=False)

    protected = []
    protect_count = 0
    for _, row in df_sorted.iterrows():
        layer = row["layer"]
        score = row["grad_norm_score"]
        if score < score_threshold:
            break  # all remaining rows are below threshold
        if layer not in sens_json:
            continue
        numel = sens_json[layer].get("numel", 0)
        if numel == 0 or numel >= numel_threshold:
            continue
        if (protect_count + numel) / total_weights > max_protect:
            continue  # would exceed cap; skip this layer but keep checking others
        protected.append(layer)
        protect_count += numel

    return protected, protect_count, total_weights


def _dequantize_and_load(model, ckpt: dict, device):
    """Dequantize int8 weights in ckpt using stored scales, load into model."""
    qsd    = ckpt["qstate_dict"]
    scales = ckpt["meta"]["scales"]
    dsd = {}
    for k, v in qsd.items():
        sinfo = scales.get(k)
        if sinfo is None:
            dsd[k] = v
        elif sinfo.get("type") == "per_tensor":
            dsd[k] = dequantize_tensor(v, sinfo["scale"])
        elif sinfo.get("type") == "per_channel":
            dsd[k] = dequantize_per_channel_conv(v, sinfo["scales"].to(v.device))
        else:
            dsd[k] = v
    dsd = strip_prefix_from_state_dict(dsd)
    model.load_state_dict(dsd, strict=True)
    return ckpt["meta"]


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate ECC-embedded model accuracy with sensitive-layer protection")
    ap.add_argument("--dataset",         required=True, choices=["CIFAR10", "CIFAR100", "IMAGENET"])
    ap.add_argument("--arch",            required=True,
                    choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--quant-bits",      required=True, type=int, choices=[4, 8, 16])
    ap.add_argument("--t-value",         required=True, type=int)
    ap.add_argument("--approach",        required=True)
    ap.add_argument("--codeword",        required=True, type=int)
    ap.add_argument("--ecc-dir",         required=True)
    ap.add_argument("--models-dir",      required=True)
    ap.add_argument("--sensitivity-dir", required=True)
    ap.add_argument("--results-dir",     required=True)
    ap.add_argument("--imagenet-root",   default=None)
    ap.add_argument("--batch-size",      type=int, default=256)
    ap.add_argument("--workers",         type=int, default=4)
    # Protection parameters
    ap.add_argument("--numel-threshold", type=int,   default=5000,
                    help="Only protect layers with fewer than this many weights")
    ap.add_argument("--score-percentile", type=float, default=0.90,
                    help="Protect only layers above this grad_norm_score percentile (0-1)")
    ap.add_argument("--max-protect",     type=float, default=0.02,
                    help="Hard cap: at most this fraction of total weights replaced")
    args = ap.parse_args()

    ds_lower  = args.dataset.lower()
    bit_label = f"{args.quant_bits}-bit"
    m_tag     = f"M{args.codeword}_t{args.t_value}"

    ecc_model_path  = os.path.join(
        args.ecc_dir, ds_lower, args.arch, "PTQ", bit_label,
        m_tag, args.approach, "ECC_Embedded_model.pth",
    )
    orig_model_path = os.path.join(
        args.models_dir, ds_lower, args.arch, "PTQ",
        f"model_int{args.quant_bits}_ptq.pth",
    )
    sens_json_path  = os.path.join(
        args.sensitivity_dir, ds_lower, args.arch, "PTQ", bit_label,
        f"sensitivity_{ds_lower}_{args.arch}_int{args.quant_bits}.json",
    )
    layer_csv_path  = os.path.join(
        args.sensitivity_dir, ds_lower, args.arch, "PTQ", bit_label,
        f"layer_summary_{ds_lower}_{args.arch}_int{args.quant_bits}_grad_norm.csv",
    )

    print(f"[7-SensitiveAccuracyTesting] {args.dataset}/{args.arch}/{bit_label}/{m_tag}/{args.approach}")

    if not os.path.exists(ecc_model_path):
        print(f"  [skip] ECC model not found: {ecc_model_path}")
        return
    if not os.path.exists(orig_model_path):
        print(f"  [skip] Original model not found: {orig_model_path}")
        return

    # ---- Determine which layers to protect ----
    protected_layers = []
    protect_count    = 0
    total_weights    = 0

    if os.path.exists(sens_json_path) and os.path.exists(layer_csv_path):
        with open(sens_json_path) as f:
            sens_data = json.load(f)
        layer_df = pd.read_csv(layer_csv_path)

        protected_layers, protect_count, total_weights = _select_protected_layers(
            sens_data, layer_df,
            numel_threshold=args.numel_threshold,
            score_percentile=args.score_percentile,
            max_protect=args.max_protect,
        )
        pct = 100.0 * protect_count / total_weights if total_weights else 0.0
        print(f"  [protect] {len(protected_layers)} layers, "
              f"{protect_count:,}/{total_weights:,} weights = {pct:.4f}% replaced with originals")
        for lname in protected_layers:
            print(f"    - {lname}  ({sens_data[lname]['numel']:,} weights)")
    else:
        print(f"  [warn] Sensitivity data not found — running without protection")
        print(f"    sens_json : {sens_json_path}")
        print(f"    layer_csv : {layer_csv_path}")

    # ---- Load checkpoints ----
    ecc_ckpt  = torch.load(ecc_model_path,  map_location="cpu")
    orig_ckpt = torch.load(orig_model_path, map_location="cpu")

    # ---- Replace ECC weights with originals for protected layers ----
    replaced = 0
    for layer in protected_layers:
        if layer in ecc_ckpt["qstate_dict"] and layer in orig_ckpt["qstate_dict"]:
            ecc_ckpt["qstate_dict"][layer] = orig_ckpt["qstate_dict"][layer]
            replaced += 1
        else:
            print(f"  [warn] Layer {layer!r} not found in one of the checkpoints — skipped")
    if protected_layers:
        print(f"  [protect] {replaced}/{len(protected_layers)} replacements applied")

    # ---- Build model and run evaluation ----
    device = pick_device("cuda", local_rank=0)
    print(f"  [device] {device}")

    _, test_loader, nc, _ = get_dataloaders(
        args.dataset,
        data_root="./data",
        batch_size=args.batch_size,
        num_workers=args.workers,
        dist_mode=False,
        imagenet_root=args.imagenet_root,
    )

    model = build_model(args.arch, nc, use_pretrained=False).to(device)
    _dequantize_and_load(model, ecc_ckpt, device)

    top1, top5 = evaluate(model, test_loader, device)
    print(f"  [result] Top-1 = {top1:.4f}%  Top-5 = {top5:.4f}%")

    # ---- Save results ----
    out_dir = (Path(args.results_dir) / ds_lower / args.arch
               / "PTQ" / bit_label / m_tag / args.approach)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "sensitive_accuracy.json"

    protect_pct = protect_count / total_weights if total_weights else 0.0
    result = {
        "dataset":          args.dataset,
        "arch":             args.arch,
        "bits":             args.quant_bits,
        "t":                args.t_value,
        "approach":         args.approach,
        "codeword":         args.codeword,
        "top1":             round(top1, 4),
        "top5":             round(top5, 4),
        "protected_layers": protected_layers,
        "protect_count":    protect_count,
        "total_weights":    total_weights,
        "protect_pct":      round(protect_pct, 6),
        "numel_threshold":  args.numel_threshold,
        "score_percentile": args.score_percentile,
        "max_protect":      args.max_protect,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
