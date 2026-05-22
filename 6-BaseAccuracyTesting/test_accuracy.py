"""
test_accuracy.py — Stage 6: Base accuracy evaluation for ECC-embedded models.

Loads an ECC-embedded quantized checkpoint, dequantizes weights, and evaluates
Top-1 / Top-5 accuracy on the test split of the given dataset.

Usage (via run.sh):
    python3 test_accuracy.py \
        --dataset CIFAR10 --arch resnet18 --quant-bits 8 \
        --t-value 1 --approach no --codeword 63 \
        --ecc-dir /path/to/embeddedECC \
        --results-dir /path/to/accuracy_results \
        --imagenet-root /path/to/imagenet-val
"""

import os, sys, json, argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.eval_functions import (
    pick_device, get_dataloaders, build_model, load_quantized_into_model, evaluate,
)


def main():
    ap = argparse.ArgumentParser(description="Evaluate ECC-embedded model accuracy")
    ap.add_argument("--dataset",      required=True, choices=["CIFAR10", "CIFAR100", "IMAGENET"])
    ap.add_argument("--arch",         required=True,
                    choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--quant-bits",   required=True, type=int, choices=[4, 8, 16])
    ap.add_argument("--t-value",      required=True, type=int)
    ap.add_argument("--approach",     required=True)
    ap.add_argument("--codeword",     required=True, type=int)
    ap.add_argument("--ecc-dir",      required=True)
    ap.add_argument("--results-dir",  required=True)
    ap.add_argument("--imagenet-root", default=None)
    ap.add_argument("--batch-size",   type=int, default=256)
    ap.add_argument("--workers",      type=int, default=4)
    args = ap.parse_args()

    ds_lower  = args.dataset.lower()
    bit_label = f"{args.quant_bits}-bit"
    m_tag     = f"M{args.codeword}_t{args.t_value}"

    ecc_model_path = os.path.join(
        args.ecc_dir, ds_lower, args.arch, "PTQ", bit_label,
        m_tag, args.approach, "ECC_Embedded_model.pth",
    )

    print(f"[6-BaseAccuracyTesting] {args.dataset}/{args.arch}/{bit_label}/{m_tag}/{args.approach}")

    if not os.path.exists(ecc_model_path):
        print(f"  [skip] ECC model not found: {ecc_model_path}")
        return

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
    meta  = load_quantized_into_model(
        model, args.dataset, args.arch, args.quant_bits, "ptq",
        map_location=device, weight_argument=ecc_model_path,
    )

    top1, top5 = evaluate(model, test_loader, device)
    print(f"  [result] Top-1 = {top1:.4f}%  Top-5 = {top5:.4f}%")

    out_dir = Path(args.results_dir) / ds_lower / args.arch / "PTQ" / bit_label / m_tag / args.approach
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "accuracy.json"

    result = {
        "dataset":  args.dataset,
        "arch":     args.arch,
        "bits":     args.quant_bits,
        "t":        args.t_value,
        "approach": args.approach,
        "codeword": args.codeword,
        "top1":     round(top1, 4),
        "top5":     round(top5, 4),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  [saved] {out_path}")


if __name__ == "__main__":
    main()
