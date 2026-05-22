# compare_archs.py
# Run sensitivity.py across multiple architectures and produce one
# combined per-layer summary CSV + a comparison report.
#
# Author: Habibur Rahaman, University of Florida, ECE Department
#
# Usage example (ImageNet, pretrained weights, no need to train anything):
#   python compare_archs.py --dataset IMAGENET --imagenet-root /path/to/imagenet \
#       --use-pretrained 1 \
#       --archs resnet18 resnet50 vgg16 mobilenet_v2 efficientnet_b0 \
#       --methods magnitude grad_abs taylor fisher \
#       --max-batches 4
#
# Or for CIFAR-10 (you must have trained checkpoints already):  
#   python compare_archs.py --dataset CIFAR10 \
#       --archs resnet18 vgg16 mobilenet_v2 \
#       --methods magnitude grad_abs taylor fisher --max-batches 8

from __future__ import annotations
import os, json, argparse, subprocess, sys, csv
from typing import List


def _run_one(arch: str, args) -> int:
    cmd = [
        sys.executable, "sensitivity.py",
        "--dataset", args.dataset,
        "--arch", arch,
        "--methods", *args.methods,
        "--max-batches", str(args.max_batches),
        "--out-dir", args.out_dir,
        "--use-pretrained", str(args.use_pretrained),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
    ]
    if args.imagenet_root:
        cmd += ["--imagenet-root", args.imagenet_root]
    if args.quantize_bits:
        cmd += ["--quantize-bits", *map(str, args.quantize_bits)]
    if args.quant_error:
        cmd += ["--quant-error"]
    print("\n" + "#" * 100)
    print("#  RUNNING:", " ".join(cmd))
    print("#" * 100)
    return subprocess.call(cmd)


def _aggregate(args):
    """
    Read each per-arch JSON summary and produce a combined CSV with
    one row per (arch, layer).
    """
    out_csv = os.path.join(args.out_dir, f"compare_{args.dataset.lower()}.csv")
    rows = []
    methods_seen = set()
    for arch in args.archs:
        tag = f"{args.dataset.lower()}_{arch.lower()}_float32"
        js = os.path.join(args.out_dir, f"sensitivity_{tag}.json")
        if not os.path.isfile(js):
            print(f"[compare] missing {js}, skipping arch {arch}")
            continue
        with open(js) as f:
            summary = json.load(f)
        for layer, row in summary.items():
            r = {"arch": arch, "layer": layer}
            r.update(row)
            rows.append(r)
            for k in row:
                methods_seen.add(k)

    if not rows:
        print("[compare] no per-arch JSON files found")
        return

    keys = ["arch", "layer"] + sorted(methods_seen)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            w.writerow([r.get(k, "") for k in keys])
    print(f"[compare] wrote combined: {out_csv}")

    # short comparison: per-arch headline numbers
    print("\n" + "=" * 80)
    print("HEADLINE PER-ARCH NUMBERS  (Taylor, averaged across layers)")
    print("=" * 80)
    print(f"{'arch':<20} | {'#layers':>8} | {'avg taylor_gini':>17} | {'avg taylor_top1pct':>20}")
    print("-" * 80)
    for arch in args.archs:
        per_arch = [r for r in rows if r["arch"] == arch]
        if not per_arch:
            continue
        ginis = [r.get("taylor_gini", None) for r in per_arch]
        tops  = [r.get("taylor_top_1pct", None) for r in per_arch]
        ginis = [v for v in ginis if v is not None and v != ""]
        tops  = [v for v in tops  if v is not None and v != ""]
        gm = sum(ginis) / len(ginis) if ginis else float("nan")
        tm = sum(tops) / len(tops) if tops else float("nan")
        print(f"{arch:<20} | {len(per_arch):>8} | {gm:>17.4f} | {tm:>20.4f}")
    print("=" * 80)


def main():
    ap = argparse.ArgumentParser("Run sensitivity.py across multiple architectures")
    ap.add_argument("--dataset", required=True,
                    choices=["CIFAR10","CIFAR100","MNIST","IMAGENET"])
    ap.add_argument("--archs", nargs="+", required=True)
    ap.add_argument("--methods", nargs="+",
                    default=["magnitude","grad_abs","taylor","fisher"])
    ap.add_argument("--max-batches", type=int, default=8)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--imagenet-root", default="")
    ap.add_argument("--use-pretrained", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quantize-bits", nargs="*", type=int, default=[])
    ap.add_argument("--quant-error", action="store_true")
    ap.add_argument("--out-dir", default="artifacts/sensitivity")
    ap.add_argument("--skip-aggregate", action="store_true",
                    help="Just run per-arch and skip the combined CSV step.")
    args = ap.parse_args()

    failed = []
    for arch in args.archs:
        rc = _run_one(arch, args)
        if rc != 0:
            failed.append(arch)

    if failed:
        print(f"\n[compare] FAILED archs: {failed}")
    if not args.skip_aggregate:
        _aggregate(args)


if __name__ == "__main__":
    main()
