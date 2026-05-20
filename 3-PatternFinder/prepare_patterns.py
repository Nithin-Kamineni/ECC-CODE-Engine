#!/usr/bin/env python3
"""
prepare_patterns.py
-------------------
Bridge between layer_then_weight_*.csv (per-weight sensitivity) and
find_pattern.py (hardware interleaver search).

For EACH layer in the CSV it:
  1. Reads the selected weights' flat_idx + score (taylor by default).
  2. Looks up the layer's TRUE size N (numel) from the model.
  3. Builds a dense  sens[0..N-1]  array:
        - "indicator" mode (default): sens[selected] = 1.0, rest = 0.0
          -> the selected weights ARE the sensitive set.
        - "value" mode: sens[selected] = score, rest = 0.0
          -> use raw score magnitudes (needs --threshold).
  4. Saves the dense array (<layer>_sens.npy) and the flat indices
     (combined sensitive_flatidx_by_layer.csv) so you can reuse them.
  5. (optional, --run-search) imports find_pattern and runs search() on that
     layer, then writes the best interleaver + its hardware rule.

WHY N MUST COME FROM THE MODEL
------------------------------
The CSV holds only the top-N selected weights, so max(flat_idx) under-counts
the layer. find_pattern permutes the WHOLE [0..N-1] space, so it needs the
real numel. We obtain it (in priority order):
    1. --shapes-json  (a JSON  {layer_name: numel}  you supply)
    2. torchvision model of --arch  (default)
    3. built-in ResNet-18 fallback table (no torch needed)

Examples
--------
  # prepare arrays AND run the interleaver search for every layer:
  python prepare_patterns.py \
      --csv artifacts/sensitivity/layer_then_weight_cifar10_resnet18_float32_L5xN200_grad_norm.csv \
      --arch resnet18 --run-search --group-size 8 --max-sens 2

  # just dump the dense arrays + flat indices, no search:
  python prepare_patterns.py --csv <file>.csv --arch resnet18

  # supply sizes yourself (skip torch entirely):
  python prepare_patterns.py --csv <file>.csv --shapes-json sizes.json
"""

from __future__ import annotations
import os
import sys
import json
import argparse
import importlib.util
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Built-in ResNet-18 weight sizes (numel per *.weight tensor that
# layer_then_weight.py would ever select: conv + fc, BN/bias excluded).
# Conv shapes are identical across num_classes; only fc depends on classes,
# and fc is rarely selected, so we compute it from --num-classes if needed.
# --------------------------------------------------------------------------
def _resnet18_numel_table(num_classes: int) -> dict:
    # (out, in, kh, kw) for every conv weight in torchvision resnet18
    conv = {
        "conv1.weight": (64, 3, 7, 7),
        "layer1.0.conv1.weight": (64, 64, 3, 3),
        "layer1.0.conv2.weight": (64, 64, 3, 3),
        "layer1.1.conv1.weight": (64, 64, 3, 3),
        "layer1.1.conv2.weight": (64, 64, 3, 3),
        "layer2.0.conv1.weight": (128, 64, 3, 3),
        "layer2.0.conv2.weight": (128, 128, 3, 3),
        "layer2.0.downsample.0.weight": (128, 64, 1, 1),
        "layer2.1.conv1.weight": (128, 128, 3, 3),
        "layer2.1.conv2.weight": (128, 128, 3, 3),
        "layer3.0.conv1.weight": (256, 128, 3, 3),
        "layer3.0.conv2.weight": (256, 256, 3, 3),
        "layer3.0.downsample.0.weight": (256, 128, 1, 1),
        "layer3.1.conv1.weight": (256, 256, 3, 3),
        "layer3.1.conv2.weight": (256, 256, 3, 3),
        "layer4.0.conv1.weight": (512, 256, 3, 3),
        "layer4.0.conv2.weight": (512, 512, 3, 3),
        "layer4.0.downsample.0.weight": (512, 256, 1, 1),
        "layer4.1.conv1.weight": (512, 512, 3, 3),
        "layer4.1.conv2.weight": (512, 512, 3, 3),
    }
    table = {k: int(np.prod(v)) for k, v in conv.items()}
    table["fc.weight"] = 512 * num_classes
    return table


def get_numel_map(layers_needed, arch, num_classes, shapes_json):
    """Return {layer_name: numel} covering layers_needed, by best available means."""
    # 1) explicit JSON
    if shapes_json:
        with open(shapes_json) as f:
            user_map = json.load(f)
        return {L: int(user_map[L]) for L in layers_needed if L in user_map}, "shapes-json"

    # 2) torchvision model
    try:
        import torch  # noqa
        import torchvision
        ctor = getattr(torchvision.models, arch, None)
        if ctor is None:
            raise RuntimeError(f"torchvision has no model '{arch}'")
        try:
            model = ctor(num_classes=num_classes)
        except TypeError:
            model = ctor()
        m = {n: int(p.numel()) for n, p in model.named_parameters()}
        return {L: m[L] for L in layers_needed if L in m}, "torchvision"
    except Exception as e:
        print(f"[numel] torchvision path unavailable ({type(e).__name__}: {e}); "
              f"falling back to built-in table.")

    # 3) built-in resnet18 table
    if arch.lower() == "resnet18":
        tbl = _resnet18_numel_table(num_classes)
        return {L: tbl[L] for L in layers_needed if L in tbl}, "builtin-resnet18"

    raise SystemExit(
        f"Could not determine layer sizes for arch={arch}. "
        f"Install torchvision or pass --shapes-json.")


# --------------------------------------------------------------------------
def _sanitize(name: str) -> str:
    return name.replace(".", "_").replace("/", "_").replace(" ", "_")


def _import_find_pattern(path_hint=None):
    cands = [path_hint, "find_pattern.py",
             os.path.join(os.path.dirname(os.path.abspath(__file__)), "find_pattern.py")]
    for p in cands:
        if p and os.path.isfile(p):
            spec = importlib.util.spec_from_file_location("find_pattern", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    raise FileNotFoundError("find_pattern.py not found (use --find-pattern-path)")


def layer_order(df):
    seen = []
    for L in df["layer"].tolist():
        if L not in seen:
            seen.append(L)
    return seen


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        "Prepare per-layer dense sensitivity arrays and (optionally) run the "
        "find_pattern interleaver search per layer.")
    ap.add_argument("--csv", required=True, help="layer_then_weight_*.csv path")
    ap.add_argument("--arch", default="resnet18")
    ap.add_argument("--num-classes", type=int, default=None,
                    help="Default inferred from filename (cifar10->10, cifar100->100).")
    ap.add_argument("--score", default="taylor",
                    choices=["taylor", "fisher", "magnitude"])
    ap.add_argument("--sensitive-mode", default="indicator",
                    choices=["indicator", "value"],
                    help="indicator: selected=1.0; value: selected=raw score.")
    ap.add_argument("--shapes-json", default=None,
                    help="Optional JSON {layer_name: numel} to override sizing.")

    # find_pattern parameters
    ap.add_argument("--run-search", action="store_true",
                    help="Also run find_pattern.search() on each layer.")
    ap.add_argument("--find-pattern-path", default=None)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-sens", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Sensitivity cutoff. Default 0.5 for indicator mode; "
                         "required for value mode.")
    ap.add_argument("--n-random-strides", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    # infer num_classes from filename if not given
    base = os.path.basename(args.csv).lower()
    if args.num_classes is None:
        args.num_classes = 100 if "cifar100" in base else (
            1000 if "imagenet" in base else 10)
    print(f"[cfg] arch={args.arch}  num_classes={args.num_classes}  score={args.score}")
    print(f"[cfg] sensitive-mode={args.sensitive_mode}  group={args.group_size}  "
          f"max_sens={args.max_sens}")

    df = pd.read_csv(args.csv)
    need = {"layer", "flat_idx", args.score}
    if need - set(df.columns):
        raise SystemExit(f"CSV missing columns: {sorted(need - set(df.columns))}")
    df["flat_idx"] = df["flat_idx"].astype(np.int64)
    df[args.score] = df[args.score].astype(float)

    layers = layer_order(df)
    print(f"[csv] {len(df)} rows across {len(layers)} layers: {layers}")

    numel_map, src = get_numel_map(layers, args.arch, args.num_classes, args.shapes_json)
    print(f"[numel] source = {src}")
    for L in layers:
        if L not in numel_map:
            print(f"[numel] WARNING: no size for '{L}'. It will be skipped. "
                  f"Provide it via --shapes-json.")

    if args.out_dir is None:
        stem = os.path.splitext(os.path.basename(args.csv))[0]
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                    f"patterns_{stem}")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[out] {args.out_dir}")

    fp = None
    if args.run_search:
        fp = _import_find_pattern(args.find_pattern_path)
        thr = args.threshold
        if thr is None:
            if args.sensitive_mode == "indicator":
                thr = 0.5
            else:
                raise SystemExit("value mode + --run-search requires --threshold")

    # combined flat-index export
    flat_export = []
    search_summary = []

    for L in layers:
        if L not in numel_map:
            continue
        N = numel_map[L]
        sub = df[df["layer"] == L]
        fidx = sub["flat_idx"].to_numpy()
        score = sub[args.score].to_numpy()

        # sanity: indices must be within N
        bad = fidx[(fidx < 0) | (fidx >= N)]
        if bad.size:
            print(f"[{L}] WARNING: {bad.size} flat_idx outside [0,{N}); "
                  f"check arch/num_classes. Clipping them out.")
            keep = (fidx >= 0) & (fidx < N)
            fidx, score = fidx[keep], score[keep]

        # dense array
        sens = np.zeros(N, dtype=np.float32)
        sens[fidx] = 1.0 if args.sensitive_mode == "indicator" else score

        npy = os.path.join(args.out_dir, f"{_sanitize(L)}_sens.npy")
        np.save(npy, sens)
        n_sens = int((sens > (args.threshold if args.threshold is not None else 0.5)).sum()) \
            if args.sensitive_mode == "indicator" else int((sens > 0).sum())
        print(f"\n[{L}]  N={N:,}  selected={fidx.size}  "
              f"density={fidx.size / N * 100:.4f}%  -> {npy}")

        for fi, sc in zip(fidx.tolist(), score.tolist()):
            flat_export.append({"layer": L, "flat_idx": fi,
                                "numel": N, args.score: sc})

        # optional search
        if args.run_search:
            if N < args.group_size * 2:
                print(f"   [skip search] layer too small for group_size")
                continue
            results = fp.search(sens, group_size=args.group_size,
                                threshold=thr, max_sens=args.max_sens,
                                n_random_strides=args.n_random_strides,
                                seed=args.seed, verbose=False)
            baseline = fp.evaluate(sens, np.arange(N), args.group_size,
                                   thr, args.max_sens)
            best = results[0]
            bm = best["metrics"]
            desc = fp.hardware_description(best["family"], best["param"],
                                           N, args.group_size)
            print(f"   identity excess = {baseline['total_excess']:,}  "
                  f"-> best({best['family']},{best['param']}) excess = "
                  f"{bm['total_excess']:,}  (max/group {baseline['max_in_group']}"
                  f"->{bm['max_in_group']})")
            txt = os.path.join(args.out_dir, f"{_sanitize(L)}_best_pattern.txt")
            with open(txt, "w") as f:
                f.write(f"layer: {L}\nN: {N}\ngroup_size: {args.group_size}\n"
                        f"max_sens: {args.max_sens}\nthreshold: {thr}\n\n")
                f.write(f"BEST: family={best['family']} param={best['param']}\n")
                f.write(f"metrics: {bm}\n")
                f.write(f"identity baseline: {baseline}\n\n")
                f.write(desc + "\n")
            search_summary.append({
                "layer": L, "N": N, "selected": int(fidx.size),
                "best_family": best["family"], "best_param": best["param"],
                "identity_excess": baseline["total_excess"],
                "best_excess": bm["total_excess"],
                "identity_max_in_group": baseline["max_in_group"],
                "best_max_in_group": bm["max_in_group"],
                "best_violating_groups": bm["violating_groups"],
            })

    # write combined exports
    fe = pd.DataFrame(flat_export)
    fe_path = os.path.join(args.out_dir, "sensitive_flatidx_by_layer.csv")
    fe.to_csv(fe_path, index=False)
    print(f"\n[saved] {fe_path}  ({len(fe)} rows)")

    if search_summary:
        ss = pd.DataFrame(search_summary)
        ss_path = os.path.join(args.out_dir, "pattern_search_summary.csv")
        ss.to_csv(ss_path, index=False)
        print(f"[saved] {ss_path}")
        print("\n=== pattern search summary ===")
        print(ss.to_string(index=False))

    print(f"\n[done] {args.out_dir}")


if __name__ == "__main__":
    main()


# python3 prepare_patterns.py \
#     --csv /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Quantization/artifacts/sensitivity/layer_then_weight_cifar10_resnet18_float32_L5xN200_grad_norm.csv \
#     --arch resnet18 \
#     --run-search --group-size 8 --max-sens 2