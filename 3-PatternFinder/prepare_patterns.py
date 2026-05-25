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


def _load_sd_from_ckpt(model_path: str) -> dict:
    """Return a plain {name: tensor} state dict from a float32 or PTQ checkpoint.

    PTQ checkpoints use qstate_dict + meta.scales; weights are dequantized back
    to float32 so the existing weight-permutation code can use them unchanged.
    """
    import torch
    ckpt = torch.load(model_path, map_location="cpu")
    if "qstate_dict" in ckpt and "meta" in ckpt:
        qsd    = ckpt["qstate_dict"]
        scales = ckpt["meta"]["scales"]
        sd: dict = {}
        for k, v in qsd.items():
            sinfo = scales.get(k)
            if sinfo is None:
                sd[k] = v
            elif sinfo["type"] == "per_tensor":
                sd[k] = v.to(torch.float32) * float(sinfo["scale"])
            elif sinfo["type"] == "per_channel":
                s = sinfo["scales"].to(torch.float32)
                q = v.to(torch.float32)
                while s.ndim < q.ndim:
                    s = s.unsqueeze(-1)
                sd[k] = q * s
            else:
                sd[k] = v
        return sd
    return ckpt.get("state_dict", ckpt)


def _load_raw_sd_from_ckpt(model_path: str) -> dict:
    """Return {name: numpy_array} with raw quantized integers for PTQ checkpoints.
    Does NOT dequantize — preserves int8/int16 values as stored, so ECC embedding
    operates on the actual bit patterns rather than float approximations."""
    import torch
    ckpt = torch.load(model_path, map_location="cpu")
    raw = ckpt["qstate_dict"] if "qstate_dict" in ckpt else ckpt.get("state_dict", ckpt)
    sd: dict = {}
    for k, v in raw.items():
        nk = k
        for pfx in ("module.", "_orig_mod."):
            if nk.startswith(pfx):
                nk = nk[len(pfx):]
        sd[nk] = v.detach().cpu().numpy() if hasattr(v, "detach") else v
    return sd


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
    ap.add_argument("--identity-perm", action="store_true",
                    help="Skip search; save identity permutation and original "
                         "weights for every layer (DISABLE_PATTERN_FIND mode).")
    ap.add_argument("--find-pattern-path", default=None)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-sens", type=int, default=2)
    ap.add_argument("--threshold", type=float, default=None,
                    help="Sensitivity cutoff. Default 0.5 for indicator mode; "
                         "required for value mode.")
    ap.add_argument("--top-sensitive", type=int, default=100,
                    help="Minimum sensitive nodes to mark per layer. "
                         "Final count = max(threshold_count, top_sensitive).")
    ap.add_argument("--max-stride", type=int, default=256,
                    help="Maximum allowed stride s (hardware burst-fetch size in weights). "
                         "All s in [2, min(max_stride, N-1)] are evaluated exhaustively.")

    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--model-path", default=None,
                    help="Path to model .pth checkpoint. If given, saves permuted "
                         "weight tensors alongside the permutation index files.")
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

    # load model state dict for weight extraction (if checkpoint provided)
    # Use raw int8 values so ECC embedding operates on the actual quantized bit patterns.
    sd = {}
    if args.model_path and (args.run_search or args.identity_perm):
        sd = _load_raw_sd_from_ckpt(args.model_path)
        print(f"[model] loaded {len(sd)} tensors from {args.model_path}")

    # combined flat-index export
    flat_export  = []
    search_summary = []
    manifest     = {}

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

        # Sort by Taylor score descending so top-N selection is straightforward
        order = np.argsort(score)[::-1]
        fidx_sorted  = fidx[order]
        score_sorted = score[order]

        # Determine how many weights to mark as sensitive:
        #   threshold_count = weights whose score exceeds the threshold
        #   n_mark = max(threshold_count, top_sensitive)  — whichever is larger
        thr_val = args.threshold if args.threshold is not None else 0.5
        threshold_count = int((score_sorted > thr_val).sum())
        n_mark = min(max(threshold_count, args.top_sensitive), len(fidx_sorted))

        fidx_to_mark  = fidx_sorted[:n_mark]
        score_to_mark = score_sorted[:n_mark]

        # dense array
        sens = np.zeros(N, dtype=np.float32)
        sens[fidx_to_mark] = 1.0 if args.sensitive_mode == "indicator" else score_to_mark

        npy = os.path.join(args.out_dir, f"{_sanitize(L)}_sens.npy")
        np.save(npy, sens)
        print(f"\n[{L}]  N={N:,}  in_csv={fidx.size}  "
              f"threshold_count={threshold_count}  top_sensitive={args.top_sensitive}  marking={n_mark}"
              f"  density={n_mark / N * 100:.4f}%  -> {npy}")

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
                                max_stride=args.max_stride,
                                verbose=False)
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

            # --- permutation index files ---
            perm = fp.make_perm(best["family"], best["param"], N)
            # Guard: non-bijective perm would corrupt weights_perm_file and inv_perm.
            # This should never trigger after the find_pattern.py fix (coprime-only),
            # but keep it as a hard safety net.
            from math import gcd as _gcd
            if len(np.unique(perm)) != N:
                raise RuntimeError(
                    f"[{L}] best stride s={best['param']} is non-bijective "
                    f"(gcd={_gcd(int(best['param']), N)}, "
                    f"{len(np.unique(perm))} unique values / {N} needed). "
                    f"This would corrupt weights_perm_file and inv_perm — aborting."
                )
            inv_perm = np.zeros(N, dtype=perm.dtype)   # zeros, not empty_like
            inv_perm[perm] = np.arange(N, dtype=perm.dtype)

            perm_file     = os.path.join(args.out_dir, f"{_sanitize(L)}_perm.npy")
            inv_perm_file = os.path.join(args.out_dir, f"{_sanitize(L)}_inv_perm.npy")
            np.save(perm_file, perm)
            np.save(inv_perm_file, inv_perm)

            # --- permuted weights (only if checkpoint was loaded) ---
            w_perm_file = None
            layer_shape = None
            if L in sd:
                layer_shape = list(sd[L].shape)
                w_flat      = sd[L].flatten()
                w_perm      = w_flat[perm]
                w_perm_file = os.path.join(args.out_dir,
                                           f"{_sanitize(L)}_weights_perm.npy")
                np.save(w_perm_file, w_perm)

            txt = os.path.join(args.out_dir, f"{_sanitize(L)}_best_pattern.txt")
            with open(txt, "w") as f:
                f.write(f"layer: {L}\nN: {N}\ngroup_size: {args.group_size}\n"
                        f"max_sens: {args.max_sens}\nthreshold: {thr}\n\n")
                f.write(f"BEST: family={best['family']} param={best['param']}\n")
                f.write(f"metrics: {bm}\n")
                f.write(f"identity baseline: {baseline}\n\n")
                f.write(desc + "\n")

            # --- manifest entry for this layer ---
            manifest[L] = {
                "layer":         L,
                "N":             N,
                "shape":         layer_shape,
                "group_size":    args.group_size,
                "max_sens":      args.max_sens,
                "threshold":     float(thr),
                "top_sensitive": args.top_sensitive,
                "n_marked":      int(n_mark),
                "best_family":   best["family"],
                "best_param":    best["param"],
                "identity_excess":        baseline["total_excess"],
                "best_excess":            bm["total_excess"],
                "best_violating_groups":  bm["violating_groups"],
                "best_max_in_group":      bm["max_in_group"],
                "model_path":    args.model_path,
                "perm_file":     perm_file,
                "inv_perm_file": inv_perm_file,
                "weights_perm_file": w_perm_file,
            }

            search_summary.append({
                "layer": L, "N": N, "selected": int(fidx.size),
                "best_family": best["family"], "best_param": best["param"],
                "identity_excess": baseline["total_excess"],
                "best_excess": bm["total_excess"],
                "identity_max_in_group": baseline["max_in_group"],
                "best_max_in_group": bm["max_in_group"],
                "best_violating_groups": bm["violating_groups"],
            })

        elif args.identity_perm:
            perm     = np.arange(N, dtype=np.int64)
            inv_perm = np.arange(N, dtype=np.int64)   # identity is its own inverse

            perm_file     = os.path.join(args.out_dir, f"{_sanitize(L)}_perm.npy")
            inv_perm_file = os.path.join(args.out_dir, f"{_sanitize(L)}_inv_perm.npy")
            np.save(perm_file, perm)
            np.save(inv_perm_file, inv_perm)

            w_perm_file = None
            layer_shape = None
            if L in sd:
                layer_shape = list(sd[L].shape)
                w_perm_file = os.path.join(args.out_dir,
                                           f"{_sanitize(L)}_weights_perm.npy")
                np.save(w_perm_file, sd[L].flatten())   # original order, no reordering

            txt = os.path.join(args.out_dir, f"{_sanitize(L)}_best_pattern.txt")
            with open(txt, "w") as f:
                f.write(f"layer: {L}\nN: {N}\npattern: identity (search disabled)\n")

            manifest[L] = {
                "layer":                 L,
                "N":                     N,
                "shape":                 layer_shape,
                "group_size":            args.group_size,
                "max_sens":              args.max_sens,
                "threshold":             None,
                "top_sensitive":         args.top_sensitive,
                "n_marked":              int(n_mark),
                "best_family":           "identity",
                "best_param":            None,
                "identity_excess":       None,
                "best_excess":           None,
                "best_violating_groups": None,
                "best_max_in_group":     None,
                "model_path":            args.model_path,
                "perm_file":             perm_file,
                "inv_perm_file":         inv_perm_file,
                "weights_perm_file":     w_perm_file,
            }
            search_summary.append({
                "layer": L, "N": N, "selected": int(fidx.size),
                "best_family": "identity", "best_param": None,
                "identity_excess": None, "best_excess": None,
                "identity_max_in_group": None, "best_max_in_group": None,
                "best_violating_groups": None,
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

    if manifest:
        import json
        manifest_path = os.path.join(args.out_dir, "pattern_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"[saved] {manifest_path}  ({len(manifest)} layers)")

    print(f"\n[done] {args.out_dir}")


if __name__ == "__main__":
    main()


# python3 prepare_patterns.py \
#     --csv /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Quantization/artifacts/sensitivity/layer_then_weight_cifar10_resnet18_float32_L5xN200_grad_norm.csv \
#     --arch resnet18 \
#     --run-search --group-size 8 --max-sens 2