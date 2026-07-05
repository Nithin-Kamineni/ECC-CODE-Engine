#!/usr/bin/env python3
"""
export_sensitivity.py — Pre-compute Python sensitivity arrays for C++.

For each layer in the pattern manifest, replicates the EXACT same sensitivity
computation as ecc_embed.py's _load_layer_sensitivity():

  1. Read Taylor gradient scores from the sensitivity CSV:
         {sensitivity_dir}/{ds}/{arch}/{qmode}/{bits}-bit/
         layer_then_weight_{ds}_{arch}_int{n}_L999xN30000_grad_norm.csv
  2. Weights absent from the CSV get baseline = min(taylor scores in layer).
  3. Build array aligned to the weight permutation order (same as Python).
  4. Normalize to [min_norm, 1.0] (--sens-norm-min, default 0.5):
         sens = min_norm + (1 - min_norm) * (taylor / max_taylor)
  5. Save as float32 → {patterns_dir}/{ds}/{arch}/{qmode}/{bits}/{layer}_sens_py.npy

  Additionally exports a permutation-INDEPENDENT array, indexed by the
  original (unpermuted) flat_idx and normalized the same way:
         {patterns_dir}/{ds}/{arch}/{qmode}/{bits}/{layer}_sens_dense_py.npy
  Since normalization divides by max(sens_dict.values()) — which does not
  depend on the permutation — this single dense array can be re-permuted in
  C++ for ANY candidate pattern's perm_file via sens_vec[k] = dense[perm[k]],
  which is what the top-K greedy pattern scoring needs.

C++ reads these files directly (no further permutation or normalization for
_sens_py.npy), so it uses the same continuous sensitivity values as Python —
not the binary {0.5, 1.0} indicator values from _sens.npy.

This follows the same pattern as export_parity_matrix.py (BCH matrix export).
Run once per (dataset, arch, quant-bits) combination; files are cached and
reused on subsequent runs unless --force is passed.

Usage:
  python3 export_sensitivity.py \\
      --dataset CIFAR10 --arch resnet18 --quant-bits 8 \\
      --patterns-dir  /blue/.../patterns \\
      --sensitivity-dir /blue/.../sensitivity \\
      --qmode PTQ
"""
import argparse
import json
import os
import sys
import numpy as np

try:
    import pandas as pd
except ImportError:
    print("[export_sensitivity] ERROR: pandas is required but not installed.", file=sys.stderr)
    sys.exit(1)


def _sanitize(name: str) -> str:
    """Match Python ecc_embed.py's _sanitize() exactly."""
    return name.replace('.', '_').replace('/', '_')


def _load_layer_sensitivity(sensitivity_dir: str, ds_lower: str, arch: str,
                             quant_bits: int, layer_name: str,
                             perm_arr, qmode: str = "PTQ",
                             min_norm: float = 0.5) -> np.ndarray:
    """
    Exact copy of ecc_embed.py's _load_layer_sensitivity().

    Returns float64 ndarray of shape (N,) with values in [min_norm, 1.0],
    already aligned to the permuted weight order (perm_arr applied).
    """
    bit_label = f"{quant_bits}-bit"
    csv_path = os.path.join(
        sensitivity_dir, ds_lower, arch, qmode, bit_label,
        f"layer_then_weight_{ds_lower}_{arch}_int{quant_bits}_L999xN30000_grad_norm.csv",
    )
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    layer_df = df[df["layer"] == layer_name][["flat_idx", "taylor"]]
    sens_dict = {int(row["flat_idx"]): float(row["taylor"])
                 for _, row in layer_df.iterrows()}

    N = len(perm_arr)
    baseline = min(sens_dict.values()) if sens_dict else 0.0

    # Build in permuted order — same as Python
    sens_arr = np.array(
        [sens_dict.get(int(perm_arr[i]), baseline) for i in range(N)],
        dtype=np.float64,
    )

    max_val = float(sens_arr.max())
    if max_val > 0.0:
        sens_arr = min_norm + (1.0 - min_norm) * (sens_arr / max_val)   # normalize to [min_norm, 1.0]
    else:
        sens_arr[:] = 1.0                               # uniform fallback

    return sens_arr


def main():
    ap = argparse.ArgumentParser(
        description="Export Python sensitivity arrays as .npy files for C++ ECC embedding.\n"
                    "Produces {layer}_sens_py.npy files alongside {layer}_sens.npy in patterns dir.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--dataset",          required=True,
                    choices=["CIFAR10", "CIFAR100", "IMAGENET"],
                    help="Dataset name (case-sensitive)")
    ap.add_argument("--arch",             required=True,
                    help="Architecture name (e.g. resnet18, mobilenet_v2)")
    ap.add_argument("--quant-bits",       required=True, type=int, choices=[4, 6, 8, 16],
                    help="Quantization bit-width")
    ap.add_argument("--patterns-dir",     required=True,
                    help="Root of 0-Data/artifacts/patterns/ (where _sens.npy files live)")
    ap.add_argument("--sensitivity-dir",  required=True,
                    help="Root of 0-Data/artifacts/sensitivity/ (where CSV files live)")
    ap.add_argument("--qmode", default="PTQ", choices=["PTQ", "QAT"],
                    help="PTQ or QAT — selects the patterns/ and sensitivity/ subdirectory.")
    ap.add_argument("--force",            action="store_true",
                    help="Re-export even if _sens_py.npy already exists")
    ap.add_argument("--sens-norm-min", type=float, default=0.5,
                    help="Floor of the Taylor-score normalization range [min,1.0] "
                         "(controlled by SENS_NORM_MIN in env.sh). 0.0 = raw "
                         "max-normalized scores; 1.0 = uniform (no sensitivity effect).")
    args = ap.parse_args()
    if not (0.0 <= args.sens_norm_min <= 1.0):
        raise ValueError(f"--sens-norm-min must be in [0.0, 1.0], got {args.sens_norm_min}")

    ds_lower  = args.dataset.lower()
    bit_label = f"{args.quant_bits}-bit"

    manifest_path = os.path.join(
        args.patterns_dir, ds_lower, args.arch, args.qmode, bit_label,
        "pattern_manifest.json",
    )
    if not os.path.exists(manifest_path):
        print(f"[export_sensitivity] No manifest found: {manifest_path}")
        sys.exit(0)

    with open(manifest_path) as fh:
        manifest = json.load(fh)

    print(f"[export_sensitivity] {args.dataset}/{args.arch}/{args.qmode}/{bit_label}")
    print(f"[export_sensitivity] manifest: {manifest_path}  ({len(manifest)} layers)")
    print(f"[export_sensitivity] patterns-dir:     {args.patterns_dir}")
    print(f"[export_sensitivity] sensitivity-dir:  {args.sensitivity_dir}")

    # Load the CSV once (it covers all layers), cache it for speed
    csv_path = os.path.join(
        args.sensitivity_dir, ds_lower, args.arch, args.qmode, bit_label,
        f"layer_then_weight_{ds_lower}_{args.arch}_int{args.quant_bits}_L999xN30000_grad_norm.csv",
    )
    if not os.path.exists(csv_path):
        print(f"[export_sensitivity] ERROR: CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    print(f"[export_sensitivity] Loading CSV: {csv_path}")
    df_all = pd.read_csv(csv_path)
    print(f"[export_sensitivity] CSV loaded: {len(df_all):,} rows")

    n_ok = n_skipped = n_failed = 0

    for layer_name, entry in manifest.items():
        perm_file = entry.get("perm_file")
        N_entry   = entry.get("N")
        if not perm_file or not os.path.exists(perm_file):
            print(f"  [skip] {layer_name}: perm_file missing ({perm_file})")
            n_skipped += 1
            continue

        layer_safe = _sanitize(layer_name)
        out_path = os.path.join(
            args.patterns_dir, ds_lower, args.arch, args.qmode, bit_label,
            f"{layer_safe}_sens_py.npy",
        )
        dense_out_path = os.path.join(
            args.patterns_dir, ds_lower, args.arch, args.qmode, bit_label,
            f"{layer_safe}_sens_dense_py.npy",
        )

        if os.path.exists(out_path) and os.path.exists(dense_out_path) and not args.force:
            n_skipped += 1
            continue  # already exported, reuse cached files

        try:
            perm_arr = np.load(perm_file)
            N = N_entry if N_entry is not None else len(perm_arr)

            # Per-layer sensitivity dict from pre-loaded CSV
            layer_df  = df_all[df_all["layer"] == layer_name][["flat_idx", "taylor"]]
            sens_dict = {int(r["flat_idx"]): float(r["taylor"])
                         for _, r in layer_df.iterrows()}

            baseline = min(sens_dict.values()) if sens_dict else 0.0

            # ---- Dense array, indexed by ORIGINAL flat_idx (permutation-independent) ----
            dense_arr = np.array(
                [sens_dict.get(i, baseline) for i in range(N)],
                dtype=np.float64,
            )
            max_val = float(dense_arr.max())
            if max_val > 0.0:
                dense_arr = args.sens_norm_min + (1.0 - args.sens_norm_min) * (dense_arr / max_val)
            else:
                dense_arr[:] = 1.0
            np.save(dense_out_path, dense_arr.astype(np.float32))

            # ---- Rank-0 permuted array (backward compat, _sens_py.npy) ----
            sens_arr = dense_arr[perm_arr]

            # Save as float32 — C++ reads float32 .npy files
            np.save(out_path, sens_arr.astype(np.float32))

            print(f"  [ok] {layer_name}: N={N:,}  "
                  f"in_CSV={len(sens_dict)}  "
                  f"range=[{sens_arr.min():.4f}, {sens_arr.max():.4f}]  "
                  f"→ {os.path.basename(out_path)}, {os.path.basename(dense_out_path)}")
            n_ok += 1

        except Exception as exc:
            print(f"  [error] {layer_name}: {exc}", file=sys.stderr)
            n_failed += 1

    print(f"\n[export_sensitivity] Done: {n_ok} exported, "
          f"{n_skipped} skipped (cached), {n_failed} failed")
    if n_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
