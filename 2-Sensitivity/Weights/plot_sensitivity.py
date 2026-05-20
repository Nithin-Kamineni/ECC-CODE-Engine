#!/usr/bin/env python3
"""
plot_layer_taylor.py
--------------------
Visualise the per-weight sensitivity stored in a
`layer_then_weight_*_L<K>xN<N>_*.csv` file produced by layer_then_weight.py.

It makes two kinds of figures:

  (A)  PER-LAYER SENSITIVITY  (default: the FIRST layer in the file)
       Three panels for the chosen layer:
         1. score vs in_layer_rank        -> how fast sensitivity decays
         2. score vs flat_idx             -> WHERE in the (flattened) tensor
                                             the sensitive weights sit
         3. histogram of log10(score)     -> spread of score magnitudes

  (B)  FLAT_IDX CLUSTERING  (across ALL layers in the file)
       - a strip/rug plot: every selected weight drawn as a tick along its
         layer's flat-index axis (each layer normalised to its own 0..1 span)
         -> see at a glance whether the chosen weights bunch together or
            spread out inside the tensor.
       - an ECDF of consecutive flat_idx gaps per layer (log-x)
         -> small gaps  = clustered;  large gaps = spread out.
       A numeric summary (count, span, median/mean gap) is printed and saved.

------------------------------------------------------------------------------
IMPORTANT CAVEAT
------------------------------------------------------------------------------
This CSV contains only the TOP-N weights per layer that layer_then_weight.py
selected (e.g. top 200), NOT every weight in the layer.  So "all weights in a
layer" here = "all *selected* weights for that layer".

To plot the TRUE full-tensor sensitivity (every weight), generate the dense
dump first:

    python sensitivity.py --dataset CIFAR10 --arch resnet18 \
        --methods magnitude grad_abs taylor fisher --dump-weights

and read the resulting  perweight_*_full.npz  instead (see plot_sensitivity.py).
------------------------------------------------------------------------------

Examples
--------
  # first layer in the file (conv1.weight), taylor score:
  python plot_layer_taylor.py --csv layer_then_weight_cifar10_resnet18_float32_L5xN200_grad_norm.csv

  # a specific layer, fisher score, custom output dir:
  python plot_layer_taylor.py --csv <file>.csv --layer layer4.1.conv2.weight \
      --score fisher --out-dir myplots

Author: helper script for Habibur Rahaman's ECC-resilience pipeline.
"""

from __future__ import annotations
import os
import argparse
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")            # headless / cluster-safe
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def _sanitize(name: str) -> str:
    """Make a layer name safe to embed in a filename."""
    return name.replace(".", "_").replace("/", "_").replace(" ", "_")


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"layer", "in_layer_rank", "flat_idx", "w",
                "magnitude", "taylor", "fisher"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    # make sure numeric columns are numeric
    for c in ["in_layer_rank", "flat_idx"]:
        df[c] = df[c].astype(np.int64)
    for c in ["w", "magnitude", "taylor", "fisher"]:
        df[c] = df[c].astype(float)
    return df


def layer_order(df: pd.DataFrame):
    """Layers in the order they first appear in the file (== layer_rank order)."""
    seen = []
    for L in df["layer"].tolist():
        if L not in seen:
            seen.append(L)
    return seen


# --------------------------------------------------------------------------
# (A) per-layer sensitivity figure
# --------------------------------------------------------------------------
def plot_layer_sensitivity(df: pd.DataFrame, layer: str, score: str,
                           out_dir: str, dpi: int = 130) -> str:
    sub = df[df["layer"] == layer].copy()
    if sub.empty:
        raise ValueError(f"layer '{layer}' not found in file")
    sub = sub.sort_values("in_layer_rank")

    rank = sub["in_layer_rank"].to_numpy()
    fidx = sub["flat_idx"].to_numpy()
    sc   = sub[score].to_numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # 1) decay curve: score vs in_layer_rank
    ax = axes[0]
    ax.plot(rank, sc, marker="o", ms=3, lw=1, color="darkred")
    ax.set_xlabel("in_layer_rank  (0 = most sensitive)")
    ax.set_ylabel(f"{score} score")
    ax.set_title("Sensitivity decay within layer")
    if sc.max() > 0 and sc[sc > 0].min() > 0 and sc.max() / sc[sc > 0].min() > 50:
        ax.set_yscale("log")
    ax.grid(alpha=0.3)

    # 2) positional scatter: score vs flat_idx
    ax = axes[1]
    sc_norm = (sc - sc.min()) / (sc.max() - sc.min() + 1e-30)
    s = ax.scatter(fidx, sc, c=sc_norm, cmap="viridis", s=18, edgecolor="none")
    ax.set_xlabel("flat_idx  (position in flattened tensor)")
    ax.set_ylabel(f"{score} score")
    ax.set_title("Where the sensitive weights sit")
    ax.grid(alpha=0.3)
    fig.colorbar(s, ax=ax, label="normalised score")

    # 3) histogram of log10(score)
    ax = axes[2]
    pos = sc[sc > 0]
    if pos.size:
        ax.hist(np.log10(pos), bins=40, color="steelblue", edgecolor="white")
    ax.set_xlabel(f"log10({score})")
    ax.set_ylabel("count")
    ax.set_title("Score distribution")
    ax.grid(alpha=0.3)

    fig.suptitle(f"{layer}   |   {len(sub)} selected weights   |   score = {score}",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = os.path.join(out_dir, f"layer_{score}_{_sanitize(layer)}.png")
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    print(f"  [saved] {out}")
    return out


# --------------------------------------------------------------------------
# (B) flat_idx clustering across all layers
# --------------------------------------------------------------------------
def _gaps(sorted_idx: np.ndarray) -> np.ndarray:
    """Consecutive differences of sorted flat indices."""
    if sorted_idx.size < 2:
        return np.array([], dtype=np.int64)
    return np.diff(sorted_idx)


def plot_flatidx_clustering(df: pd.DataFrame, out_dir: str,
                            dpi: int = 130) -> tuple[str, str, str]:
    layers = layer_order(df)
    n = len(layers)

    # ---- numeric summary ----
    summary_rows = []
    per_layer_idx = {}
    for L in layers:
        idx = np.sort(df[df["layer"] == L]["flat_idx"].to_numpy())
        per_layer_idx[L] = idx
        g = _gaps(idx)
        summary_rows.append({
            "layer": L,
            "n_selected": idx.size,
            "min_idx": int(idx.min()) if idx.size else 0,
            "max_idx": int(idx.max()) if idx.size else 0,
            "span": int(idx.max() - idx.min()) if idx.size else 0,
            "median_gap": float(np.median(g)) if g.size else 0.0,
            "mean_gap": float(g.mean()) if g.size else 0.0,
        })
    summary = pd.DataFrame(summary_rows)
    csv_out = os.path.join(out_dir, "flatidx_clustering_summary.csv")
    summary.to_csv(csv_out, index=False)
    print(f"  [saved] {csv_out}")
    print("\n  flat_idx clustering summary:")
    print(summary.to_string(index=False))

    # ---- strip / rug plot (normalised per layer) ----
    fig, ax = plt.subplots(figsize=(11, 0.9 * n + 1.5))
    for row, L in enumerate(layers):
        idx = per_layer_idx[L]
        span = idx.max() - idx.min() if idx.size else 1
        span = span if span > 0 else 1
        x = (idx - idx.min()) / span                      # normalise to 0..1
        ax.eventplot(positions=x, lineoffsets=row, linelengths=0.8,
                     colors="steelblue", linewidths=0.7)
    ax.set_yticks(range(n))
    ax.set_yticklabels(layers, fontsize=8)
    ax.set_xlabel("position within layer's selected-index span  (0 = first, 1 = last)")
    ax.set_title("flat_idx clustering per layer  "
                 "(each layer normalised to its own min..max span)")
    ax.set_xlim(-0.02, 1.02)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    strip_out = os.path.join(out_dir, "flatidx_clustering_strip.png")
    fig.savefig(strip_out, dpi=dpi)
    plt.close(fig)
    print(f"  [saved] {strip_out}")

    # ---- ECDF of gaps per layer (log-x) ----
    fig, ax = plt.subplots(figsize=(8, 5))
    cmap = plt.get_cmap("tab10")
    for i, L in enumerate(layers):
        g = _gaps(per_layer_idx[L])
        if g.size == 0:
            continue
        gs = np.sort(g)
        y = np.arange(1, gs.size + 1) / gs.size
        ax.step(np.maximum(gs, 1), y, where="post",
                label=L, color=cmap(i % 10), lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("gap between consecutive selected flat_idx  (log scale)")
    ax.set_ylabel("cumulative fraction of gaps")
    ax.set_title("How tightly packed are the selected weights?\n"
                 "(curve shifted LEFT = more clustered, RIGHT = more spread)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    ecdf_out = os.path.join(out_dir, "flatidx_gaps_ecdf.png")
    fig.savefig(ecdf_out, dpi=dpi)
    plt.close(fig)
    print(f"  [saved] {ecdf_out}")

    return strip_out, ecdf_out, csv_out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        "Plot per-layer Taylor/Fisher sensitivity and flat_idx clustering "
        "from a layer_then_weight CSV")
    ap.add_argument("--csv", required=True,
                    help="Path to layer_then_weight_*.csv")
    ap.add_argument("--layer", default=None,
                    help="Which layer to plot for figure (A). "
                         "Default = first layer in the file.")
    ap.add_argument("--score", default="taylor",
                    choices=["taylor", "fisher", "magnitude"],
                    help="Which per-weight score to plot. Default: taylor")
    ap.add_argument("--all-layers", action="store_true",
                    help="Make a per-layer figure (A) for EVERY layer, "
                         "not just the first.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Default: <csv_dir>/plots_<csvname>")
    ap.add_argument("--dpi", type=int, default=130)
    args = ap.parse_args()

    df = load_csv(args.csv)
    layers = layer_order(df)
    print(f"[load] {args.csv}")
    print(f"[load] {len(df)} rows across {len(layers)} layers: {layers}")

    if args.out_dir is None:
        base = os.path.splitext(os.path.basename(args.csv))[0]
        args.out_dir = os.path.join(os.path.dirname(os.path.abspath(args.csv)),
                                    f"plots_{base}")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[out ] {args.out_dir}")

    # ---- (A) per-layer sensitivity ----
    print("\n[A] per-layer sensitivity figure(s):")
    targets = layers if args.all_layers else [args.layer or layers[0]]
    for L in targets:
        plot_layer_sensitivity(df, L, args.score, args.out_dir, dpi=args.dpi)

    # ---- (B) flat_idx clustering ----
    print("\n[B] flat_idx clustering figures:")
    plot_flatidx_clustering(df, args.out_dir, dpi=args.dpi)

    print(f"\n[done] all figures in {args.out_dir}")


if __name__ == "__main__":
    main()

# python3 plot_sensitivity.py --csv /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/Quantization/artifacts/sensitivity/layer_then_weight_cifar10_resnet18_float32_L5xN200_grad_norm.csv
