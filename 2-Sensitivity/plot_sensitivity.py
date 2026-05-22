# plot_sensitivity.py
# Turn the per-weight .npz files produced by sensitivity.py into per-layer
# heatmap images, so you can visually answer the "spread vs. clustered"
# question for your collaborator.
#
# Author: Habibur Rahaman, University of Florida, ECE Department
#
# What it produces (per layer, for a chosen sensitivity method):
#   1) Filter heatmap (Conv2d only):
#        x-axis = output channel (0..OC-1)
#        y-axis = input  channel (0..IC-1)
#        color  = max sensitivity within that (out, in) filter slot
#      -> shows whether sensitivity clusters into specific input/output
#         channel pairs or is scattered.
#   2) Histogram of per-weight scores (log-scale y).
#   3) Sorted-score curve (cumulative sensitivity vs sorted weight rank)
#      -> a straight line means uniform spread; a sharp knee means clustered.
#
# Examples:
#   # plot taylor scores for every layer in the float32 dump:
#   python plot_sensitivity.py \
#       --npz artifacts/sensitivity/perweight_cifar10_resnet18_float32_full.npz \
#       --method taylor --out-dir artifacts/sensitivity/plots
#
#   # plot just a few layers:
#   python plot_sensitivity.py \
#       --npz artifacts/sensitivity/perweight_cifar10_resnet18_float32_full.npz \
#       --method taylor --layers conv1.weight layer4.0.conv2.weight

from __future__ import annotations
import os, argparse
import numpy as np


def _load_layers(npz_path: str):
    """
    Returns dict: layer_name -> {'w': ndarray, 'shape': tuple, <method>: ndarray, ...}
    Only handles the 'full' dump format.
    """
    data = np.load(npz_path, allow_pickle=True)
    layers = {}
    for k in data.files:
        if "__" not in k:
            continue
        prefix, sub = k.rsplit("__", 1)
        lname = prefix.replace("_", ".")
        # heuristic: collapse weights like 'layer4.0.conv2.weight'
        # We can't perfectly reverse '_' -> '.', but for torchvision models it's fine.
        layers.setdefault(prefix, {})[sub] = data[k]
    return layers


def _save_filter_heatmap(score_flat, shape, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if len(shape) != 4:
        return False  # FC layer, skip
    OC, IC, KH, KW = shape
    s = score_flat.reshape(OC, IC, KH, KW)
    # collapse spatial dims with max -> per (out, in) cell
    cell = s.max(axis=(2, 3))   # shape (OC, IC)
    fig, ax = plt.subplots(figsize=(min(12, max(4, OC / 8)),
                                     min(12, max(4, IC / 8))))
    im = ax.imshow(cell.T, aspect="auto", cmap="viridis", origin="lower")
    ax.set_xlabel("output channel")
    ax.set_ylabel("input channel")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="max score in filter")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def _save_fc_heatmap(score_flat, shape, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if len(shape) != 2:
        return False
    OUT, IN = shape
    s = score_flat.reshape(OUT, IN)
    fig, ax = plt.subplots(figsize=(min(12, max(4, IN / 64)),
                                     min(12, max(4, OUT / 64))))
    im = ax.imshow(s, aspect="auto", cmap="viridis")
    ax.set_xlabel("input feature")
    ax.set_ylabel("output class")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="score")
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def _save_hist(score_flat, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4))
    s = score_flat[score_flat > 0]
    if s.size == 0:
        plt.close(fig); return False
    ax.hist(np.log10(s + 1e-30), bins=80, color="steelblue", edgecolor="white")
    ax.set_xlabel("log10(score)")
    ax.set_ylabel("count")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def _save_sorted_curve(score_flat, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    s = np.sort(score_flat)[::-1]
    cum = np.cumsum(s) / max(1e-30, s.sum())
    x = np.arange(1, s.size + 1) / s.size
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, cum, color="darkred")
    ax.plot([0, 1], [0, 1], "--", color="gray", label="uniform")
    ax.set_xlabel("fraction of weights (ranked by score)")
    ax.set_ylabel("cumulative fraction of total sensitivity")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=120)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser("Plot per-layer sensitivity heatmaps")
    ap.add_argument("--npz", required=True,
                    help="Path to perweight_*_full.npz produced by sensitivity.py")
    ap.add_argument("--method", default="taylor",
                    help="Which score column to plot (taylor / fisher / grad_abs / "
                         "magnitude / hessian / quant_err_intN)")
    ap.add_argument("--layers", nargs="*",
                    help="Optional: only plot these layer names (substrings).")
    ap.add_argument("--out-dir", default="artifacts/sensitivity/plots")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    layers = _load_layers(args.npz)
    print(f"[plot] loaded {len(layers)} layers from {args.npz}")
    print(f"[plot] available columns per layer: example "
          f"{list(next(iter(layers.values())).keys())}")

    base = os.path.splitext(os.path.basename(args.npz))[0]

    n_done = 0
    for lname_under, blocks in layers.items():
        lname = lname_under.replace("_", ".")
        if args.layers and not any(f in lname for f in args.layers):
            continue
        if args.method not in blocks:
            continue
        if "shape" not in blocks:
            continue
        score = blocks[args.method]
        shape = tuple(blocks["shape"].tolist())
        title_root = f"{lname}  ({args.method})"

        # filter heatmap (conv only) or fc heatmap
        if len(shape) == 4:
            png = os.path.join(args.out_dir, f"{base}__{lname_under}__filtermap.png")
            ok = _save_filter_heatmap(score, shape, png, title_root + "  -- filter map")
            if ok: print(f"  {png}")
        elif len(shape) == 2:
            png = os.path.join(args.out_dir, f"{base}__{lname_under}__fcmap.png")
            ok = _save_fc_heatmap(score, shape, png, title_root + "  -- FC map")
            if ok: print(f"  {png}")

        # hist
        png = os.path.join(args.out_dir, f"{base}__{lname_under}__hist.png")
        if _save_hist(score, png, title_root + "  -- histogram"):
            print(f"  {png}")

        # sorted curve
        png = os.path.join(args.out_dir, f"{base}__{lname_under}__cumulative.png")
        if _save_sorted_curve(score, png, title_root + "  -- cumulative"):
            print(f"  {png}")

        n_done += 1

    print(f"[plot] plotted {n_done} layer(s) into {args.out_dir}")


if __name__ == "__main__":
    main()
