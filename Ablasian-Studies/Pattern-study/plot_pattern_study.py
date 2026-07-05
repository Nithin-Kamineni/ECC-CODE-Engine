"""
plot_pattern_study.py — Pattern study: codeword sensitive-weight distribution + accuracy table.

Reads staged results from run_pattern_study.sh and produces:
  1. A grouped bar chart (one bar group per x-bucket):
       x-axis : sensitive weights per codeword (0, 1, 2, ..., group_size+)
       y-axis L: number of codewords
       y-axis R: % of codewords with > k sensitive weights
       vertical dashed line at t (BCH error correction capacity)
     Three bar groups: Contiguous, Random perm, Strided (proposed)

  2. results/accuracy_table.json — top-1/top-5 for all three conditions.

Usage:
    python3 plot_pattern_study.py \
        --results-dir results \
        --dataset IMAGENET --arch efficientnet_b0 --qmode QAT \
        --quant-bits 8 --t 4 --approach search3
"""

import argparse
import json
import os
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Visual configuration — edit these two dicts to restyle the graph
# ---------------------------------------------------------------------------

FONT_SIZES = {
    "title":      14,
    "axis_label": 12,
    "tick_label": 10,
    "legend":     10,
    "annotation": 9,
}

STYLE = {
    "figure_size":    (11, 6),
    "dpi":            150,
    "bar_width":      0.25,       # width of each individual bar
    "bar_alpha":      0.90,
    "edge_width":     0.6,
    "bg_color":       "#1E1E1E",  # dark background (matches reference image)
    "fg_color":       "#DDDDDD",  # text / axis lines
    "grid_color":     "#444444",
    "grid_alpha":     0.4,
    "grid_style":     "--",
    "capacity_color": "#EF5350",  # dashed BCH capacity line
    "capacity_alpha": 0.85,
    "r_axis_color":   "#AAAAAA",  # right y-axis line colour
    # Per-condition colours and display labels
    "colors": {
        "contiguous": "#888888",
        "random":     "#F5A623",
        "strided":    "#26A69A",
    },
    "edge_colors": {
        "contiguous": "#AAAAAA",
        "random":     "#FFD080",
        "strided":    "#80CBC4",
    },
    "labels": {
        "contiguous": "Contiguous",
        "random":     "Random perm",
        "strided":    "Strided (proposed)",
    },
}

CONDITIONS = ["contiguous", "random", "strided"]

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


def log_warn(log_path, message):
    print(message)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(message + "\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_json(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        return None


def load_histograms(results_dir):
    """Load codeword_histogram.json for each condition. Returns dict or None per condition."""
    hists = {}
    for cond in CONDITIONS:
        path = os.path.join(results_dir, cond, "codeword_histogram.json")
        data = load_json(path)
        if data is None:
            print(f"[warn] histogram missing for condition={cond}: {path}")
        hists[cond] = data
    return hists


def load_accuracies(results_dir):
    """Load accuracy.json for each condition. Returns dict."""
    accs = {}
    for cond in CONDITIONS:
        path = os.path.join(results_dir, cond, "accuracy.json")
        accs[cond] = load_json(path)
    return accs


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_histogram(hists, t_value, title, out_path):
    """
    Grouped bar chart: sensitive weights per codeword.

    hists : dict condition → histogram_data dict (from codeword_histogram.json)
            histogram_data["histogram"] is a list of length group_size+1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Determine group_size from available data
    group_size = 8
    for data in hists.values():
        if data is not None:
            group_size = data.get("group_size", 8)
            break

    n_buckets = group_size + 1   # 0, 1, ..., group_size
    x = np.arange(n_buckets)

    fig, ax_l = plt.subplots(figsize=STYLE["figure_size"])
    fig.patch.set_facecolor(STYLE["bg_color"])
    ax_l.set_facecolor(STYLE["bg_color"])

    bw = STYLE["bar_width"]
    n_conds = len(CONDITIONS)
    offsets = np.linspace(-(n_conds - 1) / 2, (n_conds - 1) / 2, n_conds) * bw

    ax_r = ax_l.twinx()
    ax_r.set_facecolor(STYLE["bg_color"])

    any_data = False
    for cond, offset in zip(CONDITIONS, offsets):
        data = hists.get(cond)
        if data is None:
            continue
        hist = np.array(data["histogram"][:n_buckets], dtype=float)
        if len(hist) < n_buckets:
            hist = np.pad(hist, (0, n_buckets - len(hist)))
        total = hist.sum()
        if total == 0:
            continue
        any_data = True

        ax_l.bar(
            x + offset,
            hist,
            width=bw,
            label=STYLE["labels"][cond],
            color=STYLE["colors"][cond],
            edgecolor=STYLE["edge_colors"][cond],
            linewidth=STYLE["edge_width"],
            alpha=STYLE["bar_alpha"],
            zorder=3,
        )

        # Right axis: % codewords with > k sensitive weights (complementary CDF)
        cumsum = np.cumsum(hist)
        pct_gt_k = 100.0 * (total - cumsum) / total
        ax_r.plot(
            x + offset,
            pct_gt_k,
            color=STYLE["colors"][cond],
            linewidth=0,   # invisible line — markers only
            marker="D",
            markersize=4,
            alpha=0.7,
            zorder=4,
        )

    if not any_data:
        print(f"[error] no histogram data to plot — skipping {out_path}")
        return

    capacity = 4
    # BCH capacity dashed line
    ax_l.axvline(
        capacity - 0.75 + bw,
        color=STYLE["capacity_color"],
        linestyle="--",
        linewidth=1.6,
        alpha=STYLE["capacity_alpha"],
        zorder=5,
        label=f"t = {capacity} (BCH)",
    )
    ax_l.annotate(
        "over capacity →",
        xy=(capacity - 0.5 + bw + 0.05, ax_l.get_ylim()[1] * 0.85),
        color=STYLE["capacity_color"],
        fontsize=FONT_SIZES["annotation"],
    )
    ax_l.set_ylim(0, 2000)

    # X-axis labels: 0, 1, ..., group_size-1, "8+"
    x_labels = [str(k) for k in range(group_size)] + [f"{group_size}+"]
    ax_l.set_xticks(x)
    ax_l.set_xticklabels(x_labels, fontsize=FONT_SIZES["tick_label"], color=STYLE["fg_color"])

    # Axis styling
    for spine in ax_l.spines.values():
        spine.set_edgecolor(STYLE["fg_color"])
    for spine in ax_r.spines.values():
        spine.set_edgecolor(STYLE["r_axis_color"])

    ax_l.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"],
                     colors=STYLE["fg_color"])
    ax_r.tick_params(axis="y", labelsize=FONT_SIZES["tick_label"],
                     colors=STYLE["r_axis_color"])
    ax_l.yaxis.label.set_color(STYLE["fg_color"])
    ax_r.yaxis.label.set_color(STYLE["r_axis_color"])

    ax_l.set_xlabel("sensitive weights per codeword",
                    fontsize=FONT_SIZES["axis_label"], color=STYLE["fg_color"])
    ax_l.set_ylabel("number of codewords",
                    fontsize=FONT_SIZES["axis_label"], color=STYLE["fg_color"])
    ax_r.set_ylabel("% of codewords with > k",
                    fontsize=FONT_SIZES["axis_label"], color=STYLE["r_axis_color"])
    ax_r.set_ylim(0, 105)

    ax_l.set_title(title, fontsize=FONT_SIZES["title"], color=STYLE["fg_color"],
                   fontweight="bold", pad=12)

    ax_l.grid(True, linestyle=STYLE["grid_style"], alpha=STYLE["grid_alpha"],
              color=STYLE["grid_color"], zorder=1)

    ax_l.legend(
        fontsize=FONT_SIZES["legend"],
        facecolor="#2A2A2A",
        edgecolor="#555555",
        labelcolor=STYLE["fg_color"],
        loc="upper right",
    )

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight",
                facecolor=STYLE["bg_color"])
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot pattern study: codeword sensitive-weight distribution + accuracy table"
    )
    ap.add_argument("--results-dir",
                    default=os.path.join(os.path.dirname(__file__), "results"),
                    help="Directory containing contiguous/, random/, strided/ snapshots")
    ap.add_argument("--dataset",    default="IMAGENET",
                    choices=["CIFAR10", "CIFAR100", "IMAGENET"])
    ap.add_argument("--arch",       default="efficientnet_b0",
                    choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--qmode",      default="QAT", choices=["PTQ", "QAT"])
    ap.add_argument("--quant-bits", default=8, type=int)
    ap.add_argument("--t",          default=4, type=int, help="BCH t-value (capacity line)")
    ap.add_argument("--approach",   default="search3")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__), "plots"))
    args = ap.parse_args()

    log_path = os.path.join(LOG_DIR,
                            f"missing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    hists = load_histograms(args.results_dir)
    accs  = load_accuracies(args.results_dir)

    if all(v is None for v in hists.values()):
        print(f"[error] no histogram data found under {args.results_dir}")
        print("        Run run_pattern_study.sh first.")
        sys.exit(1)

    # ---- Graph ----
    bit_label = f"{args.quant_bits}bit"
    file_tag  = (f"{args.dataset.lower()}_{args.arch}_{args.qmode}"
                 f"_{bit_label}_t{args.t}_{args.approach}")
    combo_tag = (f"{args.dataset}/{args.arch}/{args.qmode}"
                 f"/{bit_label}/t={args.t}/{args.approach}")

    plot_histogram(
        hists,
        t_value=args.t,
        title=f"Sensitive weights per codeword — {combo_tag}",
        out_path=os.path.join(args.out_dir,
                              f"codeword_histogram_{file_tag}.png"),
    )

    # ---- Accuracy table ----
    table = {}
    for cond in CONDITIONS:
        acc = accs.get(cond)
        if acc is not None:
            table[cond] = {"top1": acc.get("top1"), "top5": acc.get("top5")}
        else:
            table[cond] = {"top1": None, "top5": None}
            log_warn(log_path, f"[warn] accuracy.json missing for condition={cond}")

    table_path = os.path.join(args.results_dir, "accuracy_table.json")
    with open(table_path, "w") as f:
        json.dump({
            "dataset":    args.dataset,
            "arch":       args.arch,
            "qmode":      args.qmode,
            "quant_bits": args.quant_bits,
            "t":          args.t,
            "approach":   args.approach,
            "conditions": table,
        }, f, indent=2)
    print(f"[saved] {table_path}")
    print("\n=== Accuracy Table ===")
    for cond in CONDITIONS:
        t1 = table[cond]["top1"]
        t5 = table[cond]["top5"]
        label = STYLE["labels"][cond]
        print(f"  {label:25s}  top-1={t1 if t1 is not None else 'N/A':>8}  "
              f"top-5={t5 if t5 is not None else 'N/A':>8}")


if __name__ == "__main__":
    main()
