"""
plot_sensitivity_graph.py — Sensitivity study Part 2: Top-1 accuracy vs SENS_NORM_MIN.

Reads staged accuracy snapshots produced by run_sensitivity_sweep.sh and produces
one plot per architecture, each showing two lines:
  1. Base Top-1 accuracy     — from results/sens_norm_*/{arch}/accuracy.json
  2. Sensitive Top-1 accuracy — from results/sens_norm_*/{arch}/sensitive_accuracy.json

x-axis: SENS_NORM_MIN (0.0 = raw taylor scores, 1.0 = uniform / no sensitivity)
y-axis: Top-1 accuracy (%)

Missing snapshot directories or JSON files are logged to logs/ and skipped;
a plot is still produced for whatever data is available per arch.

Usage:
    python3 plot_sensitivity_graph.py \
        --results-dir results \
        --dataset IMAGENET \
        --archs efficientnet_b0 mobilenet_v2 resnet18 \
        --qmode QAT --quant-bits 8 --t 2 --approach search3
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Visual configuration — edit these two dicts to restyle the graph
# ---------------------------------------------------------------------------

# Font sizes for every text element in the plot
FONT_SIZES = {
    "title":      15,   # main plot title
    "axis_label": 13,   # x-axis and y-axis labels
    "tick_label": 11,   # numbers on tick marks
    "legend":     10,   # legend entry text
}

# All other visual knobs
STYLE = {
    "figure_size": (10, 6),      # (width, height) in inches
    "dpi":         150,          # output resolution
    "line_width":  2.2,          # thickness of plotted lines
    "marker_size": 8,            # size of data-point markers
    "grid_alpha":  0.35,         # opacity of background grid
    "grid_style":  "--",         # linestyle for grid lines
    # Colors: [base accuracy, sensitive accuracy]
    "colors":  ["#2196F3", "#F44336"],
    # Markers in the same order
    "markers": ["o",       "s"      ],
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


# ---------------------------------------------------------------------------
# Logging helper — writes to stdout AND a persistent log file
# ---------------------------------------------------------------------------

def log_missing(log_path, message):
    print(message)
    os.makedirs(LOG_DIR, exist_ok=True)
    with open(log_path, "a") as f:
        f.write(message + "\n")


# ---------------------------------------------------------------------------
# Data discovery and loading
# ---------------------------------------------------------------------------

def discover_staged_values(results_dir, arch):
    """
    Discover SENS_NORM_MIN values for a specific arch.
    Looks in results/sens_norm_*/{arch}/ directories.
    Returns a sorted list of (float_value, directory_path) tuples.
    """
    pattern = os.path.join(results_dir, "sens_norm_*", arch)
    found = []
    for path in glob.glob(pattern):
        if not os.path.isdir(path):
            continue
        parent = os.path.basename(os.path.dirname(path))
        m = re.search(r"sens_norm_(\d+)_(\d+)$", parent)
        if m:
            # "0_1" → 0.1,  "1_0" → 1.0
            val = float(f"{m.group(1)}.{m.group(2)}")
            found.append((val, path))
    return sorted(found, key=lambda x: x[0])


def load_top1(json_path):
    """Read top1 from an accuracy JSON file. Returns float or None."""
    try:
        with open(json_path) as f:
            data = json.load(f)
        return data["top1"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_graph(base_series, sens_series, title, out_path):
    """
    Plot base and sensitive accuracy vs SENS_NORM_MIN on one figure.

    base_series : list[(float, float)]  — (sns_val, top1)
    sens_series : list[(float, float)]  — (sns_val, top1)
    """
    if not base_series and not sens_series:
        print(f"[error] no data to plot — skipping {out_path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=STYLE["figure_size"])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    def draw(series, label, color, marker):
        if not series:
            return
        xs, ys = zip(*series)
        ax.plot(
            xs, ys,
            label=label,
            color=color,
            marker=marker,
            linewidth=STYLE["line_width"],
            markersize=STYLE["marker_size"],
            markeredgewidth=1.2,
            markeredgecolor="white",
        )

    draw(base_series, "Base accuracy (accuracy.json)",
         STYLE["colors"][0], STYLE["markers"][0])
    draw(sens_series, "Sensitive accuracy (sensitive_accuracy.json)",
         STYLE["colors"][1], STYLE["markers"][1])

    # x-ticks at every sampled SENS_NORM_MIN value
    all_x = sorted({x for x, _ in (base_series + sens_series)})
    ax.set_xticks(all_x)
    ax.set_xlim(-0.05, 1.05)

    ax.set_xlabel("SENS_NORM_MIN", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=FONT_SIZES["axis_label"])
    ax.set_title(title, fontsize=FONT_SIZES["title"], fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    ax.grid(True, linestyle=STYLE["grid_style"], alpha=STYLE["grid_alpha"], color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=FONT_SIZES["legend"], framealpha=0.85, edgecolor="#CCCCCC")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot Top-1 accuracy vs SENS_NORM_MIN for base and sensitive accuracy"
    )
    ap.add_argument("--results-dir",
                    default=os.path.join(os.path.dirname(__file__), "results"),
                    help="Directory containing sens_norm_*/ staged snapshot subdirs")
    ap.add_argument("--dataset",     default="IMAGENET",
                    choices=["CIFAR10", "CIFAR100", "IMAGENET"])
    ap.add_argument("--archs",       nargs="+",
                    default=["efficientnet_b0"],
                    choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--qmode",       default="QAT", choices=["PTQ", "QAT"])
    ap.add_argument("--quant-bits",  default=8, type=int)
    ap.add_argument("--t",           default=2, type=int, help="BCH t-value used in the sweep")
    ap.add_argument("--approach",    default="search3")
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__), "plots"))
    args = ap.parse_args()

    log_path = os.path.join(LOG_DIR,
                            f"missing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    any_data = False
    for arch in args.archs:
        staged = discover_staged_values(args.results_dir, arch)
        if not staged:
            print(f"[warn] arch={arch}: no staged data found under {args.results_dir} — skipping")
            continue

        print(f"\n[info] arch={arch}: discovered {len(staged)} SENS_NORM_MIN values: "
              f"{[v for v, _ in staged]}")

        base_series = []
        sens_series = []

        for sns_val, snap_dir in staged:
            base_path = os.path.join(snap_dir, "accuracy.json")
            sens_path = os.path.join(snap_dir, "sensitive_accuracy.json")

            base_top1 = load_top1(base_path)
            if base_top1 is not None:
                base_series.append((sns_val, base_top1))
            else:
                log_missing(log_path,
                    f"[warn] arch={arch} SENS_NORM_MIN={sns_val}: accuracy.json missing "
                    f"at {base_path}")

            sens_top1 = load_top1(sens_path)
            if sens_top1 is not None:
                sens_series.append((sns_val, sens_top1))
            else:
                log_missing(log_path,
                    f"[warn] arch={arch} SENS_NORM_MIN={sns_val}: sensitive_accuracy.json missing "
                    f"at {sens_path}")

        print(f"[info] arch={arch}: base accuracy points:      {len(base_series)}")
        print(f"[info] arch={arch}: sensitive accuracy points: {len(sens_series)}")

        bit_label = f"{args.quant_bits}bit"
        combo_tag = (f"{args.dataset}/{arch}/{args.qmode}"
                     f"/{bit_label}/t={args.t}/{args.approach}")
        file_tag  = (f"{args.dataset.lower()}_{arch}_{args.qmode}"
                     f"_{bit_label}_t{args.t}_{args.approach}")

        plot_graph(
            base_series,
            sens_series,
            title=f"Top-1 accuracy vs SENS_NORM_MIN — {combo_tag}",
            out_path=os.path.join(args.out_dir,
                                  f"sensitivity_vs_sens_norm_min_{file_tag}.png"),
        )
        any_data = True

    if not any_data:
        print(f"[error] no staged data found for any arch under {args.results_dir}")
        print("        Run run_sensitivity_sweep.sh first to populate staged results.")
        sys.exit(1)


if __name__ == "__main__":
    main()
