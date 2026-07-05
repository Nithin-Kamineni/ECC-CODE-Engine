"""
plot_weight_distortion.py — Ablation: weight distortion vs BER, accuracy vs BER.

For a given (dataset, arch, qmode, quant-bits, codeword) combo, overlays all
requested approaches on two shared line graphs against BER (BCH t-value sweep):
  1. Weight distortion vs BER  — count-weighted mean of the "distorsion" field
     stored in embeddedECC_Chunks/.../chunks_p*.jsonl
  2. Top-1 accuracy vs BER     — from accuracy_results/.../accuracy.json

If an approach has no artifact data on disk, a warning is written to a
timestamped log file in ./logs/ and that approach is skipped; the plot is still
generated for whichever approaches do have data.

Usage:
    python3 plot_weight_distortion.py \
        --dataset IMAGENET --arch mobilenet_v2 --qmode QAT \
        --quant-bits 8 --approaches search3 greedy parfix replace --codeword 63
"""

import argparse
import glob
import json
import os
import re
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Visual configuration — edit these two dicts to restyle the graphs
# ---------------------------------------------------------------------------

# Font sizes for every text element in the plots
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
    # Colors assigned to approaches in the order they appear on --approaches
    "colors":  ["#2196F3", "#F44336", "#4CAF50", "#FF9800"],
    # Markers assigned in the same order
    "markers": ["o",       "s",       "^",       "D"      ],
}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "0-Data", "artifacts")
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
# Data helpers
# ---------------------------------------------------------------------------

def discover_t_values(chunks_dir, ds_lower, arch, qmode, bit_label, codeword, approach):
    pattern = os.path.join(chunks_dir, ds_lower, arch, qmode, bit_label,
                           f"M{codeword}_t*", approach)
    t_values = []
    for path in glob.glob(pattern):
        m = re.search(rf"M{codeword}_t(\d+)$", os.path.dirname(path))
        if m and os.path.isdir(path):
            t_values.append(int(m.group(1)))
    return sorted(set(t_values))


def compute_distortion_for_t(chunks_dir, ds_lower, arch, qmode, bit_label,
                              codeword, approach, t):
    t_dir = os.path.join(chunks_dir, ds_lower, arch, qmode, bit_label,
                         f"M{codeword}_t{t}", approach)
    jsonl_files = glob.glob(os.path.join(t_dir, "*", "chunks_p*.jsonl"))
    if not jsonl_files:
        print(f"[warn] no weight-distortion data for t={t}: no chunks_p*.jsonl under {t_dir}")
        return None

    weighted_sum = 0.0
    total_count = 0
    for fpath in jsonl_files:
        with open(fpath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                count = rec.get("count")
                distorsion = rec.get("distorsion")
                if count is None or distorsion is None:
                    continue
                weighted_sum += distorsion * count
                total_count += count

    if total_count == 0:
        print(f"[warn] no weight-distortion data for t={t}: all records unparseable under {t_dir}")
        return None
    return weighted_sum / total_count


def load_accuracy_for_t(accuracy_dir, ds_lower, arch, qmode, bit_label,
                         codeword, approach, t):
    acc_path = os.path.join(accuracy_dir, ds_lower, arch, qmode, bit_label,
                            f"M{codeword}_t{t}", approach, "accuracy.json")
    if not os.path.exists(acc_path):
        print(f"[warn] no accuracy data for t={t}: {acc_path} not found")
        return None
    try:
        with open(acc_path) as f:
            data = json.load(f)
        return data["top1"]
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[warn] no accuracy data for t={t}: failed to read {acc_path} ({e})")
        return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_line(series, ylabel, title, out_path):
    """
    Overlay multiple approaches on one figure.

    series : dict[str, (list[int], list[float])]
        Maps approach name → (t_values, y_values).
    """
    if not series:
        print(f"[error] no data points for '{title}' — skipping {out_path}")
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=STYLE["figure_size"])
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#FAFAFA")

    for idx, (approach, (t_vals, y_vals)) in enumerate(series.items()):
        color  = STYLE["colors"][idx % len(STYLE["colors"])]
        marker = STYLE["markers"][idx % len(STYLE["markers"])]
        ax.plot(
            t_vals, y_vals,
            label=approach,
            color=color,
            marker=marker,
            linewidth=STYLE["line_width"],
            markersize=STYLE["marker_size"],
            markeredgewidth=1.2,
            markeredgecolor="white",
        )

    # Collect all t-values across approaches for consistent x-ticks
    all_t = sorted({t for t_vals, _ in series.values() for t in t_vals})
    ax.set_xticks(all_t)

    ax.set_xlabel("BER (t)", fontsize=FONT_SIZES["axis_label"])
    ax.set_ylabel(ylabel,    fontsize=FONT_SIZES["axis_label"])
    ax.set_title(title,      fontsize=FONT_SIZES["title"], fontweight="bold")
    ax.tick_params(axis="both", labelsize=FONT_SIZES["tick_label"])

    ax.grid(True, linestyle=STYLE["grid_style"], alpha=STYLE["grid_alpha"], color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(fontsize=FONT_SIZES["legend"], framealpha=0.85, edgecolor="#CCCCCC")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=STYLE["dpi"], bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Plot weight distortion vs BER and accuracy vs BER for multiple approaches"
    )
    ap.add_argument("--dataset",      required=True, choices=["CIFAR10", "CIFAR100", "IMAGENET"])
    ap.add_argument("--arch",         required=True,
                    choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--qmode",        required=True, choices=["PTQ", "QAT"])
    ap.add_argument("--quant-bits",   required=True, type=int)
    ap.add_argument("--approaches",   nargs="+", default=["search3"],
                    help="One or more approach names to overlay on the same plot")
    ap.add_argument("--codeword",     default=63, type=int)
    ap.add_argument("--chunks-dir",
                    default=os.path.join(DEFAULT_ARTIFACTS_DIR, "embeddedECC_Chunks"))
    ap.add_argument("--accuracy-dir",
                    default=os.path.join(DEFAULT_ARTIFACTS_DIR, "accuracy_results"))
    ap.add_argument("--out-dir",
                    default=os.path.join(os.path.dirname(__file__), "plots"))
    args = ap.parse_args()

    ds_lower  = args.dataset.lower()
    bit_label = f"{args.quant_bits}-bit"
    base_tag  = f"{args.dataset}/{args.arch}/{args.qmode}/{bit_label}"

    print(f"[plot_weight_distortion] {base_tag}  approaches={args.approaches}")

    # Timestamped log file for approaches that have no data on disk
    log_path = os.path.join(LOG_DIR,
                            f"missing_methods_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

    distortion_series = {}   # approach -> (t_list, y_list)
    accuracy_series   = {}

    for approach in args.approaches:
        t_vals = discover_t_values(
            args.chunks_dir, ds_lower, args.arch, args.qmode,
            bit_label, args.codeword, approach,
        )
        if not t_vals:
            log_missing(log_path,
                f"[warn] no t-value directories found for approach='{approach}' "
                f"under {args.chunks_dir}/{ds_lower}/.../{bit_label}/M{args.codeword}_t*/{approach} "
                f"— skipping this method")
            continue

        print(f"[info] approach='{approach}'  discovered t-values: {t_vals}")

        # Collect weight distortion points
        dt, dy = [], []
        for t in t_vals:
            val = compute_distortion_for_t(
                args.chunks_dir, ds_lower, args.arch, args.qmode,
                bit_label, args.codeword, approach, t,
            )
            if val is not None:
                dt.append(t)
                dy.append(val)
        if dt:
            distortion_series[approach] = (dt, dy)
        else:
            log_missing(log_path,
                f"[warn] approach='{approach}' — t-value dirs exist but no "
                f"parseable distortion data found")

        # Collect accuracy points
        at, ay = [], []
        for t in t_vals:
            val = load_accuracy_for_t(
                args.accuracy_dir, ds_lower, args.arch, args.qmode,
                bit_label, args.codeword, approach, t,
            )
            if val is not None:
                at.append(t)
                ay.append(val)
        if at:
            accuracy_series[approach] = (at, ay)
        else:
            log_missing(log_path,
                f"[warn] approach='{approach}' — t-value dirs exist but no "
                f"parseable accuracy data found")

    if not distortion_series and not accuracy_series:
        print(
            f"[error] no data found for any approach in {args.approaches}. "
            f"Check log: {log_path}"
        )
        sys.exit(1)

    print(f"[info] plotting distortion for: {list(distortion_series.keys())}")
    print(f"[info] plotting accuracy   for: {list(accuracy_series.keys())}")

    # Output files — approach name omitted since multiple methods share the plot
    file_tag = f"{ds_lower}_{args.arch}_{args.qmode}_{args.quant_bits}bit"

    plot_line(
        distortion_series,
        ylabel="Mean weight distortion (|Δ value|)",
        title=f"Weight distortion vs BER — {base_tag}",
        out_path=os.path.join(args.out_dir, f"weight_distortion_vs_ber_{file_tag}.png"),
    )
    plot_line(
        accuracy_series,
        ylabel="Top-1 accuracy (%)",
        title=f"Accuracy vs BER — {base_tag}",
        out_path=os.path.join(args.out_dir, f"accuracy_top1_vs_ber_{file_tag}.png"),
    )


if __name__ == "__main__":
    main()
