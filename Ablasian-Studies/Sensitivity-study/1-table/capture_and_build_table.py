"""
capture_and_build_table.py — Ablation: sensitivity vs no-sensitivity ECC embedding.

0-Data can only hold the results of ONE EMBED_SENSITIVITY setting at a time
(stage 4 re-embeds and overwrites the same paths either way). So this script
is run TWICE, once per mode, with 0-Data switched in between:

    1. EMBED_SENSITIVITY=false, run stages 4-7, then:
       python3 capture_and_build_table.py --mode no-sensitivity
    2. EMBED_SENSITIVITY=true,  run stages 4-7 again, then:
       python3 capture_and_build_table.py --mode sensitivity

Each call snapshots the CURRENT 0-Data accuracy.json values for the given
mode into results/{mode}_accuracies.json. Once both snapshots exist, the
second call also builds the final comparison table (results/comparison_table.md
and .json) from both cached snapshots — no need to have both datasets in
0-Data simultaneously.

Missing accuracy.json for any (arch, t) is logged and recorded as null /
"N/A" rather than failing the whole run.
"""

import argparse
import json
import os
from datetime import datetime, timezone

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_ACCURACY_DIR = os.path.join(PROJECT_ROOT, "0-Data", "artifacts", "accuracy_results")

MODES = ("no-sensitivity", "sensitivity")


def other_mode(mode):
    return MODES[1] if mode == MODES[0] else MODES[0]


def load_accuracy(accuracy_dir, ds_lower, arch, qmode, bit_label, codeword, approach, t):
    path = os.path.join(accuracy_dir, ds_lower, arch, qmode, bit_label,
                         f"M{codeword}_t{t}", approach, "accuracy.json")
    if not os.path.exists(path):
        print(f"[skip] no accuracy data for {arch} t={t}: {path}")
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        return {"top1": data.get("top1"), "top5": data.get("top5")}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"[skip] failed to read accuracy data for {arch} t={t}: {path} ({e})")
        return None


def capture_snapshot(args):
    ds_lower = args.dataset.lower()
    bit_label = f"{args.quant_bits}-bit"

    snapshot = {
        "mode": args.mode,
        "dataset": args.dataset,
        "qmode": args.qmode,
        "quant_bits": args.quant_bits,
        "approach": args.approach,
        "codeword": args.codeword,
        "t_values": args.t_values,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "accuracies": {},
    }

    for arch in args.archs:
        snapshot["accuracies"][arch] = {}
        for t in args.t_values:
            val = load_accuracy(args.accuracy_dir, ds_lower, arch, args.qmode,
                                 bit_label, args.codeword, args.approach, t)
            snapshot["accuracies"][arch][str(t)] = val

    return snapshot


def build_comparison_table(out_dir, t_values, no_sens_snapshot, sens_snapshot):
    archs = sorted(set(no_sens_snapshot["accuracies"]) | set(sens_snapshot["accuracies"]))

    def cell(snapshot, arch, t):
        if snapshot is None:
            return None
        rec = snapshot["accuracies"].get(arch, {}).get(str(t))
        return rec["top1"] if rec else None

    rows = {}
    for arch in archs:
        rows[arch] = {}
        for t in t_values:
            rows[arch][str(t)] = {
                "no_sensitivity": cell(no_sens_snapshot, arch, t),
                "sensitivity": cell(sens_snapshot, arch, t),
            }

    table_json = {"t_values": t_values, "archs": archs, "rows": rows}

    header_cells = ["Model"]
    for t in t_values:
        header_cells += [f"t={t} no-sens (top1%)", f"t={t} sens (top1%)"]
    lines = [
        "| " + " | ".join(header_cells) + " |",
        "| " + " | ".join(["---"] * len(header_cells)) + " |",
    ]
    for arch in archs:
        row_cells = [arch]
        for t in t_values:
            c = rows[arch][str(t)]
            no_sens = f"{c['no_sensitivity']:.4f}" if c["no_sensitivity"] is not None else "N/A"
            sens = f"{c['sensitivity']:.4f}" if c["sensitivity"] is not None else "N/A"
            row_cells += [no_sens, sens]
        lines.append("| " + " | ".join(row_cells) + " |")

    md_text = "\n".join(lines) + "\n"

    md_path = os.path.join(out_dir, "comparison_table.md")
    json_path = os.path.join(out_dir, "comparison_table.json")
    with open(md_path, "w") as f:
        f.write(md_text)
    with open(json_path, "w") as f:
        json.dump(table_json, f, indent=2)

    print(f"[saved] {md_path}")
    print(f"[saved] {json_path}")


def main():
    ap = argparse.ArgumentParser(
        description="Capture sensitivity / no-sensitivity ECC accuracy and build a comparison table")
    ap.add_argument("--mode", required=True, choices=MODES,
                     help="Which EMBED_SENSITIVITY setting 0-Data currently reflects")
    ap.add_argument("--t-values", type=int, nargs="+", default=[1, 6])
    ap.add_argument("--dataset", default="IMAGENET", choices=["CIFAR10", "CIFAR100", "IMAGENET"])
    ap.add_argument("--archs", nargs="+",
                     default=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--qmode", default="QAT", choices=["PTQ", "QAT"])
    ap.add_argument("--quant-bits", type=int, default=8)
    ap.add_argument("--approach", default="search3")
    ap.add_argument("--codeword", type=int, default=63)
    ap.add_argument("--accuracy-dir", default=DEFAULT_ACCURACY_DIR)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(__file__), "results"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[capture] mode={args.mode}  dataset={args.dataset}  qmode={args.qmode}  "
          f"quant_bits={args.quant_bits}  approach={args.approach}  t_values={args.t_values}")

    snapshot = capture_snapshot(args)
    snapshot_path = os.path.join(args.out_dir, f"{args.mode.replace('-', '_')}_accuracies.json")
    with open(snapshot_path, "w") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[saved] {snapshot_path}")

    other = other_mode(args.mode)
    other_path = os.path.join(args.out_dir, f"{other.replace('-', '_')}_accuracies.json")
    if not os.path.exists(other_path):
        print(f"[info] only '{args.mode}' captured so far — switch 0-Data to "
              f"EMBED_SENSITIVITY={'true' if args.mode == 'no-sensitivity' else 'false'}, "
              f"re-run stages 4-7, then run again with --mode {other} to build the comparison table.")
        return

    with open(other_path) as f:
        other_snapshot = json.load(f)

    no_sens_snapshot = snapshot if args.mode == "no-sensitivity" else other_snapshot
    sens_snapshot = snapshot if args.mode == "sensitivity" else other_snapshot

    build_comparison_table(args.out_dir, args.t_values, no_sens_snapshot, sens_snapshot)


if __name__ == "__main__":
    main()
