"""
merge_ecc.py — Merge, gap-fill, and checkpoint reconstruction for ECC-CODE-Engine.

For each (dataset, arch, quant_bits, t_value, approach) combination:
  1. Scans per-worker JSONL chunk files in embeddedECC_Chunks/.
  2. Detects any missing index ranges (failed workers).
  3. Re-encodes missing ranges using the same ECC approach and saves them
     to the same embeddedECC_Chunks/ directory.
  4. Stitches all chunks in order; for each layer applies inv_perm to
     restore the original weight ordering.
  5. Saves a reconstructed int8 checkpoint (.pth) with scales/metadata
     copied from the original quantized model.

Output:
    0-Data/artifacts/embeddedECC/{ds}/{arch}/PTQ/{bit}/
        M{codeword}_t{tval}/{approach}/ECC_Embedded_model.pth

Usage:
    python3 merge_ecc.py \
        --dataset CIFAR10 --arch resnet18 --quant-bits 8 \
        --t-value 2 --approach parfix --codeword 63 \
        --patterns-dir  /path/to/0-Data/artifacts/patterns \
        --chunks-dir    /path/to/0-Data/artifacts/embeddedECC_Chunks \
        --ecc-dir       /path/to/0-Data/artifacts/embeddedECC \
        --models-dir    /path/to/0-Data/artifacts/models \
        --ecc-source    /path/to/4-EmbeddingECC
"""

import os, sys, json, argparse, pathlib
import numpy as np
import torch
from pathlib import Path
from math import prod


# =============================================================================
# BCH lookup (mirrors ecc_embed.py)
# =============================================================================
NANDT_TO_K = {
    63:  {1: 57, 2: 51, 3: 45, 4: 39, 5: 36, 6: 30, 7: 24, 8: 18},
    127: {1: 120, 2: 113, 3: 106, 4: 99, 5: 92, 6: 85, 7: 78, 8: 71,
          9: 71, 10: 64, 11: 57, 12: 50, 13: 50},
    255: {4: 223, 8: 191, 9: 187, 10: 179, 11: 171, 12: 163, 13: 155,
          14: 147, 15: 139, 16: 131, 18: 131, 25: 91},
}


def _sanitize(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


# =============================================================================
# Load ECC processing function from 4-EmbeddingECC
# =============================================================================
def _load_ecc_tools(ecc_source: str):
    """Import process_payload and related ECC tools from 4-EmbeddingECC."""
    sys.path.insert(0, ecc_source)
    from ecc_embed import process_payload, NANDT_TO_K as _LUT
    from implementations.Search3EncodeAndDecode import build_bch_parity_matrix
    return process_payload, build_bch_parity_matrix


# =============================================================================
# Read all JSONL chunks for one layer
# =============================================================================
def _read_chunks(layer_dir: str):
    """Return sorted list of (start, end, values) from all chunks_p*.jsonl."""
    records = []
    layer_path = Path(layer_dir)
    if not layer_path.exists():
        return records
    for fp in layer_path.glob("chunks_p*.jsonl"):
        with fp.open("r") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") != "ok":
                    continue
                records.append((int(rec["start"]), int(rec["end"]), rec["values"]))
    records.sort(key=lambda t: t[0])
    return records


# =============================================================================
# Find gaps in coverage
# =============================================================================
def _find_gaps(records, N: int):
    """Return list of (start, end) ranges not covered by records."""
    gaps = []
    cur = 0
    for s, e, _ in records:
        if s > cur:
            gaps.append((cur, s - 1))
        cur = max(cur, e + 1)
    if cur < N:
        gaps.append((cur, N - 1))
    return gaps


# =============================================================================
# Reprocess a single gap range
# =============================================================================
def _reprocess_gap(gap_start, gap_end, arr_u8, process_payload, chunk_size,
                   message_parity_size, message_size, approach, p_matrix,
                   layer_dir, gap_idx):
    """Encode a missing index range and append it to a new JSONL file."""
    values = arr_u8[gap_start:gap_end + 1]
    mutated, distortion = process_payload(
        values, approach, chunk_size, message_parity_size, message_size, p_matrix
    )
    rec = {
        "p":         -1,          # -1 = reprocessed by merge step
        "start":     int(gap_start),
        "end":       int(gap_end),
        "count":     int(gap_end - gap_start + 1),
        "values":    mutated,
        "distorsion": distortion,
        "status":    "ok",
    }
    out_path = Path(layer_dir) / f"chunks_gap{gap_idx}.jsonl"
    with open(out_path, "w") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"    [gap {gap_idx}] reprocessed [{gap_start}, {gap_end}] → {out_path.name}")
    return (gap_start, gap_end, mutated)


# =============================================================================
# Stitch records into a flat int8 array
# =============================================================================
def _stitch(records, N: int):
    result = [None] * N
    for s, e, vals in records:
        for i, v in enumerate(vals):
            result[s + i] = v
    if any(x is None for x in result):
        missing = [i for i, x in enumerate(result) if x is None]
        raise RuntimeError(f"Still {len(missing)} missing indices after gap fill")
    return np.array(result, dtype=np.int8)


# =============================================================================
# Process one full combination
# =============================================================================
def process_combination(args, t_value, process_payload_fn, build_bch_fn,
                         chunk_size, message_parity_size, message_size,
                         approach, p_matrix, bit_label):
    ds_lower = args.dataset.lower()
    m_tag    = f"M{args.codeword}_t{t_value}"

    # Pattern manifest
    manifest_path = Path(args.patterns_dir) / ds_lower / args.arch / "PTQ" / bit_label / "pattern_manifest.json"
    if not manifest_path.exists():
        print(f"  [skip] No manifest: {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Original quantized checkpoint (for metadata/scales)
    model_ckpt_path = Path(args.models_dir) / ds_lower / args.arch / "PTQ" / f"model_int{args.quant_bits}_ptq.pth"
    if not model_ckpt_path.exists():
        print(f"  [skip] No quantized checkpoint: {model_ckpt_path}")
        return

    orig_ckpt = torch.load(model_ckpt_path, map_location="cpu")

    # Output .pth path
    out_dir = Path(args.ecc_dir) / ds_lower / args.arch / "PTQ" / bit_label / m_tag / approach
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pth  = out_dir / "ECC_Embedded_model.pth"

    print(f"\n[merge] {args.dataset}/{args.arch}/{bit_label}/{m_tag}/{approach}")
    print(f"  manifest: {manifest_path} ({len(manifest)} layers)")

    new_state_dict = {}
    for layer_name, entry in manifest.items():
        N         = int(entry["N"])
        shape     = entry.get("shape")
        layer_safe = _sanitize(layer_name)

        # Directory where 4-EmbeddingECC wrote chunks for this layer
        layer_dir = Path(args.chunks_dir) / ds_lower / args.arch / "PTQ" / bit_label / m_tag / approach / layer_safe

        print(f"  [layer] {layer_name}  N={N:,}", end="", flush=True)

        # Read existing chunks
        records = _read_chunks(str(layer_dir))
        gaps    = _find_gaps(records, N)

        if gaps:
            print(f"  → {len(gaps)} gap(s), reprocessing ...", flush=True)
            # Load the permuted weights for reprocessing
            weights_file = entry.get("weights_perm_file")
            if not weights_file or not os.path.exists(weights_file):
                print(f"    [error] weights_perm_file missing — cannot fill gaps, skipping layer")
                continue
            arr = np.load(weights_file)
            if arr.dtype != np.int8:
                raise TypeError(
                    f"weights_perm_file has dtype={arr.dtype}, expected int8. "
                    f"Re-run prepare_patterns.py to regenerate the file."
                )
            arr_u8 = (arr.astype(np.int16) + 128).astype(np.uint8)

            layer_dir.mkdir(parents=True, exist_ok=True)
            for gi, (gs, ge) in enumerate(gaps):
                new_rec = _reprocess_gap(
                    gs, ge, arr_u8, process_payload_fn, chunk_size,
                    message_parity_size, message_size, approach, p_matrix,
                    str(layer_dir), gi,
                )
                records.append(new_rec)
            records.sort(key=lambda t: t[0])
        else:
            print("  → complete", flush=True)

        # Stitch into flat int8 array
        try:
            flat_encoded = _stitch(records, N)
        except RuntimeError as e:
            print(f"    [error] {e} — skipping layer")
            continue

        # Apply inv_perm to restore original weight ordering
        inv_perm_file = entry.get("inv_perm_file")
        if inv_perm_file and os.path.exists(inv_perm_file):
            inv_perm = np.load(inv_perm_file)
            # Validate: must be a complete permutation of [0, N).
            # Garbage values (large out-of-range ints) indicate the file was
            # generated from a non-bijective stride — re-run 3-PatternFinder.
            if (int(inv_perm.max()) >= N or int(inv_perm.min()) < 0
                    or len(np.unique(inv_perm)) != N):
                raise RuntimeError(
                    f"    [error] {layer_name}: inv_perm_file is invalid "
                    f"(max={inv_perm.max()}, min={inv_perm.min()}, "
                    f"unique={len(np.unique(inv_perm))}, N={N}). "
                    f"Re-run 3-PatternFinder (with fixed find_pattern.py) "
                    f"and 4-EmbeddingECC to regenerate correct files."
                )
            flat_original = flat_encoded[inv_perm]
        else:
            print(f"    [warn] inv_perm_file missing — using permuted ordering")
            flat_original  = flat_encoded

        # Reshape and convert to tensor
        if shape:
            arr_orig = flat_original.reshape(shape, order="C")
        else:
            arr_orig = flat_original
        new_state_dict[layer_name] = torch.from_numpy(np.array(arr_orig, dtype=np.int8, copy=False))

    # Build new checkpoint: copy metadata from original, replace weights
    out_ckpt = {}
    # Prefer qstate_dict key (PTQ checkpoints), fall back to state_dict
    if "qstate_dict" in orig_ckpt:
        merged_qsd = dict(orig_ckpt["qstate_dict"])
        merged_qsd.update(new_state_dict)          # replace ECC-processed layers
        out_ckpt["qstate_dict"] = merged_qsd
    else:
        merged_sd = dict(orig_ckpt.get("state_dict", {}))
        merged_sd.update(new_state_dict)
        out_ckpt["state_dict"] = merged_sd

    if "meta" in orig_ckpt:
        out_ckpt["meta"] = orig_ckpt["meta"]

    torch.save(out_ckpt, out_pth)
    print(f"  [saved] {out_pth}")


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Merge ECC chunks and reconstruct .pth checkpoint")
    ap.add_argument("--dataset",      required=True, choices=["CIFAR10", "CIFAR100", "IMAGENET"])
    ap.add_argument("--arch",         required=True,
                    choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--quant-bits",   required=True, type=int, choices=[4, 8, 16])
    ap.add_argument("--t-value",      required=True, type=int)
    ap.add_argument("--approach",     default="parfix",
                    choices=["parfit", "replace", "no", "parfix", "search3", "greedy"])
    ap.add_argument("--codeword",     default=63, type=int, choices=[63, 127, 255])
    ap.add_argument("--patterns-dir", required=True)
    ap.add_argument("--chunks-dir",   required=True,
                    help="Root of embeddedECC_Chunks/ (4-EmbeddingECC output)")
    ap.add_argument("--ecc-dir",      required=True,
                    help="Root of embeddedECC/ (final output)")
    ap.add_argument("--models-dir",   required=True,
                    help="Root of 0-Data/artifacts/models/")
    ap.add_argument("--ecc-source",   required=True,
                    help="Path to 4-EmbeddingECC directory (for ECC tool imports)")
    args = ap.parse_args()

    t_value             = args.t_value
    message_parity_size = args.codeword
    message_size        = NANDT_TO_K[message_parity_size][t_value]
    if args.approach != 'parfix':
        chunk_size = message_parity_size
    else:
        chunk_size = message_size

    # Load ECC tools from 4-EmbeddingECC
    process_payload_fn, build_bch_fn = _load_ecc_tools(args.ecc_source)

    p_matrix = None
    if args.approach in ('search3', 'greedy'):
        p_matrix = build_bch_fn(n=message_parity_size, k=message_size)

    bit_label = f"{args.quant_bits}-bit"

    process_combination(
        args, t_value, process_payload_fn, build_bch_fn,
        chunk_size, message_parity_size, message_size,
        args.approach, p_matrix, bit_label,
    )

    print("\n[merge_ecc] Done.")


if __name__ == "__main__":
    main()
