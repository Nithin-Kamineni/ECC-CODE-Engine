"""
ecc_embed.py — Per-layer ECC embedding pipeline for ECC-CODE-Engine.

Reads quantized weights (int8) in pattern-permuted order from the patterns
directory, applies parallel ECC encoding, and writes per-worker JSONL chunk
files to embeddedECC_Chunks/.

Usage:
    python3 ecc_embed.py --dataset CIFAR10 --arch resnet18 \
        --quant-bits 8 --t-value 2 --approach parfix \
        --codeword 63 --workers 24 \
        --patterns-dir /path/to/patterns \
        --chunks-dir   /path/to/embeddedECC_Chunks

Adapted from:
    RECC/code/dynamic_pipeline/dynamic_parallel_payload_process.py
All utility/implementation files are copied locally under utils/ and
implementations/ — no runtime imports from the RECC project directory.
"""

import os, sys, json, time, argparse, pathlib
import multiprocessing as mp
import numpy as np
from pathlib import Path

# ---- Local imports (copied from RECC project) ----
sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.convert_to_binary import convert_to_binary
from utils.messageSliceBasedOnChunkSize import messageSliceBasedOnChunkSize
from utils.reconstruct_numbers_from_chunks import reconstruct_numbers_from_chunks
from implementations.ParityOverwriteByTopWeightsEncode import ParityOverwriteByTopWeightsEncode
from implementations.OptimizedParityFittingWeightsEncodeAndDecode import OptimizedParityFittingWeightsEncodeAndDecode
from implementations.ParityFxingWeightsEncodeAndDecode import MutateWeightsEncodeAndDecode
from implementations.Search3EncodeAndDecode import Search3EncodeAndDecode, build_bch_parity_matrix
from implementations.GreedyEncodeAndDecode import GreedyEncodeAndDecode

# ---- BCH lookup: (codeword_n, t) -> message_k ----
NANDT_TO_K = {
    63:  {1: 57, 2: 51, 3: 45, 4: 39, 5: 36, 6: 30, 7: 24, 8: 18},
    127: {1: 120, 2: 113, 3: 106, 4: 99, 5: 92, 6: 85, 7: 78, 8: 71,
          9: 71, 10: 64, 11: 57, 12: 50, 13: 50},
    255: {4: 223, 8: 191, 9: 187, 10: 179, 11: 171, 12: 163, 13: 155,
          14: 147, 15: 139, 16: 131, 18: 131, 25: 91},
}

_ALLOWED_APPROACHES = ('parfit', 'replace', 'no', 'parfix', 'search3', 'greedy')


# =============================================================================
# CPU affinity
# =============================================================================
def _assign_affinity(p: int, cpus_per_worker: int = 2):
    avail = sorted(os.sched_getaffinity(0))
    start = p * cpus_per_worker
    end   = start + cpus_per_worker
    if end > len(avail):
        group = [avail[i % len(avail)] for i in range(start, end)]
    else:
        group = avail[start:end]
    os.sched_setaffinity(0, set(group))
    return group


# =============================================================================
# Atomic chunk allocator
# =============================================================================
def claim_next(next_idx: mp.Value, lock: mp.Lock, N: int, chunk_size: int):
    with lock:
        start = next_idx.value
        if start >= N:
            return None
        next_idx.value = start + chunk_size
    end = min(start + chunk_size, N) - 1
    return (start, end)


# =============================================================================
# ECC encode kernel (all approach variants)
# =============================================================================
def process_payload(values, approach, chunk_size, message_parity_size,
                    message_size, p_matrix=None):
    """
    Encode a slice of int8 values with the chosen ECC approach.

    Args:
        values: numpy int8 array, already shifted to uint8 (0-255)
        approach: one of _ALLOWED_APPROACHES
        chunk_size / message_parity_size / message_size: BCH parameters
        p_matrix: pre-built parity matrix (search3/greedy only)

    Returns:
        (mutated_int8_list, distortion_float)
    """
    vals = values.tolist()                            # uint8 list (0-255)
    message_bits = convert_to_binary(vals, bit_size=8)
    chunks = messageSliceBasedOnChunkSize(message_bits, chunk_size=chunk_size)

    mutated_chunks = []
    for chunk in chunks:
        if approach == 'replace':
            out = ParityOverwriteByTopWeightsEncode(
                chunk,
                message_parity_size=message_parity_size,
                message_size=message_size,
            )
        elif approach == 'parfit':
            out = OptimizedParityFittingWeightsEncodeAndDecode(
                chunk,
                message_parity_size=message_parity_size,
                message_size=message_size,
                solver='cpsat',
            )
        elif approach == 'parfix':
            out = MutateWeightsEncodeAndDecode(
                chunk,
                message_parity_size=message_parity_size,
                message_size=message_size,
            )
        elif approach == 'search3':
            out = Search3EncodeAndDecode(
                chunk,
                P_matrix=p_matrix,
                message_parity_size=message_parity_size,
                message_size=message_size,
                search_metric="L2",
            )
        elif approach == 'greedy':
            out = GreedyEncodeAndDecode(
                chunk,
                P_matrix=p_matrix,
                message_parity_size=message_parity_size,
                message_size=message_size,
                search_metric="L2",
                move_unit_range=4,
            )
        else:
            out = chunk          # 'no' — pass-through unchanged
        mutated_chunks.append(out)

    reconstructed = reconstruct_numbers_from_chunks(mutated_chunks)
    mutated_u8 = [reconstructed[i]['original_number'] for i in range(len(reconstructed))]

    # Shift back from uint8 (0-255) → signed int8 (-128..127)
    mutated_int8 = (np.array(mutated_u8) - 128).tolist()
    distortion   = sum(abs(mutated_u8[i] - vals[i]) for i in range(len(vals))) / len(vals)
    return mutated_int8, distortion


# =============================================================================
# Worker process
# =============================================================================
def worker(p, next_idx, lock, N, chunk_size, message_parity_size,
           message_size, approach, memmap_path, out_dir, p_matrix=None):
    grp = _assign_affinity(p, cpus_per_worker=2)
    if p == 0:
        print(f"[affinity] worker {p} -> CPUs {grp}", flush=True)

    arr = np.load(memmap_path, mmap_mode="r")       # uint8, shape (N,)

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = pathlib.Path(out_dir) / f"chunks_p{p}.jsonl"

    with open(out_path, "w", buffering=1) as f:
        while True:
            rng = claim_next(next_idx, lock, N, chunk_size)
            if rng is None:
                break
            start, end = rng
            values = arr[start:end + 1]
            mutated, distortion = process_payload(
                values, approach, chunk_size, message_parity_size,
                message_size, p_matrix,
            )
            rec = {
                "p":         p,
                "start":     int(start),
                "end":       int(end),
                "count":     int(end - start + 1),
                "values":    mutated,
                "distorsion": distortion,
                "status":    "ok",
            }
            if next_idx.value % (chunk_size * 1000) == 0:
                print(f"[progress] worker={p} idx={next_idx.value}", flush=True)
            f.write(json.dumps(rec) + "\n")


# =============================================================================
# Coverage validator
# =============================================================================
def validate_coverage(N: int, log_dir: str):
    intervals = []
    for name in os.listdir(log_dir):
        if not name.endswith(".jsonl"):
            continue
        with open(os.path.join(log_dir, name)) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") == "ok":
                    intervals.append((rec["start"], rec["end"]))

    if not intervals and N > 0:
        print(f"[WARN] No completed intervals found in {log_dir}")
        return False

    intervals = sorted(set(intervals))
    cur = 0
    for s, e in intervals:
        if s != cur:
            print(f"[WARN] Gap detected at index {cur}; next interval starts at {s}")
            return False
        cur = e + 1
    if cur != N:
        print(f"[WARN] Did not reach N={N}. Last covered index is {cur - 1}")
        return False
    print(f"[OK] Coverage validated: [0, {N}) with {len(intervals)} chunks.")
    return True


# =============================================================================
# Layer sanitizer (matches prepare_patterns.py's _sanitize)
# =============================================================================
def _sanitize(name: str) -> str:
    return name.replace(".", "_").replace("/", "_")


# =============================================================================
# Process one layer
# =============================================================================
def run_layer(layer_name, entry, args, t_value, chunk_size, message_parity_size,
              message_size, p_matrix, bit_label):
    weights_file = entry.get("weights_perm_file")
    if not weights_file or not os.path.exists(weights_file):
        print(f"  [skip] {layer_name}: weights_perm_file missing ({weights_file})")
        return

    # Load weights (expected int8 after upstream quantization)
    arr = np.load(weights_file)
    if arr.dtype != np.int8:
        raise TypeError(
            f"{layer_name}: weights_perm_file has dtype={arr.dtype}, expected int8. "
            f"Re-run prepare_patterns.py to regenerate the file."
        )
    # Shift int8 (-128..127) → uint8 (0..255) for binary conversion
    arr_u8 = (arr.astype(np.int16) + 128).astype(np.uint8)
    N = int(arr_u8.shape[0])

    # Output directory for this layer
    ds_lower   = args.dataset.lower()
    m_tag      = f"M{args.codeword}_t{t_value}"
    layer_safe = _sanitize(layer_name)
    out_dir    = os.path.join(
        args.chunks_dir,
        ds_lower, args.arch, "PTQ", bit_label,
        m_tag, args.approach, layer_safe,
    )
    os.makedirs(out_dir, exist_ok=True)

    # Temp memmap for worker read-only access
    memmap_path = os.path.join(out_dir, ".tmp_weights_u8.npy")
    np.save(memmap_path, arr_u8)

    # Clean up any previous chunk files for this layer
    for f in pathlib.Path(out_dir).glob("chunks_p*.jsonl"):
        try:
            f.unlink()
        except FileNotFoundError:
            pass

    print(f"  [embed] {layer_name}  N={N:,}  out={out_dir}", flush=True)

    ctx      = mp.get_context("spawn")
    next_idx = ctx.Value('q', 0)
    lock     = ctx.Lock()

    procs = [
        ctx.Process(
            target=worker,
            args=(p, next_idx, lock, N, chunk_size, message_parity_size,
                  message_size, args.approach, memmap_path, out_dir, p_matrix),
        )
        for p in range(args.workers)
    ]
    for proc in procs:
        proc.start()
    for proc in procs:
        proc.join()

    validate_coverage(N, out_dir)

    try:
        os.remove(memmap_path)
    except FileNotFoundError:
        pass


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description="Parallel ECC embedding for quantized NN weights")
    ap.add_argument("--dataset",      required=True,
                    choices=["CIFAR10", "CIFAR100", "IMAGENET"])
    ap.add_argument("--arch",         required=True,
                    choices=["resnet18", "resnet50", "mobilenet_v2", "efficientnet_b0"])
    ap.add_argument("--quant-bits",   required=True, type=int, choices=[4, 8, 16])
    ap.add_argument("--t-value",      required=True, type=int)
    ap.add_argument("--approach",     default="parfix", choices=list(_ALLOWED_APPROACHES))
    ap.add_argument("--codeword",     default=63,    type=int, choices=[63, 127, 255])
    ap.add_argument("--workers",      default=24,    type=int)
    ap.add_argument("--patterns-dir", required=True,
                    help="Root of 0-Data/artifacts/patterns/")
    ap.add_argument("--chunks-dir",   required=True,
                    help="Root of 0-Data/artifacts/embeddedECC_Chunks/")
    args = ap.parse_args()

    t_value           = args.t_value
    message_parity_size = args.codeword
    if args.approach != 'parfix':
        chunk_size = message_parity_size
    else:
        chunk_size = NANDT_TO_K[message_parity_size][t_value]
    message_size = NANDT_TO_K[message_parity_size][t_value]

    p_matrix = None
    if args.approach in ('search3', 'greedy'):
        p_matrix = build_bch_parity_matrix(n=message_parity_size, k=message_size)

    # Map quant_bits to the path label used by PatternFinder
    if args.quant_bits == 32:
        bit_label = "float32"
    else:
        bit_label = f"{args.quant_bits}-bit"

    ds_lower     = args.dataset.lower()
    manifest_path = os.path.join(
        args.patterns_dir, ds_lower, args.arch, "PTQ", bit_label,
        "pattern_manifest.json",
    )
    if not os.path.exists(manifest_path):
        print(f"[skip] No manifest found: {manifest_path}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    print(f"[ecc_embed] dataset={args.dataset}  arch={args.arch}  "
          f"bits={bit_label}  t={t_value}  approach={args.approach}  "
          f"codeword={message_parity_size}  chunk_size={chunk_size}  "
          f"message_size={message_size}  workers={args.workers}")
    print(f"[ecc_embed] manifest: {manifest_path}  ({len(manifest)} layers)")

    for layer_name, entry in manifest.items():
        run_layer(layer_name, entry, args, t_value, chunk_size,
                  message_parity_size, message_size, p_matrix, bit_label)

    print(f"[ecc_embed] Done. All layers processed.")


if __name__ == "__main__":
    main()
