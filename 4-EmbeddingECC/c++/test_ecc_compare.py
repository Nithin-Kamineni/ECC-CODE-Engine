#!/usr/bin/env python3
"""
test_ecc_compare.py — Compare Python vs C++ ECC embedding on a small test case.

This script has two modes driven by run_test.sh:

  Mode 1: --mode generate  (run in Python SIF)
    Generates a deterministic test case (63 uint8 values), builds the BCH
    parity matrix using galois.BCH, runs the Python search3/greedy algorithm,
    and saves both the test inputs and the expected outputs to --workdir.

  Mode 2: --mode compare   (run in Python SIF, after C++ has been run)
    Reads the Python output (from mode 1) and the C++ output (from test_ecc),
    prints a detailed side-by-side comparison showing where they differ.

The C++ binary (test_ecc) is run separately by run_test.sh inside the C++ SIF:
    singularity exec --bind /blue ecc_cpp.sif ./test_ecc \\
        --input WORKDIR/test_input.npy \\
        --parity-matrix WORKDIR/test_P.npy \\
        --n 63 --t T --approach APPROACH \\
        --output WORKDIR/cpp_output.json [--verbose]

See run_test.sh for the full orchestration.
"""
import argparse
import json
import os
import sys
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# BCH parameters
# ──────────────────────────────────────────────────────────────────────────────
NANDT_TO_K = {
    (63, 1): 57, (63, 2): 51, (63, 3): 45, (63, 4): 39,
    (63, 5): 36, (63, 6): 30, (63, 7): 24, (63, 8): 18,
}

# ──────────────────────────────────────────────────────────────────────────────
# Transparent Python search3 implementation (for reference comparison)
# ──────────────────────────────────────────────────────────────────────────────

def partition_bits(bits, weights, k):
    """Top-k by weight (desc weight, asc index) → message; rest → parity."""
    n = len(bits)
    order = sorted(range(n), key=lambda i: (-weights[i], i))
    msg_idx = order[:k]
    par_idx = sorted(set(range(n)) - set(msg_idx), key=lambda i: (weights[i], i))
    return msg_idx, par_idx

def slice_chunk(u8_vals, chunk_size=63, bit_size=8):
    """Exact port of slicing.h slice_into_chunks — returns list of (bits, weights, buckets)."""
    total_bits = len(u8_vals) * bit_size
    chunks = []
    for chunk_start in range(0, total_bits, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_bits)
        bits, weights, buckets = [], [], []
        start_num = chunk_start // bit_size
        end_num   = (chunk_end - 1) // bit_size
        for j in range(start_num, end_num + 1):
            block_start = j * bit_size
            ov0 = max(block_start, chunk_start)
            ov1 = min(block_start + bit_size, chunk_end)
            if ov0 >= ov1:
                continue
            local_start = ov0 - block_start
            local_end   = ov1 - block_start
            cstart = len(bits)
            v = int(u8_vals[j])
            for b in range(local_start, local_end):
                weight = bit_size - 1 - b
                bit    = (v >> (bit_size - 1 - b)) & 1
                bits.append(bit)
                weights.append(weight)
            cend = len(bits)
            length  = local_end - local_start
            pos_low = bit_size - local_end
            mask    = ((1 << length) - 1) << pos_low
            partial_val = v & mask
            buckets.append((j, partial_val, cstart, cend))
        while len(bits) < chunk_size:
            bits.append(0);  weights.append(0)
        chunks.append((bits, weights, buckets))
    return chunks

def python_search3_chunk(bits, weights, buckets, P, k, verbose=False):
    """
    Exhaustive 3^B search (Python reference).
    Returns mutated bits (list of 0/1).

    Key correctness points that differ from the current C++ implementation:
      1. old_msg = MESSAGE-PARTIAL value (sum of 2^w for message bits that are 1)
         NOT the full bucket value (which includes parity bit contributions too)
      2. Range check uses old_msg vs max_msg (both message-partial) → correct
      3. Bit extraction: (old_msg + delta) >> w → correct
      4. Score: FULL bucket value change (message + parity contributions)
    """
    n = len(bits)
    msg_idx, par_idx = partition_bits(bits, weights, k)
    msg_pos_to_mi = {pos: mi for mi, pos in enumerate(msg_idx)}

    # Build bucket metadata
    bk_meta = []
    for idx, orig_full, bk_start, bk_end in buckets:
        mp = [pos for pos in msg_idx if bk_start <= pos < bk_end]
        # *** old_msg = MESSAGE PARTIAL (only message bits) ***
        old_msg = sum((1 << weights[pos]) for pos in mp if bits[pos])
        max_msg = sum((1 << weights[pos]) for pos in mp)
        step    = (1 << min(weights[pos] for pos in mp)) if mp else 0
        bk_meta.append({
            'start': bk_start, 'end': bk_end, 'mp': mp,
            'old_msg': old_msg,   # message-partial for range + bit extraction
            'max_msg': max_msg,
            'step':    step,
            'orig_full': orig_full,   # FULL bucket value for score
            'has_msg': bool(mp),
        })

    if verbose:
        print(f"\n[Python] msg_idx: {msg_idx}")
        print(f"[Python] par_idx: {par_idx}")
        for i, m in enumerate(bk_meta):
            print(f"  b={i}: range=[{m['start']},{m['end']}) msg_pos={m['mp']} "
                  f"old_msg={m['old_msg']} max_msg={m['max_msg']} "
                  f"step={m['step']} orig_full={m['orig_full']}")

    B = len(bk_meta)
    bits_arr    = np.array(bits, dtype=np.int8)
    msg_idx_arr = np.array(msg_idx, dtype=np.int64)
    par_idx_arr = np.array(par_idx, dtype=np.int64)
    pow2        = np.array([1 << w for w in weights], dtype=np.int64)
    orig_fulls  = np.array([m['orig_full'] for m in bk_meta], dtype=np.int64)
    base_m      = bits_arr[msg_idx_arr].copy()
    bk_slices   = [(m['start'], m['end']) for m in bk_meta]

    def evaluate(deltas):
        """Python evaluation: message-partial deltas → full bit change → full score."""
        m = base_m.copy()
        for bi, meta in enumerate(bk_meta):
            if not meta['mp']:
                continue
            # new_msg_val = message-partial value after delta
            new_msg_val = meta['old_msg'] + int(deltas[bi])
            for pos in meta['mp']:
                mi = msg_pos_to_mi[pos]
                m[mi] = (new_msg_val >> weights[pos]) & 1

        parity = (m.astype(np.int32) @ P.astype(np.int32)) % 2
        full = bits_arr.copy()
        full[msg_idx_arr] = m
        full[par_idx_arr] = parity.astype(np.int8)

        # Score on FULL bucket value change (message + parity contributions)
        weighted = full.astype(np.int64) * pow2
        bk_vals = np.array([weighted[s:e].sum() for s, e in bk_slices], dtype=np.int64)
        diffs   = bk_vals - orig_fulls
        score   = int((diffs ** 2).sum())
        tie     = int(np.abs(deltas).sum())
        return score, tie, full.tolist()

    # Exhaustive 3^B enumeration
    best_score, best_tie, best_bits = None, None, bits[:]
    if verbose:
        print(f"\n[Python] Searching {3**B} combinations (B={B}) ...")

    for ci in range(3 ** B):
        tmp = ci
        deltas = np.zeros(B, dtype=np.int64)
        valid = True
        for bi in range(B - 1, -1, -1):
            d_unit = (tmp % 3) - 1  # 0→-1, 1→0, 2→+1
            tmp //= 3
            step = bk_meta[bi]['step']
            d = d_unit * step
            new_mp = bk_meta[bi]['old_msg'] + d
            if d_unit != 0 and not bk_meta[bi]['has_msg']:
                valid = False; break
            if new_mp < 0 or new_mp > bk_meta[bi]['max_msg']:
                valid = False; break
            deltas[bi] = d

        if not valid:
            continue

        s, t, full = evaluate(deltas)
        if best_score is None or (s, t) < (best_score, best_tie):
            best_score, best_tie, best_bits = s, t, full
            if verbose:
                print(f"  [ci={ci}] NEW BEST: score={s} tie={t} deltas={deltas.tolist()}")

    if verbose:
        print(f"\n[Python] Final: score={best_score} tie={best_tie}")
    return best_bits

def python_greedy_chunk(bits, weights, buckets, P, k, move_unit_range=4, verbose=False):
    """
    Greedy hill-climbing BCH embedding (Python reference).
    Returns mutated bits (list of 0/1).

    Mirrors GreedyEncodeAndDecode.py with:
      - old_msg  = MESSAGE-PARTIAL value (range checks + bit extraction)
      - score    = FULL bucket value change (message + parity contributions)
      - tie      = sum of |message-partial deltas| (matching C++ greedy.h)
      - move set = {u * step_b : u in ±1..±move_unit_range, new_partial in [0, max_msg]}
    """
    n = len(bits)
    msg_idx, par_idx = partition_bits(bits, weights, k)
    msg_pos_to_mi = {pos: mi for mi, pos in enumerate(msg_idx)}

    # Build bucket metadata (same as search3 but keep orig_full separate)
    bk_meta = []
    for idx, orig_full, bk_start, bk_end in buckets:
        mp = [pos for pos in msg_idx if bk_start <= pos < bk_end]
        old_msg = sum((1 << weights[pos]) for pos in mp if bits[pos])
        max_msg = sum((1 << weights[pos]) for pos in mp)
        step    = (1 << min(weights[pos] for pos in mp)) if mp else 0
        bk_meta.append({
            'start': bk_start, 'end': bk_end, 'mp': mp,
            'old_msg': old_msg,
            'max_msg': max_msg,
            'step':    step,
            'orig_full': orig_full,
            'has_msg': bool(mp),
        })

    B = len(bk_meta)
    bits_arr    = np.array(bits, dtype=np.int8)
    msg_idx_arr = np.array(msg_idx, dtype=np.int64)
    par_idx_arr = np.array(par_idx, dtype=np.int64)
    pow2        = np.array([1 << w for w in weights], dtype=np.int64)
    orig_fulls  = np.array([m['orig_full'] for m in bk_meta], dtype=np.int64)
    base_m      = bits_arr[msg_idx_arr].copy()
    bk_slices   = [(m['start'], m['end']) for m in bk_meta]

    # Current message-partial values (mutable as hill-climbing proceeds)
    cur_msg = np.array([m['old_msg'] for m in bk_meta], dtype=np.int64)

    def evaluate(deltas):
        """Evaluate a delta vector: returns (score, tie, full_bits)."""
        m = base_m.copy()
        for bi, meta in enumerate(bk_meta):
            if not meta['mp']:
                continue
            new_val = int(cur_msg[bi]) + int(deltas[bi])
            for pos in meta['mp']:
                m[msg_pos_to_mi[pos]] = (new_val >> weights[pos]) & 1
        parity = (m.astype(np.int32) @ P.astype(np.int32)) % 2
        full = bits_arr.copy()
        full[msg_idx_arr] = m
        full[par_idx_arr] = parity.astype(np.int8)
        # Full bucket value change for score
        weighted = full.astype(np.int64) * pow2
        bk_vals  = np.array([weighted[s:e].sum() for s, e in bk_slices], dtype=np.int64)
        diffs    = bk_vals - orig_fulls
        score    = int((diffs ** 2).sum())
        tie      = int(np.abs(deltas).sum())
        return score, tie, full.tolist()

    # Move units: ±1 .. ±move_unit_range (scaled by step_b per bucket)
    move_units = [u for u in range(-move_unit_range, move_unit_range + 1) if u != 0]
    current_deltas = np.zeros(B, dtype=np.int64)
    best_score, best_tie, best_bits = evaluate(current_deltas)

    while True:
        move_score, move_tie = best_score, best_tie
        move_b, move_d, move_deltas, move_bits = None, None, None, None

        for bi in range(B):
            if not bk_meta[bi]['has_msg']:
                continue
            step_b = bk_meta[bi]['step']
            max_msg = bk_meta[bi]['max_msg']
            for u in move_units:
                d = u * step_b
                new_partial = int(cur_msg[bi]) + int(current_deltas[bi]) + d
                if new_partial < 0 or new_partial > max_msg:
                    continue
                trial = current_deltas.copy()
                trial[bi] += d
                s, t, full = evaluate(trial)
                if (s, t) < (move_score, move_tie):
                    move_score, move_tie = s, t
                    move_b, move_d = bi, d
                    move_deltas, move_bits = trial.copy(), full

        if move_b is None:
            break  # converged

        current_deltas = move_deltas
        best_score, best_tie, best_bits = move_score, move_tie, move_bits
        if verbose:
            print(f"  [greedy] improved: score={best_score} tie={best_tie} b={move_b} d={move_d}")

    if verbose:
        print(f"\n[Python greedy] Final: score={best_score} tie={best_tie}")
    return best_bits


def reconstruct_u8(chunks_with_bits, num_vals):
    """
    Reconstruct uint8 values from mutated chunk bits.

    A uint8 value can span TWO chunks (when its 8 bits straddle a chunk
    boundary). Each chunk contributes a partial value; we OR them together
    (same as C++ reconstruct_from_chunks which uses |=).
    """
    out = [0] * num_vals
    for (bits, weights, buckets) in chunks_with_bits:
        # Single pass — never reset; OR-accumulate partial contributions
        for idx, _, bk_start, bk_end in buckets:
            if 0 <= idx < num_vals:
                for pos in range(bk_start, bk_end):
                    if bits[pos]:
                        out[idx] |= (1 << weights[pos])
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Mode: generate  (run Python, save all data to workdir)
# ──────────────────────────────────────────────────────────────────────────────
def mode_generate(args):
    import galois

    t       = args.t
    n       = args.n
    k       = NANDT_TO_K[(n, t)]
    seed    = args.seed
    n_vals  = args.n_vals
    workdir = args.workdir
    os.makedirs(workdir, exist_ok=True)

    print(f"[generate] BCH(n={n}, k={k}, t={t}), approach={args.approach}, seed={seed}")
    print(f"[generate] n_vals={n_vals}, workdir={workdir}")

    # Generate test values
    rng    = np.random.default_rng(seed)
    u8_vals = rng.integers(0, 256, size=n_vals, dtype=np.uint8)
    print(f"\n[generate] Input uint8[{n_vals}]: {u8_vals.tolist()}")

    # Save int8 (.npy) for C++ binary
    i8_vals = (u8_vals.astype(np.int32) - 128).astype(np.int8)
    input_path = os.path.join(workdir, 'test_input.npy')
    np.save(input_path, i8_vals)
    print(f"[generate] Saved input → {input_path}")

    # Build and save BCH parity matrix
    print(f"[generate] Building BCH({n},{k}) parity matrix via galois ...")
    bch = galois.BCH(n, k)
    P   = np.array(bch.G[:, bch.k:], dtype=np.int8)
    pmat_path = os.path.join(workdir, f'test_P_t{t}.npy')
    np.save(pmat_path, P)
    print(f"[generate] P shape={P.shape} → {pmat_path}")

    # Run Python algorithm
    print(f"\n[generate] Running Python {args.approach} ...")
    chunks = slice_chunk(u8_vals.tolist(), chunk_size=n, bit_size=8)
    out_chunks = []
    for ci, (bits, weights, buckets) in enumerate(chunks):
        if args.approach == 'search3':
            mutated = python_search3_chunk(bits, weights, buckets, P, k,
                                           verbose=args.verbose)
        else:  # greedy
            mutated = python_greedy_chunk(bits, weights, buckets, P, k,
                                          move_unit_range=4,
                                          verbose=args.verbose)
        out_chunks.append((mutated, weights, buckets))

    py_u8 = reconstruct_u8(out_chunks, n_vals)
    py_l1 = sum(abs(int(py_u8[i]) - int(u8_vals[i])) for i in range(n_vals))
    print(f"\n[generate] Python output uint8: {py_u8}")
    print(f"[generate] Python L1 distortion: {py_l1}")

    # Save Python output
    py_out_path = os.path.join(workdir, 'py_output.json')
    with open(py_out_path, 'w') as f:
        json.dump({'values': py_u8, 'l1': py_l1, 'n': n, 'k': k, 't': t,
                   'approach': args.approach, 'n_vals': n_vals,
                   'input': u8_vals.tolist()}, f, indent=2)
    print(f"[generate] Saved Python output → {py_out_path}")
    print(f"\n[generate] NEXT: run C++ binary inside C++ SIF:")
    print(f"  singularity exec --bind /blue ecc_cpp.sif ./test_ecc \\")
    print(f"      --input {input_path} \\")
    print(f"      --parity-matrix {pmat_path} \\")
    print(f"      --n {n} --t {t} --approach {args.approach} \\")
    print(f"      --output {workdir}/cpp_output.json --verbose")


# ──────────────────────────────────────────────────────────────────────────────
# Mode: compare  (read both outputs, print diff)
# ──────────────────────────────────────────────────────────────────────────────
def mode_compare(args):
    workdir = args.workdir
    py_path  = os.path.join(workdir, 'py_output.json')
    cpp_path = os.path.join(workdir, 'cpp_output.json')

    if not os.path.exists(py_path):
        sys.exit(f"[compare] Python output not found: {py_path}\nRun --mode generate first.")
    if not os.path.exists(cpp_path):
        sys.exit(f"[compare] C++ output not found: {cpp_path}\nRun test_ecc binary first.")

    with open(py_path) as f:
        py_data = json.load(f)
    with open(cpp_path) as f:
        cpp_data = json.load(f)

    n_vals = py_data['n_vals']
    orig   = np.array(py_data['input'], dtype=np.int32)
    py_out = np.array(py_data['values'], dtype=np.int32)
    cpp_out= np.array(cpp_data['values'], dtype=np.int32)

    if len(cpp_out) != n_vals:
        print(f"[compare] WARNING: C++ returned {len(cpp_out)} values, expected {n_vals}")

    n_compare = min(len(py_out), len(cpp_out), n_vals)

    print(f"\n{'='*60}")
    print(f"ECC COMPARISON: BCH(n={py_data['n']}, k={py_data['k']}, t={py_data['t']})")
    print(f"approach={py_data['approach']}, n_vals={n_vals}")
    print(f"{'='*60}")
    print(f"\n{'idx':>5}  {'orig':>5}  {'python':>7}  {'cpp':>6}  {'py_Δ':>6}  {'cpp_Δ':>6}  match")
    print("-" * 60)

    mismatches = 0
    for i in range(n_compare):
        py_v  = py_out[i]
        cpp_v = cpp_out[i]
        match = "✓" if py_v == cpp_v else "✗"
        if py_v != cpp_v:
            mismatches += 1
        print(f"  {i:3d}   {orig[i]:5d}   {py_v:6d}   {cpp_v:5d}   {py_v-orig[i]:+5d}   {cpp_v-orig[i]:+5d}   {match}")

    py_l1  = int(np.abs(py_out[:n_compare]  - orig[:n_compare]).sum())
    cpp_l1 = int(np.abs(cpp_out[:n_compare] - orig[:n_compare]).sum())

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"  Mismatches    : {mismatches} / {n_compare}")
    print(f"  Python  L1    : {py_l1}")
    print(f"  C++     L1    : {cpp_l1}")
    if mismatches == 0:
        print(f"\n  ✓ OUTPUTS MATCH EXACTLY — no divergence found.")
    else:
        delta = cpp_l1 - py_l1
        sign  = "more" if delta > 0 else "less"
        print(f"\n  ✗ C++ has {abs(delta)} {sign} distortion than Python")
        if delta > 0:
            print(f"    → C++ is WORSE (not preserving weights as well as Python)")
        else:
            print(f"    → C++ is BETTER (lower distortion than Python reference)")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',     required=True, choices=['generate', 'compare'],
                        help='generate: run Python + save data; compare: diff saved outputs')
    parser.add_argument('--workdir',  default='/tmp/ecc_test',
                        help='Directory to store test data and outputs (default /tmp/ecc_test)')
    parser.add_argument('--t',        type=int, default=2)
    parser.add_argument('--n',        type=int, default=63)
    parser.add_argument('--approach', default='search3',
                        choices=['search3', 'greedy'])
    parser.add_argument('--n-vals',   type=int, default=63)
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--verbose',  action='store_true')
    args = parser.parse_args()

    if (args.n, args.t) not in NANDT_TO_K:
        sys.exit(f"Unsupported (n={args.n}, t={args.t})")

    if args.mode == 'generate':
        mode_generate(args)
    else:
        mode_compare(args)


if __name__ == '__main__':
    main()
