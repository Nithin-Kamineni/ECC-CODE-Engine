# import random
import galois
import numpy as np
# import pulp as pl
# import time
# import math
# from collections.abc import Iterable
# from typing import Dict, Any, List, Tuple
# from collections import Counter
# from utils.solve_mod2_mip_weighted_per_spans import solve_mod2_mip_weighted_per_spans
# from utils.solve_parity_fit_cpsat import solve_parity_fit_cpsat
import itertools

def _decode_value_from_positions(bits, positions, weights):
    val = 0
    for pos in positions:
        if bits[pos]:
            val += (1 << weights[pos])
    return val


def build_bch_parity_matrix(n, k):
    """Call ONCE outside the chunk loop; galois.BCH() is expensive."""
    bch = galois.BCH(n, k)
    return np.array(bch.G[:, bch.k:], dtype=np.int8)

def GreedyEncodeAndDecode(
    chunk,
    P_matrix,
    message_parity_size=63,
    message_size=30,
    search_metric='L2',
    move_unit_range=8,
    bucket_sens=None,
):
    """
    Greedy BCH embedding search with per-bucket STEP-aware moves.

    SAME FIX AS THE ENUM VERSION
    ----------------------------
    A bucket's message-partial value is always a multiple of
        step_b = 2^(min message weight in bucket b)
    so any move that isn't a multiple of step_b either does nothing
    (positive sub-step moves) or collapses to the single "drop by
    step_b" candidate (negative sub-step moves). The original move
    set { ±1, ±2, ..., ±8 } therefore explored at most 2 distinct
    candidates per bucket — one of them a no-op clone of delta=0.

    This version moves in integer multiples of step_b:
        d ∈ { ±1, ±2, ..., ±move_unit_range } * step_b
    Each move now toggles a real message bit (or set of bits via
    carry), and the greedy can actually hill-climb.

    Parameters
    ----------
    move_unit_range : int
        Number of step-units to explore on each side, default 4.
        Effective per-bucket move set has 2*move_unit_range entries.
    """

    # ------------------------------------------------------------------
    # 0) Unpack chunk
    # ------------------------------------------------------------------
    original_message_bits_slice = chunk["sliced_message_bits"]
    original_bits_weights_slice = chunk["sliced_bit_weights"]
    original_nums_spans         = chunk.get("sliced_message_nums", [])

    bits    = list(map(int, original_message_bits_slice))
    weights = list(map(int, original_bits_weights_slice))

    if len(bits) != len(weights):
        raise ValueError("bits and weights must have the same length.")

    n = len(bits)

    if n == 0:
        return {
            "sliced_message_bits":  [],
            "sliced_bit_weights":   [],
            "sliced_message_nums":  [],
            "message_indices":      [],
            "parity_indices":       [],
            "bucket_metadata":      [],
            "best_bucket_deltas":   [],
            "best_score":           0,
            "search_metric":        search_metric,
        }

    if message_parity_size != n:
        raise ValueError(f"message_parity_size (={message_parity_size}) must equal chunk length (={n}).")
    if not (0 < message_size < n):
        raise ValueError("message_size must be in [1, n-1].")
    if any(b not in (0, 1) for b in bits):
        raise ValueError("original_message_bits_slice must contain only 0/1.")
    if search_metric not in ("L1", "L2"):
        raise ValueError("search_metric must be either 'L1' or 'L2'.")

    k = message_size

    # ------------------------------------------------------------------
    # 1) Partition bits → message (top-k by weight) and parity
    # ------------------------------------------------------------------
    ranked_idx_desc = sorted(range(n), key=lambda i: (-weights[i], i))
    message_indices = ranked_idx_desc[:k]
    parity_indices  = sorted(
        set(range(n)) - set(message_indices),
        key=lambda i: (weights[i], i)
    )

    bits_arr    = np.array(bits,    dtype=np.int8)
    weights_arr = np.array(weights, dtype=np.int64)
    msg_idx_arr = np.array(message_indices, dtype=np.int64)
    par_idx_arr = np.array(parity_indices,  dtype=np.int64)
    msg_pos_to_m_idx = {pos: m_i for m_i, pos in enumerate(message_indices)}

    # ------------------------------------------------------------------
    # 2) Build bucket metadata + per-bucket STEP
    # ------------------------------------------------------------------
    bucket_metadata        = []
    original_bucket_values = []

    for rec in original_nums_spans:
        s = rec["start"]
        e = rec["end"]

        bucket_message_positions = [pos for pos in message_indices if s <= pos < e]
        bucket_parity_positions  = [pos for pos in parity_indices  if s <= pos < e]
        bucket_msg_max   = sum(1 << weights[pos] for pos in bucket_message_positions)
        bucket_msg_value = _decode_value_from_positions(bits, bucket_message_positions, weights)

        # *** NEW: smallest representable change to this bucket's message-partial value ***
        if bucket_message_positions:
            bucket_min_msg_weight = min(weights[pos] for pos in bucket_message_positions)
            bucket_msg_step       = 1 << bucket_min_msg_weight
        else:
            bucket_min_msg_weight = 0
            bucket_msg_step       = 0     # locked anyway by has_msg check

        bucket_metadata.append({
            "index":                 rec.get("index"),
            "start":                 s,
            "end":                   e,
            "bucket_positions":      list(range(s, e)),
            "message_positions":     bucket_message_positions,
            "parity_positions":      bucket_parity_positions,
            "message_partial_value":      bucket_msg_value,
            "message_partial_max":        bucket_msg_max,
            "message_partial_step":       bucket_msg_step,         # NEW
            "message_partial_min_weight": bucket_min_msg_weight,   # NEW
            "original_value":             rec.get("value", 0),
        })
        original_bucket_values.append(rec.get("value", 0))

    num_buckets = len(bucket_metadata)

    old_vals = np.array([m["message_partial_value"] for m in bucket_metadata], dtype=np.int64)
    max_vals = np.array([m["message_partial_max"]   for m in bucket_metadata], dtype=np.int64)
    has_msg  = np.array([bool(m["message_positions"]) for m in bucket_metadata], dtype=bool)
    steps    = np.array([m["message_partial_step"]  for m in bucket_metadata], dtype=np.int64)  # NEW

    orig_vals_arr = np.array(original_bucket_values, dtype=np.int64)
    pow2          = (1 << weights_arr).astype(np.int64)
    bucket_slices = [(rec["start"], rec["end"]) for rec in original_nums_spans]
    base_m        = bits_arr[msg_idx_arr].astype(np.int8).copy()

    bucket_sens_arr = np.array(bucket_sens, dtype=np.float64) if bucket_sens is not None else None

    # ------------------------------------------------------------------
    # 3) Helper: evaluate a delta vector
    # ------------------------------------------------------------------
    def evaluate(deltas):
        # Rebuild message bits from new bucket integer values.
        # NOTE: bit extraction is correct for any delta that is a
        # multiple of the bucket's step, because then new_val stays
        # representable as a sum of message-bit values.
        m = base_m.copy()
        for b_i, meta in enumerate(bucket_metadata):
            mp = meta["message_positions"]
            if not mp:
                continue
            new_val = int(old_vals[b_i]) + int(deltas[b_i])
            for pos in mp:
                m[msg_pos_to_m_idx[pos]] = (new_val >> weights[pos]) & 1

        parity = (m.astype(np.int32) @ P_matrix.astype(np.int32)) % 2

        full = bits_arr.copy()
        full[msg_idx_arr] = m
        full[par_idx_arr] = parity.astype(np.int8)

        weighted = full.astype(np.int64) * pow2
        bucket_vals = np.array(
            [weighted[s:e].sum() for s, e in bucket_slices],
            dtype=np.int64,
        )

        diffs = bucket_vals - orig_vals_arr
        if bucket_sens_arr is not None:
            score = int((np.abs(diffs) * bucket_sens_arr).sum()) if search_metric == "L1" \
                    else int(((diffs ** 2) * bucket_sens_arr).sum())
        else:
            score = int(np.abs(diffs).sum()) if search_metric == "L1" else int((diffs ** 2).sum())
        tie_break = int(np.abs(deltas).sum())

        return score, tie_break, full, bucket_vals

    # ------------------------------------------------------------------
    # 4) Greedy hill-climbing over STEP-aligned moves
    # ------------------------------------------------------------------
    # Move alphabet in "step-units"; per-bucket actual delta = unit * step_b.
    # Excludes 0 (no-op).
    move_units = [u for u in range(-move_unit_range, move_unit_range + 1) if u != 0]

    current_deltas = np.zeros(num_buckets, dtype=np.int64)
    best_score, best_tie, best_full, best_bv = evaluate(current_deltas)

    while True:
        move_score, move_tie = best_score, best_tie
        move_b_idx = move_d = move_full = move_bv = move_deltas = None

        for b_i in range(num_buckets):
            if not has_msg[b_i]:
                continue                            # locked bucket
            step_b = int(steps[b_i])

            for u in move_units:
                d = u * step_b                      # *** SCALED MOVE ***

                # Validity: new message-partial value stays in [0, max]
                new_partial = int(old_vals[b_i]) + int(current_deltas[b_i]) + d
                if new_partial < 0 or new_partial > int(max_vals[b_i]):
                    continue

                trial = current_deltas.copy()
                trial[b_i] += d

                s, t, full, bv = evaluate(trial)

                if (s, t) < (move_score, move_tie):
                    move_score, move_tie = s, t
                    move_b_idx, move_d   = b_i, d
                    move_full, move_bv   = full, bv
                    move_deltas          = trial.copy()

        if move_b_idx is None:
            break                                   # converged

        current_deltas = move_deltas
        best_score, best_tie = move_score, move_tie
        best_full, best_bv   = move_full, move_bv

    # ------------------------------------------------------------------
    # 5) Package results
    # ------------------------------------------------------------------
    best_updated_nums = [
        {
            "index": rec.get("index"),
            "value": int(best_bv[b_i]),
            "start": rec["start"],
            "end":   rec["end"],
        }
        for b_i, rec in enumerate(original_nums_spans)
    ]

    return {
        "sliced_message_bits":  best_full.tolist(),
        "sliced_bit_weights":   original_bits_weights_slice,
        "sliced_message_nums":  best_updated_nums,
        "message_indices":      message_indices,
        "parity_indices":       parity_indices,
        "bucket_metadata":      bucket_metadata,
        "best_bucket_deltas":   current_deltas.tolist(),
        "best_score": {
            "metric":   search_metric,
            "value":    best_score,
            "delta_l1": best_tie,
        },
        "search_metric": search_metric,
    }

# values = [random.randint(0,100) for _ in range(63)]
# message_bits = convert_to_binary(values, bit_size=8)
# print('values',values)
# print()
# chunks = messageSliceBasedOnChunkSize(message_bits, chunk_size=63)

# mutated_chunks = []
# # print()
# for chunk in chunks:
#     # print('chunk',chunk)
#     mutated_chunk = Search3EncodeAndDecode(
#                 chunk,
#                 message_parity_size=63,
#                 message_size=30,
#               )
#     # print()
#     # print('mutated_chunk',mutated_chunk)
#     # print('--------------------------')
#     mutated_chunks.append(mutated_chunk)

# reconstructed_chunks = reconstruct_numbers_from_chunks(mutated_chunks)
# mutated_nums = [reconstructed_chunks[i]['original_number'] for i in range(len(reconstructed_chunks))]
# print('mutated_nums',mutated_nums)
# print(sum([abs(mutated_nums[i]-values[i]) for i in range(len(values))])/len(values))