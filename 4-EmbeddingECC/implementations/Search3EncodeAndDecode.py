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

def Search3EncodeAndDecode(
    chunk,
    P_matrix,
    message_parity_size=63,
    message_size=30,
    search_metric='L2',
    bucket_sens=None,
):
    """
    BCH embedding with bucket perturbation search.

    SEARCH STEP — THE FIX
    ---------------------
    For each bucket, the per-bucket perturbation set is now

        { -2^w_min ,  0 ,  +2^w_min }

    where  w_min  is the LOWEST message-bit weight inside that bucket.
    That is the smallest representable change to the bucket's message-
    partial value — it corresponds to toggling the lowest-weight
    message bit.

    Why this matters: the partition picks message bits as the top-k by
    weight, so a bucket's message bits sit at its HIGHEST weights and
    its message-partial value is always a multiple of 2^w_min. A naive
    delta of ±1 falls below every message bit, so:
        • delta = +1  →  bits at message weights are unchanged
                          (candidate is a duplicate of delta=0)
        • delta = -1  →  forces a borrow that flips the w_min bit,
                          producing an effective change of -2^w_min
                          (asymmetric and coarser than intended)
    Scaling by  step = 2^w_min  makes ±1 actually mean "toggle the
    lowest message bit", giving three genuinely distinct, symmetric
    candidates per bucket. The full bucket value can still take
    finer values than multiples of step, because BCH recomputes
    the parity bits from the mutated message and those parity bits
    contribute to the bucket's full reconstructed value.
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
            "sliced_message_bits": [], "sliced_bit_weights": [],
            "sliced_message_nums": [], "message_indices": [],
            "parity_indices": [], "bucket_metadata": [],
            "best_bucket_deltas": [], "best_score": 0,
            "search_metric": search_metric,
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
    # 1) Partition: top-k weights → message; rest → parity
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
    # 2) Per-bucket metadata — and the per-bucket STEP that step 3/4 use
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
            bucket_msg_step       = 0          # locked anyway by has_msg check

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

    # delta INDICES used in combos. Each is multiplied by the per-bucket step.
    delta_vals = [-1, 0, 1]

    # ------------------------------------------------------------------
    # 3) Combo enumeration + validity — now uses per-bucket step
    # ------------------------------------------------------------------
    combo_arr = np.array(
        list(itertools.product(range(3), repeat=num_buckets)),
        dtype=np.int8
    )                                                              # (C, B)

    old_vals = np.array([m["message_partial_value"] for m in bucket_metadata], dtype=np.int64)
    max_vals = np.array([m["message_partial_max"]   for m in bucket_metadata], dtype=np.int64)
    has_msg  = np.array([bool(m["message_positions"]) for m in bucket_metadata], dtype=bool)
    steps    = np.array([m["message_partial_step"]  for m in bucket_metadata], dtype=np.int64)  # NEW

    delta_map        = np.array(delta_vals, dtype=np.int64)        # [-1, 0, +1]
    deltas_per_combo = delta_map[combo_arr.astype(np.int64)]       # (C, B), in {-1,0,+1}

    # *** CHANGED: scale deltas by per-bucket step before the range check ***
    new_vals = old_vals[np.newaxis, :] + deltas_per_combo * steps[np.newaxis, :]   # (C, B)

    no_msg_violated = (~has_msg[np.newaxis, :]) & (deltas_per_combo != 0)
    out_of_range    = (new_vals < 0) | (new_vals > max_vals[np.newaxis, :])

    valid_mask = ~(no_msg_violated | out_of_range).any(axis=1)
    valid_idx  = np.where(valid_mask)[0]
    num_valid  = len(valid_idx)

    # ------------------------------------------------------------------
    # 4) Build per-(bucket, delta) atoms — now uses per-bucket step
    # ------------------------------------------------------------------
    base_m = bits_arr[message_indices].astype(np.int64)            # (k,)
    bucket_delta_vecs = np.tile(base_m, (num_buckets, 3, 1)).astype(np.int64)   # (B, 3, k)

    for b_idx, meta in enumerate(bucket_metadata):
        msg_pos = meta["message_positions"]
        if not msg_pos:
            continue
        old_val = meta["message_partial_value"]
        max_val = meta["message_partial_max"]
        step    = meta["message_partial_step"]                      # NEW
        for d_i, delta in enumerate(delta_vals):
            new_val = old_val + delta * step                        # *** CHANGED ***
            if new_val < 0 or new_val > max_val:
                continue
            for pos in msg_pos:
                w   = weights[pos]
                m_i = msg_pos_to_m_idx[pos]
                bucket_delta_vecs[b_idx, d_i, m_i] = (new_val >> w) & 1

    bucket_diffs = bucket_delta_vecs - base_m[np.newaxis, np.newaxis, :]   # (B, 3, k)

    b_range = np.arange(num_buckets)
    selected_diffs = bucket_diffs[
        b_range[np.newaxis, :],
        combo_arr.astype(np.int64),
    ]                                                              # (C, B, k)

    all_messages = np.clip(
        base_m[np.newaxis, :] + selected_diffs.sum(axis=1),
        0, 1
    ).astype(np.int8)                                              # (C, k)

    valid_messages = all_messages[valid_idx]                       # (V, k)

    # ------------------------------------------------------------------
    # 5) Parity from mutated message — single matmul
    # ------------------------------------------------------------------
    all_parities = (
        valid_messages.astype(np.int32) @ P_matrix.astype(np.int32)
    ) % 2
    all_parities = all_parities.astype(np.int8)                    # (V, n-k)

    # ------------------------------------------------------------------
    # 6) Reassemble full n-bit candidates
    # ------------------------------------------------------------------
    all_bits = np.tile(bits_arr, (num_valid, 1))                   # (V, n)
    all_bits[:, msg_idx_arr] = valid_messages
    all_bits[:, par_idx_arr] = all_parities

    # ------------------------------------------------------------------
    # 7) Score on FULL reconstructed bucket values (message + parity)
    # ------------------------------------------------------------------
    all_pow2 = (1 << weights_arr).astype(np.int64)                 # (n,)
    weighted = all_bits.astype(np.int64) * all_pow2[np.newaxis, :] # (V, n)

    n_spans  = len(original_nums_spans)
    selector = np.zeros((n, n_spans), dtype=np.int64)
    for b_idx, span in enumerate(original_nums_spans):
        selector[span["start"]:span["end"], b_idx] = 1

    all_bucket_vals = weighted @ selector                          # (V, B)

    orig_vals = np.array(original_bucket_values, dtype=np.int64)   # (B,)
    diffs     = all_bucket_vals - orig_vals[np.newaxis, :]         # (V, B)

    if bucket_sens is not None:
        bucket_sens_arr = np.array(bucket_sens, dtype=np.float64)  # (B,)
        if search_metric == "L1":
            scores = (np.abs(diffs) * bucket_sens_arr[np.newaxis, :]).sum(axis=1)
        else:
            scores = ((diffs ** 2) * bucket_sens_arr[np.newaxis, :]).sum(axis=1)
    elif search_metric == "L1":
        scores = np.abs(diffs).sum(axis=1)
    else:
        scores = (diffs * diffs).sum(axis=1)

    # ------------------------------------------------------------------
    # 8) Tie-break: prefer fewer perturbed buckets among equal scores
    # ------------------------------------------------------------------
    valid_deltas    = deltas_per_combo[valid_idx]                  # (V, B), ∈ {-1,0,+1}
    tie_break_costs = np.abs(valid_deltas).sum(axis=1)             # (V,)

    lex_keys   = scores * (int(tie_break_costs.max()) + 1) + tie_break_costs
    best_local = int(np.argmin(lex_keys))

    best_bits        = all_bits[best_local].tolist()
    best_deltas_list = valid_deltas[best_local].tolist()
    best_score_val   = int(scores[best_local])
    best_tie_break   = int(tie_break_costs[best_local])

    # ------------------------------------------------------------------
    # 9) Per-bucket reconstructed values for the winner
    # ------------------------------------------------------------------
    best_bucket_vals_vec = all_bucket_vals[best_local]
    best_updated_nums = [
        {
            "index": rec.get("index"),
            "value": int(best_bucket_vals_vec[b_idx]),
            "start": rec["start"],
            "end":   rec["end"],
        }
        for b_idx, rec in enumerate(original_nums_spans)
    ]

    return {
        "sliced_message_bits":  best_bits,
        "sliced_bit_weights":   original_bits_weights_slice,
        "sliced_message_nums":  best_updated_nums,
        "message_indices":      message_indices,
        "parity_indices":       parity_indices,
        "bucket_metadata":      bucket_metadata,
        "best_bucket_deltas":   best_deltas_list,
        "best_score": {
            "metric":   search_metric,
            "value":    best_score_val,
            "delta_l1": best_tie_break,
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