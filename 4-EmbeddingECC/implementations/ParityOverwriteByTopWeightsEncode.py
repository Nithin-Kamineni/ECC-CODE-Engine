import random
import galois
import numpy as np
import pulp as pl
import time
import math
from collections.abc import Iterable
from typing import Dict, Any, List, Tuple
from collections import Counter

def ParityOverwriteByTopWeightsEncode(
    chunk,
    message_parity_size=63,
    message_size=30,
):
    """
    Select the top-weighted positions (count = message_size) from a chunk,
    use their bits (in that rank order) as the BCH message, compute parity,
    and overwrite the remaining (lower-weight) positions with the parity bits.

    Args:
        original_message_bits_slice: list[int] of 0/1, length n
        original_bits_weights_slice: list[int] weights (0=LSB ...), length n
        message_parity_size: n (must match len(slice))
        message_size: k (number of top-weighted positions to keep as message)

    Returns:
        dict with:
          - "mutated_bits": list[int] length n (message on top-weight idx, parity on the rest)
          - "message_indices": list[int] length k (positions used as message, in the exact order fed to BCH)
          - "parity_indices":  list[int] length n-k (positions overwritten with parity, in the order parity was placed)
          - "parity_bits":     list[int] length n-k (parity computed by BCH, aligned to parity_indices)
    """

    original_message_bits_slice = chunk['sliced_message_bits']
    original_bits_weights_slice = chunk['sliced_bit_weights']
    original_nums_spans        = chunk.get('sliced_message_nums', [])  # list of dicts with index,start,end

    
    bits = list(map(int, original_message_bits_slice))
    weights = list(map(int, original_bits_weights_slice))
    
    if len(bits) != len(weights):
        raise ValueError("bits and weights must have the same length.")
    n = len(bits)
    if n == 0:
        return {"mutated_bits": [], "message_indices": [], "parity_indices": [], "parity_bits": []}
    if message_parity_size != n:
        raise ValueError(f"message_parity_size (={message_parity_size}) must equal chunk length (={n}).")
    if not (0 < message_size < n):
        raise ValueError("message_size must be in [1, n-1].")
    if any(b not in (0,1) for b in bits):
        raise ValueError("original_message_bits_slice must contain only 0/1.")
    k = message_size

    # --- 1) Rank positions by weight for message selection ---
    # Top-k by weight (desc). Tie-breaker: lower index first (stable).
    ranked_idx_desc = sorted(range(n), key=lambda i: (-weights[i], i))
    message_indices = ranked_idx_desc[:k]

    # Remaining indices are parity slots; put parity into the *lowest weights first*
    parity_indices = sorted(
        set(range(n)) - set(message_indices),
        key=lambda i: (weights[i], i)   # ascending weight → "LSB first"
    )
    if len(parity_indices) != (n - k):
        raise RuntimeError("Internal error: parity index count mismatch.")

    # --- 2) Build BCH(n,k) with systematic generator (I | P) ---
    bch = galois.BCH(n, k)     # requires n = 2^m - 1; e.g., 63, 127, ...
    G = bch.G                  # shape (k, n)
    P_from_G = G[:, bch.k:]    # shape (k, n-k); systematic parity

    # --- 3) Form the message vector in the exact order of message_indices ---
    # NOTE: message order == rank order of the top weights (not original spatial order)
    m_vec = np.array([bits[i] for i in message_indices], dtype=int)
    m0 = bch.field(m_vec)                 # GF element row vector length k
    p0 = m0 @ P_from_G                    # GF vector length n-k
    parity_bits = np.array(p0, dtype=int).tolist()

    # --- 4) Overwrite low-weight slots with parity bits (in ascending weight order) ---
    mutated = bits[:]                     # copy
    if len(parity_bits) != len(parity_indices):
        raise RuntimeError("Parity length does not match the number of parity slots.")
    for pos, pb in zip(parity_indices, parity_bits):
        mutated[pos] = int(pb)

    # Optional: sanity check — recompute parity from *current* message positions and compare.
    # (The codeword is *not* contiguous [message|parity]; we just validate parity placement.)
    GF2 = galois.GF(2)
    # Re-extract the message bits (they didn't change)
    m_chk = GF2(np.array([mutated[i] for i in message_indices], dtype=int))
    p_chk = m_chk @ P_from_G
    assert p_chk.tolist() == parity_bits, "Parity verification failed; mapping mismatch."

    # 5) UPDATE sliced_message_nums based on mutated bits and original spans
    #    For each record {index, start, end}, recompute partial value from mutated bits
    #    within [start, end) using the chunk weights.
    updated_nums = []
    for rec in original_nums_spans:
        # Some splitters only include overlapping numbers; keep the same list semantics
        idx = rec.get("index")
        s = rec.get("start")
        e = rec.get("end")
        if s is None or e is None or s >= e:
            updated_nums.append({"index": idx, "value": 0, "start": s, "end": e})
            continue

        # Compute partial value from the mutated bits in this local slice
        # numeric value = sum( mutated[i] * 2^(weights[i]) ) for i in [s, e)
        val = 0
        for i in range(s, e):
            if mutated[i]:
                val += (1 << weights[i])

        updated_nums.append({
            "index": idx,
            "value": val,
            "start": s,
            "end": e
        })


    return {
        "sliced_message_bits": mutated,
        "sliced_bit_weights": original_bits_weights_slice,
        "message_indices": message_indices,
        "parity_indices": parity_indices,
        # "parity_bits": parity_bits,
        "sliced_message_nums": updated_nums
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
#     mutated_chunk = ParityOverwriteByTopWeightsEncode(
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