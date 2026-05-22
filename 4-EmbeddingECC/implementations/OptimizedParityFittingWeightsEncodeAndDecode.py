import random
import galois
import numpy as np
import pulp as pl
import time
import math
from collections.abc import Iterable
from typing import Dict, Any, List, Tuple
from collections import Counter
# from utils.solve_mod2_mip_weighted_per_spans import solve_mod2_mip_weighted_per_spans
from utils.solve_parity_fit_cpsat import solve_parity_fit_cpsat

def OptimizedParityFittingWeightsEncodeAndDecode(
    chunk,
    message_parity_size=63,
    message_size=30,
    errors=0,
    sensitivity_index=None,   # kept for signature compatibility (unused here)
    warm_start=None,
    solver='cplex',
    ExternalParityBits=0,
):
    """
    - Select top-`message_size` positions by weight as message indices.
    - Use BCH(n=message_parity_size, k=message_size) to impose parity via MIP.
    - Verify parity using the index mapping.
    - (Optional) Inject `errors` random flips into the canonical codeword [xm|xp] and
      verify BCH decoding corrects them.
    - Recompute sliced_message_nums from mutated bits and original spans.
    - Return the mutated chunk dict.
    """
    # Extract chunk slices
    original_message_bits_slice = list(map(int, chunk['sliced_message_bits']))
    original_bits_weights_slice = list(map(int, chunk['sliced_bit_weights']))
    original_nums_spans         = chunk.get('sliced_message_nums', [])  # [{index,start,end,value?}, ...]

    n = len(original_message_bits_slice)
    if n == 0:
        return {
            "sliced_message_bits": [],
            "sliced_bit_weights": [],
            "message_indices": [],
            "parity_indices": [],
            "sliced_message_nums": []
        }
    if message_parity_size-ExternalParityBits != n:
        raise ValueError(f"message_parity_size ({message_parity_size}-{ExternalParityBits}) must equal chunk length ({n}).")
    if not (0 < message_size < n):
        raise ValueError("message_size must be in [1, n-1].")
    if any(b not in (0,1) for b in original_message_bits_slice):
        raise ValueError("chunk bits must be 0/1.")

    k = message_size
    # BCH setup (n must be 2^m - 1, e.g., 63)
    bch = galois.BCH(message_parity_size, message_size)
    G = bch.G
    P_from_G = G[:, bch.k:]              # (k, n-k)
    encoding_parity_matrix = np.asarray(P_from_G.T)

    # Select indices: top-k weights as message; rest (ascending weight) as parity
    ranked_idx_desc = sorted(range(n), key=lambda i: (-original_bits_weights_slice[i], i))
    message_indices = ranked_idx_desc[:k]
    parity_indices  = sorted(
        set(range(n)) - set(message_indices),
        key=lambda i: (original_bits_weights_slice[i], i)
    )
    # print('len(parity_indices)1',len(parity_indices))
    # assert len(parity_indices) == n - k

    if(solver=="cplex"):
        # Solve MIP on full vector with index mapping
        # mutated_message_bits, objective_change, status = solve_mod2_mip_weighted_per_spans(
        #     encoding_parity_matrix=encoding_parity_matrix,
        #     original_message_bits=original_message_bits_slice,
        #     original_bits_weights=original_bits_weights_slice,
        #     message_indices=message_indices,
        #     parity_indices=parity_indices,
        #     number_spans=original_nums_spans,
        #     time_limit_sec=60,
        #     use_cplex=True,
        #     warm_start=warm_start
        # )
        pass

    elif(solver=="cpsat"):
        mutated_message_bits, objective_change, status = solve_parity_fit_cpsat(
            encoding_parity_matrix=encoding_parity_matrix,
            original_message_bits=original_message_bits_slice,
            original_bits_weights=original_bits_weights_slice,
            message_indices=message_indices,
            parity_indices=parity_indices,
            number_spans=original_nums_spans,
            time_limit_sec=450,
            warm_start=warm_start,
            ExternalParityBits=ExternalParityBits
        )
    # print('len(parity_indices)2',len(parity_indices))
    
    

    # if mutated_message_bits is None:
    #     raise RuntimeError(f"MIP failed with status '{status}'")

    mutated = list(map(int, mutated_message_bits))

    # ---- Parity verification with mapping ----
    # xm = bits at message_indices (rank order); xp = bits at parity_indices (ascending weight order)
    GF2 = galois.GF(2)
    xm_vec = GF2(np.array([mutated[i] for i in message_indices], dtype=int))
    xp_vec = [mutated[i] for i in parity_indices]
    xp_calc = (xm_vec @ P_from_G)        # GF(2) vector of length n-k
    xp_calc_Internal = xp_calc.tolist()[:-ExternalParityBits] if ExternalParityBits != 0 else xp_calc.tolist()
    xp_vec_Internal = xp_vec[:-ExternalParityBits] if ExternalParityBits != 0 else xp_vec
    # print(len(xp_calc), len(xp_calc_Internal), len(xp_vec_Internal))   #24 23 24
    assert xp_calc_Internal == xp_vec_Internal, \
        f"Parity mismatch under mapping.\nexpected xp={xp_calc_Internal}\n   got xp={xp_vec_Internal}"

    # ---- Optional error injection test on the canonical codeword [xm|xp] ----
    if errors:
        if not isinstance(errors, int) or errors < 0:
            raise ValueError("errors must be a non-negative integer.")
        # Build canonical codeword [xm|xp] in systematic order expected by BCH
        cw = [int(v) for v in xm_vec.tolist()] + [int(v) for v in xp_vec]
        if errors > 0:
            # inject exactly `errors` flips at random distinct positions
            flip_idx = sorted(random.sample(range(n), k=min(errors, n)))
            noisy = cw[:]
            for idx in flip_idx:
                noisy[idx] ^= 1
            # Decode back to a valid codeword (unique bounded-distance decoder)
            decoded = bch.decode(GF2(np.array(noisy, dtype=int)), output="codeword")
            decoded_bits = np.array(decoded, dtype=int).tolist()
            # We expect decoding to match the original clean codeword
            assert decoded_bits == cw, \
                f"Error correction failed.\nclean={cw}\nnoisy={noisy}\ndecoded={decoded_bits}"

    # ---- Recompute per-number partials from mutated bits and original spans ----
    updated_nums = []
    for rec in original_nums_spans:
        idx = rec.get("index")
        s   = rec.get("start")
        e   = rec.get("end")
        if s is None or e is None or s >= e:
            updated_nums.append({"index": idx, "value": 0, "start": s, "end": e})
            continue
        val = 0
        # numeric value = Σ mutated[i] * 2^(weight[i]) over i in [s, e)
        for i in range(s, e):
            if mutated[i]:
                val += (1 << original_bits_weights_slice[i])
        updated_nums.append({
            "index": idx,
            "value": val,
            "start": s,
            "end": e
        })

    return {
        "sliced_message_bits": mutated,
        "sliced_bit_weights": original_bits_weights_slice,  # unchanged
        "message_indices": message_indices,
        "parity_indices": parity_indices,
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
#     mutated_chunk = OptimizedParityFittingWeightsEncodeAndDecode(
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