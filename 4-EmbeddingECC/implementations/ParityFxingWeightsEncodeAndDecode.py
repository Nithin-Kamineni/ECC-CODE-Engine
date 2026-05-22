import numpy as np
import galois
import numpy as np
from math import floor
import random
import time

def solve_mod2_gaussjordan(A, b, bitpriority):
    GF = galois.GF(2)  # GF(2)
    A = np.asarray(A)
    m, n = A.shape

    # --- Build column permutation from bitpriority ---
    if bitpriority is None:
        perm = list(range(n))
    else:
        bitpriority = list(bitpriority)
        # If user passes a full permutation, this just uses it.
        # If not, append any missing indices at the end.
        remaining = [i for i in range(n) if i not in bitpriority]
        perm = bitpriority + remaining
        if len(perm) != n:
            raise ValueError("bitpriority produces invalid permutation")

    # Permute columns according to priority
    A_perm = A[:, perm]

    # RHS
    b = GF(b).reshape(-1, 1)

    # Augmented matrix [A | b] over GF(2)
    Aug = GF(np.hstack([A_perm, b]))
    R = Aug.row_reduce()
    m, n_perm = A_perm.shape

    # Inconsistency check: row [0 ... 0 | 1]
    inconsistent = np.any(np.all(R[:, :n_perm] == 0, axis=1) & (R[:, n_perm] != 0))
    if inconsistent:
        raise ValueError("No solution over GF(2).")

    # Build one particular solution in permuted variable order
    x_perm = GF.Zeros(n_perm)
    row = 0
    for col in range(n_perm):
        # Pivot condition in RREF: leading 1 with zeros to the left
        if row < m and R[row, col] == 1 and np.all(R[row, :col] == 0):
            x_perm[col] = R[row, n_perm]
            row += 1
        # else: this is a free variable, stays 0

    # Un-permute back to original variable indices
    x = GF.Zeros(n)
    for j, orig_idx in enumerate(perm):
        x[orig_idx] = x_perm[j]

    em = x.tolist()  # no [::-1] now

    return em


def MutateWeightsEncodeAndDecode(chunk, message_parity_size=63, message_size=57):

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
    if message_size != n:
        raise ValueError(f"message_size (={message_size}) must equal chunk length (={n}).")
    if any(b not in (0,1) for b in bits):
        raise ValueError("original_message_bits_slice must contain only 0/1.")
    k = message_size

    # --- 1) Rank positions by weight for message selection ---
    # Top-k by weight (desc). Tie-breaker: lower index first (stable).
    message_indices = sorted(range(message_size), key=lambda i: (weights[i], i))

    # --- 2) Build BCH(n,k) with systematic generator (I | P) ---
    bch = galois.BCH(message_parity_size, k)     # requires n = 2^m - 1; e.g., 63, 127, ...
    G = bch.G                  # shape (k, n)
    P_from_G = G[:, bch.k:]    # shape (k, n-k); systematic parity
    encoding_parity_matrix = np.asarray(P_from_G.T)

    # --- 3) Form the message vector in the exact order of message_indices ---
    # NOTE: message order == rank order of the top weights (not original spatial order)
    m0 = bch.field(bits)                 # GF element row vector length k
    p0 = m0 @ P_from_G                    # GF vector length n-k
    originalParity = np.array(p0, dtype=int).tolist()

    # Step 4: Find difference betweem orginal pairty bits and target parity bits
    targetParity = [0 for _ in range(len(originalParity))]
    deltaParity = [targetParity[i] ^ originalParity[i] for i in range(len(originalParity))]

    # print('message_indices',message_indices)
    
    # Step 6: Find em(message changes) with Gauss-Jordan
    em = solve_mod2_gaussjordan(A=encoding_parity_matrix, b=deltaParity, bitpriority=message_indices)
    
    # Step 7: Apply the changes of em to message to create mutated message (m xor em)
    mutated_message_bits = galois.GF2([bits[i]^em[i] for i in range(len(bits))])
    # print("message_bits         =",message_bits)
    # print("mutated_message_bits =",mutated_message_bits.tolist())
    # print()
    
    # Step 8: Generate the parity vector from mutated weight
    flexible_parity = mutated_message_bits @ P_from_G                 
    # print('fitted parity =',flexible_parity)

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
            if mutated_message_bits[i]:
                val += (1 << weights[i])

        updated_nums.append({
            "index": idx,
            "value": val,
            "start": s,
            "end": e
        })


    return {
        "sliced_message_bits": mutated_message_bits.tolist(),
        "sliced_bit_weights": original_bits_weights_slice,
        "message_indices": message_indices,
        "parity_indices": [],
        # "parity_bits": parity_bits,
        "sliced_message_nums": updated_nums
    }