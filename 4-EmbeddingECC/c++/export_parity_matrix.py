#!/usr/bin/env python3
"""Export BCH parity matrix P (k × r) as .npy for a given (n, t).

Used by 4-EmbeddingECC/run.sh to generate a bit-exact Python BCH parity matrix
that the C++ ecc_embed_cpp binary loads at runtime.  This ensures Python and C++
use the same generator matrix (galois.BCH standard), eliminating the main source
of accuracy divergence between the two implementations.

Usage:
    python3 export_parity_matrix.py --n 63 --t 2 --output /tmp/bch_P_63_t2.npy
"""
import argparse
import sys
import numpy as np
import galois  # pip install galois

# Mirrors bch_message_size() table in bch.h
NANDT_TO_K = {
    (63, 1): 57, (63, 2): 51, (63, 3): 45, (63, 4): 39,
    (63, 5): 36, (63, 6): 30, (63, 7): 24, (63, 8): 18,
    (127,  1): 120, (127,  2): 113, (127,  3): 106, (127,  4):  99,
    (127,  5):  92, (127,  6):  85, (127,  7):  78, (127,  8):  71,
    (127,  9):  71, (127, 10):  64, (127, 11):  57, (127, 12):  50,
    (127, 13):  50,
    (255,  4): 223, (255,  8): 191, (255,  9): 187, (255, 10): 179,
    (255, 11): 171, (255, 12): 163, (255, 13): 155, (255, 14): 147,
    (255, 15): 139, (255, 16): 131, (255, 18): 131, (255, 25):  91,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n',      type=int, required=True,
                        help='Codeword length (63, 127, or 255)')
    parser.add_argument('--t',      type=int, required=True,
                        help='Error correction capability')
    parser.add_argument('--output', required=True,
                        help='Output .npy file path')
    args = parser.parse_args()

    key = (args.n, args.t)
    if key not in NANDT_TO_K:
        sys.exit(
            f"ERROR: Unsupported (n={args.n}, t={args.t}).  "
            f"Valid keys: {sorted(NANDT_TO_K.keys())}"
        )
    k = NANDT_TO_K[key]
    r = args.n - k

    print(
        f"[export_parity_matrix] BCH(n={args.n}, k={k}, t={args.t}) — "
        f"building via galois.BCH ...",
        flush=True,
    )
    bch = galois.BCH(args.n, k)
    # G is the (k × n) systematic generator: G = [I_k | P]
    # We extract the (k × r) parity sub-matrix P = G[:, k:]
    P = np.array(bch.G[:, bch.k:], dtype=np.int8)

    if P.shape != (k, r):
        sys.exit(
            f"ERROR: unexpected P shape {P.shape}, expected ({k}, {r})"
        )

    print(f"[export_parity_matrix] P shape: {P.shape}  (k={k}, r={r})")
    np.save(args.output, P)
    print(f"[export_parity_matrix] Saved to {args.output}")


if __name__ == '__main__':
    main()
