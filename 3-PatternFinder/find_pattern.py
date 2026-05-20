"""
find_pattern.py
---------------
Find a hardware-friendly permutation of N weights so that, when the hardware
reads consecutive groups of GROUP positions, each group contains at most
MAX_SENS high-sensitivity weights.

CONSTRAINTS
- The permutation must be expressible as a simple regular rule (so it can be
  implemented in hardware without a 1M-entry lookup table).
- The original sensitivity ORDERING is preserved (we only change the *order*
  in which the hardware visits indices, not the underlying array).

PERMUTATION FAMILIES SEARCHED
  1. Stride:            perm[k] = (k * s) mod N,    gcd(s, N) = 1
  2. Block interleaver: perm[k] = (k mod R) * C + (k div R),  R*C = N
  3. Bit-reversal:      only when N is a power of 2

These are all standard interleavers used in ECC hardware (CD-ROM, DVB, LTE, FFT).
"""

import numpy as np
from math import gcd, sqrt
from typing import Optional, List, Dict


# ===============================================================
# 1. Cost / evaluation
# ===============================================================
def evaluate(sens: np.ndarray, perm: np.ndarray, group_size: int,
             threshold: float, max_sens: int) -> Dict:
    """Score a permutation; lower 'total_excess' is better."""
    n = len(sens)
    n_groups = n // group_size
    g = perm[: n_groups * group_size].reshape(n_groups, group_size)
    counts = (sens[g] > threshold).sum(axis=1)
    return {
        "violating_groups": int((counts > max_sens).sum()),
        "frac_violating":   float((counts > max_sens).mean()),
        "total_excess":     int(np.maximum(counts - max_sens, 0).sum()),
        "max_in_group":     int(counts.max()),
        "mean_per_group":   float(counts.mean()),
    }


# ===============================================================
# 2. Pattern generators (each is a bijection of [0..N-1])
# ===============================================================
def stride_perm(n: int, s: int) -> Optional[np.ndarray]:
    """perm[k] = (k * s) mod n.  Bijection iff gcd(s, n) = 1."""
    if gcd(s, n) != 1:
        return None
    return (np.arange(n, dtype=np.int64) * s) % n


def block_interleaver(n: int, rows: int) -> Optional[np.ndarray]:
    """
    Classical block interleaver: write n elements row-wise into a rows x cols
    matrix, then read out column-wise.
        perm[k] = (k mod rows) * cols + (k div rows)
    """
    if n % rows != 0:
        return None
    cols = n // rows
    k = np.arange(n, dtype=np.int64)
    return (k % rows) * cols + (k // rows)


def bit_reversal_perm(n: int) -> Optional[np.ndarray]:
    """Bit-reversal permutation (only if n is a power of 2)."""
    if n & (n - 1) != 0:
        return None
    bits = n.bit_length() - 1
    k = np.arange(n, dtype=np.int64)
    out = np.zeros_like(k)
    for b in range(bits):
        out |= ((k >> b) & 1) << (bits - 1 - b)
    return out


# ===============================================================
# 3. Stride candidate generation
# ===============================================================
def candidate_strides(n: int, n_random: int, seed: int = 0) -> set:
    """
    Diverse set of coprime-to-n stride candidates.

    Includes:
      * The golden-ratio stride round(n/phi). Of all irrational ratios, phi
        gives the most uniform spread (worst rational approximations), which
        makes it disrupt periodic clumping best.
      * Other irrational anchors (sqrt(2), sqrt(3), e, pi, ...).
      * Simple integer ratios n/k for small k.
      * Random fill.
    """
    rng = np.random.default_rng(seed)
    strides = set()
    phi = (1 + sqrt(5)) / 2

    anchors = [phi, phi - 1, sqrt(2), sqrt(3), sqrt(5), sqrt(7),
               np.e - 2, np.pi - 3]
    for a in anchors:
        frac = a - int(a)
        base = int(round(n * frac))
        for delta in range(-30, 31):
            s = base + delta
            if 1 < s < n and gcd(s, n) == 1:
                strides.add(s)

    for k in range(2, 64):
        base = n // k
        for delta in range(-10, 11):
            s = base + delta
            if 1 < s < n and gcd(s, n) == 1:
                strides.add(s)

    while len(strides) < n_random:
        s = int(rng.integers(2, n))
        if gcd(s, n) == 1:
            strides.add(s)

    return strides


# ===============================================================
# 4. Master search
# ===============================================================
def search(sens: np.ndarray, group_size: int = 8,
           threshold: float = 0.5, max_sens: int = 2,
           n_random_strides: int = 400, seed: int = 0,
           verbose: bool = True) -> List[Dict]:
    n = len(sens)
    results: List[Dict] = []

    if verbose:
        print(f"  - searching strides ...")
    for s in candidate_strides(n, n_random_strides, seed):
        perm = stride_perm(n, s)
        m = evaluate(sens, perm, group_size, threshold, max_sens)
        results.append({"family": "stride", "param": s, "metrics": m})

    if verbose:
        print(f"  - searching block interleavers ...")
    for rows in range(2, int(sqrt(n)) + 1):
        if n % rows == 0:
            perm = block_interleaver(n, rows)
            m = evaluate(sens, perm, group_size, threshold, max_sens)
            results.append({"family": "block", "param": rows, "metrics": m})

    perm = bit_reversal_perm(n)
    if perm is not None:
        m = evaluate(sens, perm, group_size, threshold, max_sens)
        results.append({"family": "bit_reversal", "param": None, "metrics": m})

    results.sort(key=lambda r: (r["metrics"]["total_excess"],
                                 r["metrics"]["max_in_group"]))

    # Local search around the current best stride
    if verbose:
        print(f"  - local search around best stride ...")
    best_stride = next((r for r in results if r["family"] == "stride"), None)
    if best_stride is not None:
        s0 = best_stride["param"]
        for delta in range(-300, 301):
            s = s0 + delta
            if delta == 0 or s < 2 or s >= n or gcd(s, n) != 1:
                continue
            perm = stride_perm(n, s)
            m = evaluate(sens, perm, group_size, threshold, max_sens)
            results.append({"family": "stride", "param": s, "metrics": m})
        results.sort(key=lambda r: (r["metrics"]["total_excess"],
                                     r["metrics"]["max_in_group"]))

    if verbose:
        print(f"\n--- Identity (no permutation) baseline ---")
        baseline = evaluate(sens, np.arange(n), group_size, threshold, max_sens)
        for k, v in baseline.items():
            print(f"  {k:18s}: {v}")
        print(f"\n--- Top 10 patterns ---")
        for r in results[:10]:
            m = r["metrics"]
            print(f"  {r['family']:12s} param={str(r['param']):>10s}  "
                  f"excess={m['total_excess']:>7d}  "
                  f"violating={m['violating_groups']:>6d} "
                  f"({m['frac_violating']*100:5.2f}%)  "
                  f"max_in_group={m['max_in_group']}")

    return results


def make_perm(family: str, param, n: int) -> np.ndarray:
    if family == "stride":       return stride_perm(n, param)
    if family == "block":        return block_interleaver(n, param)
    if family == "bit_reversal": return bit_reversal_perm(n)
    raise ValueError(family)


def hardware_description(family: str, param, n: int, group_size: int) -> str:
    if family == "stride":
        s = param
        return (f"Hardware rule: for read step k = 0, 1, 2, ..., "
                f"the weight index is (k * {s}) mod {n}.\n"
                f"Process every {group_size} consecutive reads as one group.")
    if family == "block":
        rows = param
        cols = n // rows
        return (f"Hardware rule: block interleaver with R={rows}, C={cols}.\n"
                f"For read step k, weight index = (k mod {rows}) * {cols} "
                f"+ (k div {rows}).")
    if family == "bit_reversal":
        bits = n.bit_length() - 1
        return (f"Hardware rule: bit-reverse the {bits}-bit read counter k "
                f"to get the weight index.")
    return "n/a"


# ===============================================================
# 5. Demo with synthetic clumpy sensitivities
# ===============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 1_000_000

    # Realistic-ish: ~12% high-sensitivity weights, clustered into "hot" regions.
    # This mimics the situation where consecutive grouping fails badly.
    sens = rng.uniform(0, 0.4, size=n)
    n_high = int(0.12 * n)
    n_clusters = 50
    centers = rng.integers(n // n_clusters, n - n // n_clusters, size=n_clusters)
    width = (n // n_clusters) // 3
    per_cluster = n_high // n_clusters
    for c in centers:
        idx = rng.integers(c - width, c + width, size=per_cluster)
        idx = np.clip(idx, 0, n - 1)
        sens[idx] = rng.uniform(0.7, 1.0, size=per_cluster)

    GROUP, THRESHOLD, MAX_SENS = 8, 0.65, 2

    print(f"N = {n:,}")
    print(f"Fraction high-sensitivity (> {THRESHOLD}): "
          f"{(sens > THRESHOLD).mean()*100:.2f}%")
    print(f"Group size = {GROUP}, allow at most {MAX_SENS} sensitive per group\n")

    results = search(sens, group_size=GROUP, threshold=THRESHOLD,
                     max_sens=MAX_SENS, n_random_strides=400)

    best = results[0]
    perm = make_perm(best["family"], best["param"], n)
    print(f"\n>>> Best pattern: {best['family']}  param={best['param']}")
    print(hardware_description(best["family"], best["param"], n, GROUP))