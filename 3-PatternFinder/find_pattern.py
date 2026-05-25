"""
find_pattern.py
---------------
Find a hardware-friendly stride permutation of N weights so that, when the
hardware reads consecutive groups of GROUP positions, each group contains at
most MAX_SENS high-sensitivity weights.

PERMUTATION FAMILY
  Stride only:  perm[k] = (k * s) mod N

  s is constrained to [2, MAX_STRIDE] so that every consecutive read in
  permuted order maps to an original-array position within MAX_STRIDE steps
  of the previous one — keeping accesses inside the hardware's burst-fetch
  tile (cache line / DMA burst).

SEARCH STRATEGY
  Exhaustive enumeration of all s in [2, min(MAX_STRIDE, N-1)].
  With MAX_STRIDE ≤ 256 this is at most 254 candidates — negligible cost.
  Coprime strides (gcd(s,N)=1, true bijections) are evaluated first.
  Non-coprime strides are evaluated as a fallback when no coprime stride
  achieves a lower cost within the constraint.
"""

import numpy as np
from math import gcd
from typing import List, Dict


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
# 2. Stride permutation
# ===============================================================
def stride_perm(n: int, s: int) -> np.ndarray:
    """perm[k] = (k * s) mod n.
    When gcd(s, n) == 1 this is a bijection (all N weights visited once).
    When gcd(s, n) > 1 it is a partial cycle (fallback; some weights skipped).
    """
    return (np.arange(n, dtype=np.int64) * s) % n


# ===============================================================
# 3. Master search — exhaustive over bounded stride range
# ===============================================================
def search(sens: np.ndarray, group_size: int = 8,
           threshold: float = 0.5, max_sens: int = 2,
           max_stride: int = 256,
           verbose: bool = True) -> List[Dict]:
    """
    Evaluate all strides s in [2, min(max_stride, N-1)] exhaustively.

    Coprime strides (true bijections) are evaluated first so they win ties.
    Non-coprime strides are appended as fallback — they only win if no
    coprime stride within the range achieves a lower total_excess.

    Returns results sorted by (total_excess, max_in_group) ascending.
    """
    n = len(sens)
    limit = min(max_stride, n - 1)
    results: List[Dict] = []

    if verbose:
        print(f"  - exhaustive stride search  s ∈ [2, {limit}]  "
              f"({max(0, limit - 1)} candidates) ...")

    # Only coprime strides produce valid bijections and can be safely inverted.
    # Non-coprime strides are excluded: they game the evaluation metric
    # (a non-bijective perm only samples gcd(s,N) distinct positions, so
    # total_excess is artifically low) and corrupt downstream perm/weights files.
    for s in range(2, limit + 1):
        if gcd(s, n) == 1:
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
        print(f"\n--- Top 10 stride patterns (s ≤ {limit}, coprime only) ---")
        for r in results[:10]:
            m = r["metrics"]
            print(f"  stride s={str(r['param']):>5s}  "
                  f"excess={m['total_excess']:>7d}  "
                  f"violating={m['violating_groups']:>6d} "
                  f"({m['frac_violating']*100:5.2f}%)  "
                  f"max_in_group={m['max_in_group']}")

    return results


def make_perm(family: str, param, n: int) -> np.ndarray:
    if family == "stride":
        return stride_perm(n, param)
    raise ValueError(f"Unknown permutation family: {family!r}")


def hardware_description(family: str, param, n: int, group_size: int) -> str:
    if family == "stride":
        s = param
        return (f"Hardware rule: for read step k = 0, 1, 2, ..., "
                f"the weight index is (k * {s}) mod {n}.\n"
                f"Stride s={s} ≤ MAX_STRIDE — all accesses within the burst-fetch tile.\n"
                f"Process every {group_size} consecutive reads as one ECC group.")
    return "n/a"


# ===============================================================
# Demo / self-test
# ===============================================================
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    n = 9_408    # conv1 weight of ResNet-18 (64 × 3 × 7 × 7)

    # ~10% high-sensitivity weights clustered into hot regions
    sens = rng.uniform(0, 0.4, size=n)
    n_high = int(0.10 * n)
    centers = rng.integers(n // 20, n - n // 20, size=20)
    for c in centers:
        idx = rng.integers(max(0, c - 50), min(n, c + 50), size=n_high // 20)
        sens[idx] = rng.uniform(0.7, 1.0, size=idx.shape)

    MAX_STRIDE = 256
    GROUP, THRESHOLD, MAX_SENS = 8, 0.5, 2

    print(f"N = {n:,}  MAX_STRIDE = {MAX_STRIDE}")
    print(f"Fraction high-sensitivity (> {THRESHOLD}): "
          f"{(sens > THRESHOLD).mean()*100:.2f}%")
    print(f"Group size = {GROUP}, allow at most {MAX_SENS} sensitive per group\n")

    results = search(sens, group_size=GROUP, threshold=THRESHOLD,
                     max_sens=MAX_SENS, max_stride=MAX_STRIDE)

    best = results[0]
    perm = make_perm(best["family"], best["param"], n)
    print(f"\n>>> Best pattern: {best['family']}  param={best['param']}  "
          f"coprime={best['coprime']}")
    print(hardware_description(best["family"], best["param"], n, GROUP))
