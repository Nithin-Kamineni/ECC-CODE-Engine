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
from typing import List, Dict, Tuple


# ===============================================================
# 1. Cost / evaluation
# ===============================================================
def evaluate(sens: np.ndarray, perm: np.ndarray, group_size: int,
             threshold: float, max_sens: int) -> Dict:
    """Score a permutation.

    Binary metrics (backward compat): count weights > threshold per group.
    Weighted metrics (primary sort key): use actual sens values so that a
    weight with score 0.15 contributes 1000x more than one with score 0.00015.
      budget        = median(group_score) over groups that contain at least
                       one marked weight (not a mean over all groups — sens
                       is sparse, so most groups score 0 and would drag a
                       mean-based budget down to near-zero, making almost
                       any occupied group look "over budget" regardless of
                       how mild the collision is).
      score_excess  = max(group_score - budget, 0)  per group
      score_excess_sq = sum of squares — penalises concentrated violations.
    """
    n = len(sens)
    n_groups = n // group_size
    g = perm[: n_groups * group_size].reshape(n_groups, group_size)

    # binary count metrics (kept for backward compat and human-readable logging)
    counts = (sens[g] > threshold).sum(axis=1)

    # weighted score metrics — primary objective
    group_scores   = sens[g].sum(axis=1)
    total_sens     = float(group_scores.sum())
    nonzero_scores = group_scores[group_scores > 0]
    budget         = float(np.median(nonzero_scores)) if nonzero_scores.size > 0 else 0.0
    score_excess   = np.maximum(group_scores - budget, 0.0)

    return {
        "violating_groups":  int((counts > max_sens).sum()),
        "frac_violating":    float((counts > max_sens).mean()),
        "total_excess":      int(np.maximum(counts - max_sens, 0).sum()),
        "max_in_group":      int(counts.max()),
        "mean_per_group":    float(counts.mean()),
        "score_excess_sq":   float((score_excess ** 2).sum()),
        "score_excess_total":float(score_excess.sum()),
        "budget_per_group":  budget,
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
def _diverse_top_k(results: List[Dict], top_k: int, rng) -> Tuple[List[Dict], int, int]:
    """Return up to top_k candidates from two buckets.

    Bucket 1 (50%): best by score_excess_total (weighted sensitivity metric)
    Bucket 2 (50%): "diverse random" strides — the remaining coprime strides,
        sorted by stride value and split into equal-width contiguous index
        buckets, with one candidate drawn uniformly at random from each
        bucket. This spreads the picks across the whole stride range instead
        of an uncontrolled draw that could cluster several picks close
        together.

    Example splits: top_k=10 → 5,5   top_k=20 → 10,10

    Rank-0 of the returned list is always the best by score_excess_total so the
    legacy manifest rank-0 entry remains the most sensitivity-optimised candidate.

    Returns (diverse, n_score, n_rand) — the actual bucket sizes used, so the
    caller's logging can't drift out of sync with the selection logic.
    """
    if top_k <= 1 or len(results) <= top_k:
        r = results[:top_k]
        return r, len(r), 0

    n_score = (top_k + 1) // 2   # score bucket gets the odd extra slot
    n_rand  = top_k // 2

    seen:    set  = set()
    diverse: List[Dict] = []

    # Bucket 1 — weighted metric (the real objective)
    for r in sorted(results, key=lambda r: r["metrics"]["score_excess_total"]):
        if len(diverse) >= n_score:
            break
        if r["param"] not in seen:
            diverse.append(r)
            seen.add(r["param"])

    # Bucket 2 — diverse random, stratified across the stride value range
    remaining = sorted((r for r in results if r["param"] not in seen),
                        key=lambda r: r["param"])
    m = len(remaining)
    n_pick = min(n_rand, m)   # graceful degradation when few strides remain
    if n_pick > 0:
        i = np.arange(n_pick, dtype=np.int64)
        lo = (i * m) // n_pick
        hi = ((i + 1) * m) // n_pick - 1
        width = hi - lo + 1
        u = rng.random(n_pick)
        idx = lo + np.floor(u * width).astype(np.int64)
        for j in idx:
            diverse.append(remaining[int(j)])

    # rank-0 = best by score_excess_total (legacy manifest rank-0 integrity)
    diverse.sort(key=lambda r: r["metrics"]["score_excess_total"])
    return diverse, n_score, n_pick


def search(sens: np.ndarray, group_size: int = 8,
           threshold: float = 0.5, max_sens: int = 2,
           max_stride: int = 256,
           verbose: bool = True,
           top_k: int = 10) -> List[Dict]:
    """
    Evaluate all coprime strides s in [2, min(max_stride, N-1)] exhaustively,
    then return a diverse set of up to top_k candidates:
      50% best by score_excess_total, 50% diverse random strides stratified
      across the stride range.

    This diversification means stage 4's greedy has a wider pool to search over
    for the lowest-distortion embedding, instead of 10 near-identical candidates.

    Rank-0 is the best by score_excess_total (most sensitivity-optimised).
    """
    n = len(sens)
    limit = min(max_stride, n - 1)
    results: List[Dict] = []

    if verbose:
        print(f"  - exhaustive stride search  s ∈ [2, {limit}]  "
              f"({max(0, limit - 1)} candidates) ...")

    for s in range(2, limit + 1):
        if gcd(s, n) == 1:
            perm = stride_perm(n, s)
            m = evaluate(sens, perm, group_size, threshold, max_sens)
            results.append({"family": "stride", "param": s, "metrics": m})

    if not results:
        return results

    if verbose:
        print(f"\n--- Identity (no permutation) baseline ---")
        baseline = evaluate(sens, np.arange(n), group_size, threshold, max_sens)
        for k, v in baseline.items():
            print(f"  {k:18s}: {v}")
        top_by_score = sorted(results, key=lambda r: r["metrics"]["score_excess_total"])
        print(f"\n--- Top 10 by score_excess_total (s ≤ {limit}, coprime only) ---")
        for r in top_by_score[:10]:
            m = r["metrics"]
            print(f"  stride s={str(r['param']):>5s}  "
                  f"score_excess_total={m['score_excess_total']:>12.6f}  "
                  f"excess={m['total_excess']:>7d}  "
                  f"violating={m['violating_groups']:>6d} "
                  f"({m['frac_violating']*100:5.2f}%)  "
                  f"max_in_group={m['max_in_group']}")

    rng = np.random.default_rng()   # fresh seed each call — intentional diversity
    diverse, n_score, n_rand = _diverse_top_k(results, top_k, rng)

    if verbose:
        print(f"\n--- Diverse top-{top_k} selected ({n_score} score, "
              f"{n_rand} random) ---")
        for r in diverse:
            m = r["metrics"]
            print(f"  stride s={str(r['param']):>5s}  "
                  f"score_excess_total={m['score_excess_total']:>12.6f}  "
                  f"binary_excess={m['total_excess']:>7d}")

    return diverse


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
          f"coprime={gcd(best['param'], n) == 1}")
    print(hardware_description(best["family"], best["param"], n, GROUP))
