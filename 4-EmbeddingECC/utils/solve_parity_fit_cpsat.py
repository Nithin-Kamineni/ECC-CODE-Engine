import random
import galois
import numpy as np
import pulp as pl
import time
import math
from collections.abc import Iterable
from typing import Dict, Any, List, Tuple
from collections import Counter
import os
from typing import List, Dict, Optional, Tuple
from ortools.sat.python import cp_model

threads = int(os.environ.get("SLURM_CPUS_PER_TASK", "16"))

def solve_parity_fit_cpsat(
    encoding_parity_matrix: List[List[int]],   # A: m x k (entries 0/1)
    original_message_bits: List[int],          # length n = k + m (0/1)
    original_bits_weights: List[int],          # per-bit exponents p -> weight = 1<<p (int)
    message_indices: List[int],                # length k; indices in [0..n-1]
    parity_indices: List[int],                 # length m; indices in [0..n-1]
    number_spans: List[Dict[str, int]],        # spans: [{"start":s, "end":e}, ...]
    time_limit_sec: Optional[float] = 20,
    warm_start: Optional[List[int]] = None,    # optional 0/1 vector length n
    ExternalParityBits: int = 0,               # NEW: number of extra free parity bits
) -> Tuple[Optional[List[int]], Optional[int], str]:
    """
    Minimize sum_t |S_t| subject to:
      - For each span t: S_t = sum_{i in span_t} w_i * (orig_i - y_i)
      - XOR parity rows: XOR_j (A[r][j] * y[msg_j]) == y[parity_idx[r]]

    Returns: (mutated_bits list[int] length n, objective_value (int), solver_status)
             or (None, None, status) if no solution.
    """
    # ---------- Validate & normalize ----------
    # orig = [int(b) & 1 for b in original_message_bits]
    orig = original_message_bits
    n = len(orig)
    k = len(message_indices)
    m = len(parity_indices)
    
    # spans = []
    # for t, rec in enumerate(number_spans):
    #     if not isinstance(rec, dict) or "start" not in rec or "end" not in rec:
    #         raise ValueError(f"Span {t} must have 'start' and 'end'.")
    #     s, e = int(rec["start"]), int(rec["end"])
    #     if not (0 <= s <= e <= n):
    #         raise ValueError(f"Span {t} out of bounds: [{s}, {e})")
    #     spans.append((s, e))

    A2 = encoding_parity_matrix
    # if len(A) != m:
    #     raise ValueError(f"Parity matrix row count must be m={m}, got {len(A)}.")
    # for r in range(m):
    #     if len(A[r]) != k:
    #         raise ValueError(f"Row {r} of A must have length k={k}, got {len(A[r])}.")
    # A2 = [[int(a) & 1 for a in row] for row in A]

    # Integer weights: w[i] = 2^(original_bits_weights[i])
    # (If you risk overflow, downscale exponents or cap with max_scale_bits.)
    w = [int(1) << int(p) for p in original_bits_weights]

    # NEW: add ExternalParityBits at the tail, with zero weight and initial value 0
    #      and extend parity_indices accordingly (same behavior as your CPLEX code).
    for _ in range(ExternalParityBits):
        parity_indices.append(len(orig))  # new parity position at the end
        orig.append(0)                    # initial bit value 0
        w.append(0)                       # free to flip (no cost in S_t)

    n = len(orig)
    k = len(message_indices)
    m = len(parity_indices)

    # Quick overflow guard per span (CP-SAT 64-bit domain)
    # for t, rec in enumerate(number_spans):
    #     s, e = int(rec["start"]), int(rec["end"])
    #     span_sum = sum(w[i] for i in range(s, e))
    #     if span_sum.bit_length() > max_scale_bits:
    #         raise ValueError(
    #             f"Span {t} weight sum 2^{span_sum.bit_length()-1} exceeds guard "
    #             f"(>2^{max_scale_bits}). Consider scaling weights or shrinking spans."
    #         )

    # ---------- Model ----------
    model = cp_model.CpModel()

    # Bits (BoolVars) for all positions
    y = [model.NewBoolVar(f"y[{i}]") for i in range(n)]

    # Create a fixed-True literal to flip parity when using AddBoolXOr
    b_true = model.NewBoolVar("b_true")
    model.Add(b_true == 1)

    # Parity: XOR_j y[msg_j] == y[parity_idx]
    # Implemented as: XOR( y[msg_j] ∪ { y[parity_idx], b_true } ) == True
    # Because including b_true (True) flips parity, enforcing even parity on (msg_j ∪ {y_parity})
    for r in range(m):
        msg_lits = [y[message_indices[j]] for j in range(k) if A2[r][j] == 1]
        par_lit = y[parity_indices[r]]
        model.AddBoolXOr(msg_lits + [par_lit, b_true])  # enforces XOR == True -> even parity on msg_lits ⊕ par_lit

    # Build |S_t| via AddAbsEquality
    abs_list = []
    for t, rec in enumerate(number_spans):
        s, e = int(rec["start"]), int(rec["end"])
        # S_t = sum w[i]*(orig[i] - y[i]) = const - sum w[i]*y[i]
        const_term = sum(w[i] * orig[i] for i in range(s, e))
        # Domain bounds for S_t: [-sum_w, +sum_w]
        span_wsum = sum(w[i] for i in range(s, e))
        S_t = model.NewIntVar(-span_wsum, span_wsum, f"S[{t}]")

        # Build linear expression for S_t
        # S_t + sum w[i]*y[i] == const_term
        model.Add(S_t + sum(w[i] * y[i] for i in range(s, e)) == const_term)

        # A_t = |S_t|
        A_t = model.NewIntVar(0, span_wsum, f"AbsS[{t}]")
        model.AddAbsEquality(A_t, S_t)
        abs_list.append(A_t)

    # Objective: minimize sum_t |S_t|
    model.Minimize(sum(abs_list))

    # ---- Decision strategy: least-significant (lowest weight) variables first ----
    # Sort indices by increasing weight; then tell CP-SAT to pick vars in that order.
    # idx_by_weight = sorted(range(n), key=lambda i: w[i])
    # model.AddDecisionStrategy([y[i] for i in idx_by_weight],
    #                           cp_model.CHOOSE_FIRST,
    #                           cp_model.SELECT_MIN_VALUE)  # try 0 before 1

    # Optional warm start (hints)
    if warm_start is not None:
        if len(warm_start) != n:
            raise ValueError(f"warm_start length {len(warm_start)} != n {n}")
        for i, v in enumerate(warm_start):
            model.AddHint(y[i], int(v) & 1)

    # ---------- Solve ----------
    solver = cp_model.CpSolver()
    if time_limit_sec is not None and time_limit_sec > 0:
        solver.parameters.max_time_in_seconds = float(time_limit_sec)
    solver.parameters.num_search_workers = 0  # 0 = all cores

    solver.parameters.cp_model_presolve = True  # default
    # solver.parameters.search_branching = cp_model.AUTOMATIC
    # solver.parameters.search_branching = cp_model.FIXED_SEARCH
    # solver.parameters.search_branching = cp_model.PORTFOLIO_SEARCH
    solver.parameters.symmetry_level = 1
    solver.parameters.linearization_level = 2
    solver.parameters.relative_gap_limit = float(0.05)
    # solver.parameters.absolute_gap_limit = float(5)

    status = solver.Solve(model)
    status_name = solver.StatusName(status)

    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        mutated = [int(solver.Value(v)) for v in y]
        obj_val = int(solver.ObjectiveValue())
        return mutated, obj_val, status_name

    return None, None, status_name