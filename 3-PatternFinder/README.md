# 3-PatternFinder

Finds a weight permutation for each model layer so that, when weights are read in
groups of `GROUP_SIZE`, no group contains more than `MAX_SENS` high-sensitivity
weights. This is required before ECC embedding (stage 4) so that a BCH code with
correction capacity `t` can actually protect every codeword.

---

## Files

| File | Role |
|---|---|
| `prepare_patterns.py` | Main driver. Reads the Taylor sensitivity CSV, builds a dense `sens` array per layer, runs the selected search mode, and writes all output files. |
| `find_pattern.py` | Pure search library. Implements the stride-permutation scoring (`evaluate`) and exhaustive search (`search`). Imported by `prepare_patterns.py`. |
| `run.sh` | SLURM/bash wrapper. Loops over `DATASETS × ARCHS × QUANT_LEVELS` and calls `prepare_patterns.py` via Singularity. Controls the search mode via env vars. |

---

## Search Modes

The mode is selected by env vars read in `run.sh` before calling `prepare_patterns.py`.

| Env var | Value | Mode | `prepare_patterns.py` flag |
|---|---|---|---|
| `DISABLE_PATTERN_FIND` | `true` | Contiguous (identity) | `--identity-perm` |
| `RANDOM_PATTERN_FIND` | `true` | Random coprime stride | `--random-stride` |
| both false (default) | — | Exhaustive search | `--run-search` |

---

## What Happens Per Layer (all three modes)

For every layer listed in the sensitivity CSV, `prepare_patterns.py` does this
regardless of mode:

1. **Read sensitivity CSV** — extract `flat_idx` + `taylor` score for all
   selected weights in this layer.
2. **Get true layer size N** — from the torchvision model (or `--shapes-json`).
   The CSV only contains the top-selected weights, so `max(flat_idx)` would
   under-count the layer; the real `N = numel(layer)` is needed because the
   permutation must cover the full `[0, N-1]` index space.
3. **Build dense `sens[0..N-1]`** — `sens[selected] = 1.0`, rest `= 0.0`
   (indicator mode). Any weight with `sens > threshold` (default 0.5) is
   considered sensitive.
4. **Run the selected search mode** (see below).
5. **Save output files** for rank 0 (and optionally top-K ranks for `--run-search`):
   - `{layer}_rank0_perm.npy` — permutation index array (`perm[k]` = original
     flat index of the k-th element in permuted order)
   - `{layer}_rank0_inv_perm.npy` — inverse permutation (`inv_perm[orig] = permuted pos`)
   - `{layer}_rank0_weights_perm.npy` — original weights reordered by `perm`
     (actual int8/int16 values from the quantized checkpoint)
   - `{layer}_best_pattern.txt` — human-readable metrics
6. **Accumulate histogram** — counts how many codewords (groups of `GROUP_SIZE`)
   have exactly 0, 1, 2, ... `GROUP_SIZE` sensitive weights. Saved at the end as
   `codeword_histogram.json`.
7. **Write manifests** after all layers:
   - `pattern_manifest.json` — rank-0 entry per layer (backward compat)
   - `top_patterns_manifest.json` — all K candidates per layer (used by stage 4)
   - `pattern_search_summary.csv` — tabular summary across layers

---

## The Three Search Modes in Detail

### Contiguous (`--identity-perm`)

No search. Assigns `perm[k] = k` — weights stay in their original storage order.
`inv_perm` is also identity. Weights are saved in original order.
Used as the baseline in the Pattern-study ablation.

### Random (`--random-stride`)

No search. Per layer:
1. Collect all strides `s ∈ [2, MAX_STRIDE]` where `gcd(s, N) == 1` (coprime).
2. Pick **one** at random — fresh `numpy.random.default_rng()` per layer, no seed,
   so results differ on every run.
3. Apply `perm[k] = (k × s) mod N`.
4. Compute inverse: `inv_perm[perm[k]] = k`.

Falls back to identity if no coprime strides exist for a layer (rare; only happens
when N is very small).

### Strided / Exhaustive Search (`--run-search`)

Finds the best coprime stride for each layer. Per layer:

```
[1] Build sens array (as above)
        │
        ▼
[2] Generate all candidate strides s ∈ [2, min(MAX_STRIDE, N-1)]
      Keep only coprime strides: gcd(s, N) == 1
      (Non-coprime strides are non-bijective — some weights are visited
       multiple times and others never, which corrupts ECC groups and
       makes the evaluation score artificially low.)
        │
        ▼
[3] Score each candidate: perm[k] = (k × s) mod N
      Divide permuted weights into groups of GROUP_SIZE.
      For each group, count sensitive weights (> threshold).
      Metrics computed by find_pattern.evaluate():
        • total_excess       = Σ max(count − MAX_SENS, 0)   ← primary sort key
        • violating_groups   = # groups exceeding MAX_SENS
        • max_in_group       = worst-case group count
        │
        ▼
[4] Sort by (total_excess, max_in_group) ascending
      rank 0 = best stride for this layer
        │
        ▼
[5] Save top-K candidates (K = TOP_PATTERNS env var)
      Ranks 0..K-1 each get perm, inv_perm, weights_perm files.
      Stage 4 (EmbeddingECC) reads top_patterns_manifest.json and
      selects among them using its own approach (e.g. search3).
        │
        ▼
[6] Histogram for rank-0 pattern → codeword_histogram.json
```

A hard `RuntimeError` guard (prepare_patterns.py line ~440) aborts if a
non-bijective permutation slips through, protecting the weights_perm and
inv_perm files from silent corruption.

---

## Key Invariant (all modes)

```
weights_perm  = weights_orig[perm]      # permute: original → hardware order
weights_orig  = weights_perm[inv_perm]  # invert:  hardware → original order
perm[inv_perm[i]] == i  for all i       # round-trip identity
```

Stage 5 (EmbeddingsMerging) reads `inv_perm_file` from the manifest and uses it
to restore the corrected weights back to their original positions. No changes to
stage 5 are needed regardless of which search mode was used — it is permutation-
strategy-agnostic.

---

## Output Directory Structure

```
0-Data/artifacts/patterns/
  {dataset}/
    {arch}/
      {PTQ|QAT}/
        {float32|8-bit|4-bit}/
          {layer}_rank0_perm.npy
          {layer}_rank0_inv_perm.npy
          {layer}_rank0_weights_perm.npy
          {layer}_rank1_perm.npy          ← only for --run-search with top-k > 1
          ...
          {layer}_best_pattern.txt
          pattern_manifest.json
          top_patterns_manifest.json
          pattern_search_summary.csv
          codeword_histogram.json         ← new: histogram across all layers
```

---

## Key Parameters (set via env vars in env.sh / run.sh)

| Param | Env var | Meaning |
|---|---|---|
| `GROUP_SIZE` | `GROUP_SIZE` | Weights per ECC codeword (default 8) |
| `MAX_SENS` | `MAX_SENS` | Max sensitive weights allowed per group (e.g. 2) |
| `MAX_STRIDE` | `MAX_STRIDE` | Upper bound on stride search space (default 256) |
| `TOP_PATTERNS` | `TOP_PATTERNS` | How many top-K candidates to save (for `--run-search`) |
| `SENS_THRESHOLD` | `SENS_THRESHOLD` | Score cutoff for "sensitive" (default 0.5) |
| `TOP_SENSITIVE` | `TOP_SENSITIVE` | Minimum sensitive weights to mark per layer |
| `DISABLE_PATTERN_FIND` | `DISABLE_PATTERN_FIND` | `true` → identity mode |
| `RANDOM_PATTERN_FIND` | `RANDOM_PATTERN_FIND` | `true` → random stride mode |

---

## Relationship to Adjacent Stages

- **Reads from stage 2** (`2-Sensitivity`): `layer_then_weight_*.csv` — per-weight
  Taylor sensitivity scores.
- **Writes to stage 4** (`4-EmbeddingECC`): `top_patterns_manifest.json` with the
  top-K stride candidates per layer.
- **Stage 5** (`5-EmbeddingsMerging`) uses `inv_perm_file` from `pattern_manifest.json`
  to restore corrected weights to original positions. Works identically for all
  three search modes.
