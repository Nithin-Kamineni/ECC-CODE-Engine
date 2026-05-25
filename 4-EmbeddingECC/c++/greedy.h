#pragma once
// greedy.h — C++ port of GreedyEncodeAndDecode.
//
// Hill-climbing: at each step, try all (bucket, move_unit) pairs and accept the
// lexicographically best (score, tie_break) improvement.  Converges when no
// single-bucket move reduces the score.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include "bch.h"
#include "slicing.h"
#include "reconstruct.h"
#include "search3.h"   // reuse Partition, BucketMeta, partition_bits, build_bucket_meta

// ---- Evaluate a given delta vector: returns (score, tie, full_bits, new_vals) ----
struct EvalResult {
    float             score;
    int               tie;
    std::vector<int>  bits;
    std::vector<int64_t> new_vals;
};

static EvalResult greedy_evaluate(
    const Chunk&             ch,
    const Partition&         part,
    const std::vector<BucketMeta>& metas,
    const PMatrix&           P,
    const std::vector<int64_t>& current_deltas,
    int k, int /* n */)
{
    int B = (int)metas.size();

    // Compute new bucket values
    std::vector<int64_t> new_vals(B);
    for (int b = 0; b < B; b++)
        new_vals[b] = metas[b].old_val + current_deltas[b];

    // Build message bits from new bucket values
    std::vector<int> base_bits = ch.bits;
    std::vector<uint64_t> m_bits(P.words_per_col > 0 ? P.words_per_col : 1, 0);

    // Initialise from original bits at message positions
    for (int i = 0; i < k; i++) {
        int pos = part.message_indices[i];
        if (ch.bits[pos]) m_bits[i / 64] |= (1ULL << (i % 64));
    }

    // Apply bucket deltas: update message bits for changed buckets
    for (int b = 0; b < B; b++) {
        if (!metas[b].has_msg || current_deltas[b] == 0) continue;
        for (int pi = 0; pi < (int)metas[b].message_pos.size(); pi++) {
            int pos = metas[b].message_pos[pi];
            int mi  = metas[b].m_indices[pi];
            int bit = (new_vals[b] >> ch.weights[pos]) & 1;
            int w = mi / 64, b2 = mi % 64;
            m_bits[w] = (m_bits[w] & ~(1ULL << b2)) | ((uint64_t)bit << b2);
        }
    }

    // Compute BCH parity
    auto parity = P.compute_parity(m_bits);

    // Assemble full bit vector
    std::vector<int> bits = ch.bits;
    for (int i = 0; i < k; i++)
        bits[part.message_indices[i]] = (m_bits[i / 64] >> (i % 64)) & 1;
    for (int j = 0; j < (int)parity.size(); j++)
        bits[part.parity_indices[j]] = parity[j];

    // L2 score — use FULL bucket value change (message bits + BCH parity).
    // Parity bits also reside in buckets; true distortion includes their contribution.
    // Compare assembled bits[] against ch.buckets[b].value (original full value).
    float score = 0.0f;
    for (int b = 0; b < B; b++) {
        const auto& bk_orig = ch.buckets[b];
        int64_t full_new = 0;
        for (int pos = bk_orig.start; pos < bk_orig.end; pos++) {
            if (bits[pos]) full_new += (1LL << ch.weights[pos]);
        }
        float diff = (float)(full_new - bk_orig.value);
        score += metas[b].sens * diff * diff;
    }

    // Tie-break: sum of absolute value-space deltas.
    // Mirrors Python GreedyEncodeAndDecode.py line 203:
    //   tie_break = int(np.abs(deltas).sum())
    // where deltas are accumulated in value space (delta = move_unit * step).
    // Do NOT divide by step here — that would count move-units, not value deltas.
    int tie = 0;
    for (int b = 0; b < B; b++) tie += (int)std::abs(current_deltas[b]);

    return {score, tie, bits, new_vals};
}

// ---- greedy_encode -----------------------------------------------------------
inline Chunk greedy_encode(
    const Chunk& ch,
    const PMatrix& P,
    int message_parity_size,
    int message_size,
    int move_unit_range = 4,
    const std::vector<float>* sens_weights = nullptr)
{
    int n = message_parity_size;
    int k = message_size;
    assert((int)ch.bits.size() == n);

    Partition part = partition_bits(ch, k);
    std::vector<BucketMeta> metas = build_bucket_meta(ch, part, sens_weights);
    int B = (int)metas.size();

    // Build list of move units: {-range, ..., -1, +1, ..., +range}
    std::vector<int> move_units;
    for (int u = -move_unit_range; u <= move_unit_range; u++)
        if (u != 0) move_units.push_back(u);

    // Initial deltas: all zero
    std::vector<int64_t> current_deltas(B, 0);
    EvalResult best = greedy_evaluate(ch, part, metas, P, current_deltas, k, n);

    while (true) {
        float  move_score = best.score;
        int    move_tie   = best.tie;
        int    move_b     = -1;
        int64_t move_d    = 0;
        EvalResult move_result;

        for (int b = 0; b < B; b++) {
            if (!metas[b].has_msg) continue;
            int64_t step_b = metas[b].step;
            if (step_b == 0) continue;

            for (int u : move_units) {
                int64_t d = (int64_t)u * step_b;
                int64_t new_bucket_val = metas[b].old_val + current_deltas[b] + d;
                // Range check
                if (new_bucket_val < 0 || new_bucket_val > metas[b].max_val) continue;

                // Trial deltas
                std::vector<int64_t> trial = current_deltas;
                trial[b] += d;

                EvalResult res = greedy_evaluate(ch, part, metas, P, trial, k, n);

                if (res.score < move_score ||
                    (res.score == move_score && res.tie < move_tie)) {
                    move_score  = res.score;
                    move_tie    = res.tie;
                    move_b      = b;
                    move_d      = d;
                    move_result = res;
                }
            }
        }

        if (move_b < 0) break;  // no improvement found → converged

        current_deltas[move_b] += move_d;
        best = move_result;
    }

    // Build output chunk from best result
    Chunk out = ch;
    out.bits = best.bits;
    update_chunk_buckets(out, out.bits);
    return out;
}
