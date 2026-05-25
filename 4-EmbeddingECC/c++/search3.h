#pragma once
// search3.h — C++ port of Search3EncodeAndDecode.
//
// Exhaustive 3^B search over bucket deltas {-1, 0, +1}.
// Selects the combination that minimises L2 distortion while satisfying BCH parity.

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include "bch.h"
#include "slicing.h"
#include "reconstruct.h"

// ---- Bucket metadata (mirrors Python's bucket_metadata list) ---- ----
struct BucketMeta {
    int             b_idx;           // bucket index
    std::vector<int> message_pos;    // positions in ch.bits that are message bits for this bucket
    std::vector<int> m_indices;      // corresponding indices into the message_indices array
    // old_val = MESSAGE-PARTIAL value: sum of 2^w for message bits that are 1.
    // This intentionally does NOT include parity bit contributions to the bucket.
    // Reason: range checks and bit extraction operate in message-partial space;
    //         mixing in parity contributions (which can be large at high t-values)
    //         causes valid moves to be rejected and wrong bits to be extracted.
    // The ORIGINAL FULL bucket value (for score computation) is in ch.buckets[b].value.
    int64_t          old_val;
    int64_t          max_val;        // max message-partial value (all message bits = 1)
    int64_t          step;           // 2^(min_weight_in_message_bits)
    bool             has_msg;        // any message bits in this bucket?
    float            sens;           // sensitivity weight (1.0 if none)
};

// ---- Partition bits by weight into message (top-k) and parity ----
struct Partition {
    std::vector<int> message_indices;  // indices into ch.bits (top-k by weight, desc)
    std::vector<int> parity_indices;   // indices into ch.bits (remaining, ascending weight)
};

inline Partition partition_bits(const Chunk& ch, int k) {
    int n = (int)ch.bits.size();
    // Sort all indices by (-weight, index) descending
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
        return (ch.weights[a] != ch.weights[b])
             ? ch.weights[a] > ch.weights[b]
             : a < b;
    });

    Partition p;
    p.message_indices.assign(order.begin(), order.begin() + k);
    // parity: remaining sorted by ascending weight
    std::vector<int> rest(order.begin() + k, order.end());
    std::stable_sort(rest.begin(), rest.end(), [&](int a, int b) {
        return (ch.weights[a] != ch.weights[b]) ? ch.weights[a] < ch.weights[b] : a < b;
    });
    p.parity_indices = rest;
    return p;
}

// ---- Build per-bucket metadata ----
inline std::vector<BucketMeta> build_bucket_meta(
    const Chunk& ch,
    const Partition& part,
    const std::vector<float>* sens_weights)
{
    int B = (int)ch.buckets.size();
    // Map message position → its index within message_indices
    std::vector<int> pos_to_midx(ch.bits.size(), -1);
    for (int i = 0; i < (int)part.message_indices.size(); i++)
        pos_to_midx[part.message_indices[i]] = i;

    std::vector<BucketMeta> metas(B);
    for (int b = 0; b < B; b++) {
        const auto& bk = ch.buckets[b];
        BucketMeta& meta = metas[b];
        meta.b_idx   = b;
        meta.sens    = (sens_weights && b < (int)sens_weights->size()) ? (*sens_weights)[b] : 1.0f;

        // Collect message positions in [bk.start, bk.end)
        for (int pos = bk.start; pos < bk.end; pos++) {
            int mi = pos_to_midx[pos];
            if (mi >= 0) {
                meta.message_pos.push_back(pos);
                meta.m_indices.push_back(mi);
            }
        }
        meta.has_msg = !meta.message_pos.empty();

        if (meta.has_msg) {
            int min_w = ch.weights[meta.message_pos[0]];
            for (int pos : meta.message_pos) min_w = std::min(min_w, ch.weights[pos]);
            meta.step = (1LL << min_w);
            // max value representable by message bits in this bucket
            int64_t mx = 0;
            for (int pos : meta.message_pos) mx += (1LL << ch.weights[pos]);
            meta.max_val = mx;
            // MESSAGE-PARTIAL old value: sum of 2^w for message bits that are 1.
            // Intentionally excludes parity bit contributions so that range checks
            // (0 ≤ new_val ≤ max_val) and bit extraction (new_val >> w) & 1 are correct.
            int64_t msg_partial = 0;
            for (int pos : meta.message_pos)
                if (ch.bits[pos]) msg_partial += (1LL << ch.weights[pos]);
            meta.old_val = msg_partial;
        } else {
            meta.step    = 0;
            meta.max_val = 0;
            meta.old_val = 0;
        }
    }
    return metas;
}

// ---- Extract message bit value from new bucket integer value ----
// Returns the bit at weight w given new_val.
inline int extract_bit(int64_t new_val, int weight) {
    return (new_val >> weight) & 1;
}

// ---- Compute L2 score for candidate bucket values ----
inline float compute_score_l2(
    const std::vector<BucketMeta>& metas,
    const std::vector<int64_t>& new_vals)
{
    float score = 0.0f;
    for (int b = 0; b < (int)metas.size(); b++) {
        float diff = (float)(new_vals[b] - metas[b].old_val);
        score += metas[b].sens * diff * diff;
    }
    return score;
}

// ---- Search3EncodeAndDecode ----
// Returns the mutated Chunk (bits updated to satisfy BCH parity + minimise L2).
inline Chunk search3_encode(
    const Chunk& ch,
    const PMatrix& P,
    int message_parity_size,
    int message_size,
    const std::vector<float>* sens_weights = nullptr)
{
    int n = message_parity_size;
    int k = message_size;
    assert((int)ch.bits.size() == n);

    Partition part = partition_bits(ch, k);
    std::vector<BucketMeta> metas = build_bucket_meta(ch, part, sens_weights);
    int B = (int)metas.size();

    // Base message bits (packed)
    std::vector<int> base_bits = ch.bits;
    std::vector<uint64_t> base_m = pack_message_bits(base_bits, part.message_indices);

    // Original bucket values
    std::vector<int64_t> orig_vals(B);
    for (int b = 0; b < B; b++) orig_vals[b] = metas[b].old_val;

    // Enumerate 3^B combinations of deltas in {-1, 0, +1}
    // For efficiency, iterate as base-3 counter
    int num_combos = 1;
    for (int b = 0; b < B; b++) num_combos *= 3;

    float   best_score  = std::numeric_limits<float>::max();
    int     best_tie    = INT_MAX;
    std::vector<int64_t>  best_new_vals;
    std::vector<int>      best_bits;

    std::vector<int> delta_map = {-1, 0, 1};  // index 0,1,2 → delta -1,0,+1
    std::vector<int> combo(B, 1);              // start at (0,0,...,0) = all zeros

    for (int ci = 0; ci < num_combos; ci++) {
        // Decode combo: combo[b] ∈ {0,1,2} → delta ∈ {-1,0,+1}
        {
            int tmp = ci;
            for (int b = B - 1; b >= 0; b--) {
                combo[b] = tmp % 3;
                tmp /= 3;
            }
        }

        // Compute new bucket values and validate
        std::vector<int64_t> new_vals(B);
        bool valid = true;
        int tie = 0;
        for (int b = 0; b < B; b++) {
            int delta_i = delta_map[combo[b]];
            if (delta_i != 0 && !metas[b].has_msg) { valid = false; break; }
            new_vals[b] = orig_vals[b] + delta_i * metas[b].step;
            if (new_vals[b] < 0 || new_vals[b] > metas[b].max_val) { valid = false; break; }
            if (delta_i != 0) tie++;
        }
        if (!valid) continue;

        // Build candidate message bits from new bucket values
        std::vector<uint64_t> m_bits = base_m;
        for (int b = 0; b < B; b++) {
            int delta_i = delta_map[combo[b]];
            if (delta_i == 0) continue;
            // Update message bits in this bucket
            for (int pi = 0; pi < (int)metas[b].message_pos.size(); pi++) {
                int pos = metas[b].message_pos[pi];
                int mi  = metas[b].m_indices[pi];
                int bit = extract_bit(new_vals[b], ch.weights[pos]);
                int w = mi / 64, b2 = mi % 64;
                m_bits[w] = (m_bits[w] & ~(1ULL << b2)) | ((uint64_t)bit << b2);
            }
        }

        // Compute BCH parity
        auto parity = P.compute_parity(m_bits);

        // Assemble full candidate bit vector
        std::vector<int> candidate(ch.bits);
        for (int i = 0; i < k; i++) {
            int mi = part.message_indices[i];
            candidate[mi] = (m_bits[i / 64] >> (i % 64)) & 1;
        }
        for (int j = 0; j < (int)parity.size(); j++) {
            candidate[part.parity_indices[j]] = parity[j];
        }

        // Score (L2 weighted) — use FULL bucket value change (message bits + BCH parity).
        // Parity bits also reside in buckets, so the true distortion includes their
        // contribution.  Compare against ch.buckets[b].value (the original full value).
        float score = 0.0f;
        for (int b = 0; b < B; b++) {
            const auto& bk_orig = ch.buckets[b];
            int64_t full_new = 0;
            for (int pos = bk_orig.start; pos < bk_orig.end; pos++) {
                if (candidate[pos]) full_new += (1LL << ch.weights[pos]);
            }
            float diff = (float)(full_new - bk_orig.value);
            score += metas[b].sens * diff * diff;
        }

        if (score < best_score || (score == best_score && tie < best_tie)) {
            best_score    = score;
            best_tie      = tie;
            best_new_vals = new_vals;
            best_bits     = candidate;
        }
    }

    // Build output chunk
    Chunk out = ch;
    if (!best_bits.empty()) {
        out.bits = best_bits;
        update_chunk_buckets(out, out.bits);
    }
    return out;
}
