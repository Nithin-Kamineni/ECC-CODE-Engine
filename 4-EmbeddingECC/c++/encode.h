#pragma once
// encode.h — process_payload dispatcher (mirrors ecc_embed.py::process_payload).
//
// Encodes a slice of uint8 values (already shifted from int8 by +128) using the
// chosen ECC approach. Returns the mutated values (still uint8) and L1 distortion.

#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include "bch.h"
#include "slicing.h"
#include "reconstruct.h"
#include "search3.h"
#include "greedy.h"

enum class Approach { SEARCH3, GREEDY, NO };

inline Approach parse_approach(const std::string& s) {
    if (s == "search3") return Approach::SEARCH3;
    if (s == "greedy")  return Approach::GREEDY;
    if (s == "no")      return Approach::NO;
    throw std::invalid_argument("C++ ECC: unsupported approach '" + s +
                                "' (only search3, greedy, no)");
}

struct PayloadResult {
    std::vector<uint8_t> values;   // mutated uint8 (0..255)
    double               distortion;  // mean L1 distortion in uint8 space
};

// ---- process_payload ---------------------------------------------------------
// vals:        uint8 values (shifted from int8 by +128), length num_vals
// sens:        per-value sensitivity float (may be nullptr)
// approach:    encoding approach
// chunk_size:  bits per chunk
// message_parity_size, message_size: BCH parameters
// P:           packed BCH parity matrix
// move_range:  greedy move range (default 4, matches Python)
// ---- -------------------------------------------------------------------------
inline PayloadResult process_payload(
    const uint8_t* vals, int num_vals,
    const float*   sens,
    Approach approach,
    int chunk_size,
    int message_parity_size,
    int message_size,
    const PMatrix& P,
    int move_range = 4)
{
    // 'no' approach: pass through unchanged
    if (approach == Approach::NO) {
        PayloadResult r;
        r.values.assign(vals, vals + num_vals);
        r.distortion = 0.0;
        return r;
    }

    // Slice into chunks
    auto chunks = slice_into_chunks(vals, num_vals, 8, chunk_size);

    // Per-chunk sensitivity weights (one per bucket/number in each chunk)
    // We pass per-bucket sensitivity (indexed by original number index).
    // sens[i] is the sensitivity of the i-th value in this slice.

    std::vector<Chunk> mutated_chunks;
    mutated_chunks.reserve(chunks.size());

    for (auto& ch : chunks) {
        // Build per-bucket sensitivity list for this chunk
        std::vector<float> bucket_sens;
        if (sens) {
            bucket_sens.reserve(ch.buckets.size());
            for (const auto& bk : ch.buckets)
                bucket_sens.push_back(bk.index < num_vals ? sens[bk.index] : 1.0f);
        }
        const std::vector<float>* sens_ptr = sens ? &bucket_sens : nullptr;

        Chunk out;
        if (approach == Approach::SEARCH3) {
            out = search3_encode(ch, P, message_parity_size, message_size, sens_ptr);
        } else { // GREEDY
            out = greedy_encode(ch, P, message_parity_size, message_size, move_range, sens_ptr);
        }
        mutated_chunks.push_back(std::move(out));
    }

    // Reconstruct uint8 values from mutated chunks
    auto mutated_u8 = reconstruct_from_chunks(mutated_chunks, num_vals);

    // Compute mean L1 distortion in uint8 space (mirrors Python)
    double dist = 0.0;
    for (int i = 0; i < num_vals; i++)
        dist += std::abs((int)mutated_u8[i] - (int)vals[i]);
    if (num_vals > 0) dist /= num_vals;

    return {mutated_u8, dist};
}
