#pragma once
// reconstruct.h — C++ port of reconstruct_numbers_from_chunks().
// Reassembles original integer values from mutated chunk data.

#include <cstdint>
#include <vector>
#include "slicing.h"

// Reconstruct uint8 values from a list of mutated Chunks.
// Mirrors Python: totals[j] |= chunk.buckets[b].value; then extract bits.
//
// Returns the reconstructed values as uint8 (0..255).
inline std::vector<uint8_t> reconstruct_from_chunks(
    const std::vector<Chunk>& chunks,
    int num_vals,
    int bit_size = 8)
{
    // Accumulate partial values via bitwise OR
    std::vector<int64_t> totals(num_vals, 0);
    for (const auto& ch : chunks) {
        for (const auto& bk : ch.buckets) {
            if (bk.index >= 0 && bk.index < num_vals) {
                totals[bk.index] |= bk.value;
            }
        }
    }

    // Convert total integer values back to uint8
    // Each total should be in range [0, 255]; if it went out of range due to
    // the encoding, clip it.
    std::vector<uint8_t> result(num_vals);
    for (int i = 0; i < num_vals; i++) {
        int64_t v = totals[i];
        if (v < 0)   v = 0;
        if (v > 255) v = 255;
        result[i] = static_cast<uint8_t>(v);
    }
    return result;
}

// Apply mutated bits back into chunk bucket values.
// Called after encoding to rebuild the sliced_message_nums values
// from the mutated bit array and original bucket spans.
inline void update_chunk_buckets(Chunk& ch, const std::vector<int>& mutated_bits) {
    for (auto& bk : ch.buckets) {
        // Recompute partial value from mutated bits using weights
        int64_t val = 0;
        for (int i = bk.start; i < bk.end; i++) {
            if (mutated_bits[i]) {
                val += (1LL << ch.weights[i]);
            }
        }
        bk.value = val;
    }
}
