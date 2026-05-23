#pragma once
// slicing.h — C++ port of convert_to_binary + messageSliceBasedOnChunkSize.
// Exact algorithmic equivalent of the Python utilities.

#include <cassert>
#include <cstdint>
#include <vector>

// ---- Chunk data structure ----
struct BucketSpan {
    int     index;      // which original number this span belongs to
    int64_t value;      // partial masked integer value (bits in this span only)
    int     start;      // chunk-local start bit (inclusive)
    int     end;        // chunk-local end bit (exclusive)
};

struct Chunk {
    std::vector<int>         bits;     // 0/1, length = chunk_size
    std::vector<int>         weights;  // bit weight (bit_size-1 down to 0)
    std::vector<BucketSpan>  buckets;  // per-number spans within this chunk
};

// ---- slice_into_chunks -------------------------------------------------------
// Direct port of messageSliceBasedOnChunkSize() from Python.
//
// vals:       raw uint8 values (each in 0..255), num_vals elements
// bit_size:   always 8 (one byte per weight value)
// chunk_size: number of bits per chunk (e.g. 63 for n=63, 30 for parfix k=30)
//
// Returns one Chunk per chunk_size bits; pads last chunk with zeros.
// ---- -------------------------------------------------------------------------
inline std::vector<Chunk> slice_into_chunks(
    const uint8_t* vals, int num_vals,
    int bit_size,        // = 8
    int chunk_size)
{
    // Weight[j] = bit_size - 1 - j  (MSB-first: weight of the j-th bit)
    // Bit j within a number represents value 2^(bit_size-1-j).
    // Exactly mirrors Python: Weights = [i for i in range(bit_size-1, -1, -1)]
    // Weights[j] = bit_size - 1 - j

    int total_bits = num_vals * bit_size;
    std::vector<Chunk> out;

    for (int chunk_start = 0; chunk_start < total_bits; chunk_start += chunk_size) {
        int chunk_end = std::min(chunk_start + chunk_size, total_bits);

        Chunk ch;
        ch.bits.reserve(chunk_size);
        ch.weights.reserve(chunk_size);

        // Which numbers overlap this chunk?
        int start_num = chunk_start / bit_size;
        int end_num   = (chunk_end - 1) / bit_size;

        for (int j = start_num; j <= end_num; j++) {
            int block_start = j * bit_size;
            int block_end   = block_start + bit_size;

            int ov0 = std::max(block_start, chunk_start);
            int ov1 = std::min(block_end,   chunk_end);
            if (ov0 >= ov1) continue;

            // Local indices within number j
            int local_start = ov0 - block_start;  // inclusive
            int local_end   = ov1 - block_start;  // exclusive

            int chunk_local_start = (int)ch.bits.size();

            // Append bits and weights for this overlap
            uint8_t v = vals[j];
            for (int b = local_start; b < local_end; b++) {
                int weight = bit_size - 1 - b;  // MSB-first weight
                int bit    = (v >> (bit_size - 1 - b)) & 1;
                ch.bits.push_back(bit);
                ch.weights.push_back(weight);
            }
            int chunk_local_end = (int)ch.bits.size();

            // Compute partial masked integer value for this span
            // Mirrors Python: pos_low = bit_size - local_end
            //                 mask = ((1 << length) - 1) << pos_low
            //                 partial_val = v & mask
            int length  = local_end - local_start;
            int pos_low = bit_size - local_end;
            int64_t mask = ((1LL << length) - 1) << pos_low;
            int64_t partial_val = v & mask;

            ch.buckets.push_back({j, partial_val, chunk_local_start, chunk_local_end});
        }

        // Pad to chunk_size with zeros (bits=0, weight=0)
        int deficit = chunk_size - (int)ch.bits.size();
        for (int i = 0; i < deficit; i++) {
            ch.bits.push_back(0);
            ch.weights.push_back(0);
        }

        out.push_back(std::move(ch));
    }
    return out;
}

// ---- Convenience: pack message bits into uint64_t words ----------------------
// message_indices: indices into ch.bits that are message bits (top-k by weight)
// Returns packed uint64_t vector (bit i = ch.bits[message_indices[i]])
inline std::vector<uint64_t> pack_message_bits(
    const std::vector<int>& bits,
    const std::vector<int>& message_indices)
{
    int k = (int)message_indices.size();
    int words = (k + 63) / 64;
    std::vector<uint64_t> packed(words, 0);
    for (int i = 0; i < k; i++) {
        if (bits[message_indices[i]])
            packed[i / 64] |= (1ULL << (i % 64));
    }
    return packed;
}
