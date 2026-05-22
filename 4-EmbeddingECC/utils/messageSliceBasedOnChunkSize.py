def messageSliceBasedOnChunkSize(encoded_rows, chunk_size):
    """
    Faster splitter for the output of convert_to_binary(...).

    For each chunk (size = chunk_size), return:
      - "sliced_message_bits": list[int]
      - "sliced_bit_weights":  list[int]
      - "sliced_message_nums": list[{"index": j, "value": partial_masked_int,
                                     "start": local_start, "end": local_end}]
        Only numbers that overlap the chunk are included.

    Optimizations:
      * Visit only numbers that intersect the chunk (no full scan).
      * Use bit masks on the original integer to compute the partial value
        in O(1) for the overlapped slice (no per-bit summing).
      * Stream construction of chunks; no big "flat" arrays allocated.

    Assumptions:
      * All rows have identical bit_size.
      * Row format from convert_to_binary():
           {"original_number": int, "bits": list[int], "weights": list[int]}
      * Weights encode positions [0..bit_size-1] (0 = LSB).
      * Order can be MSB->LSB or LSB->MSB; auto-detected per rows.
    """

    if(not encoded_rows):
        raise ValueError("encoded_rows is empty")
        
    # Validate rows and infer bit_size; ensure consistency across rows
    num_values = len(encoded_rows)
    bit_size = len(encoded_rows[0]["bits"])
    total_bits = num_values * bit_size

    Weights = [i for i in range(bit_size-1,-1,-1)]
    
    out = []

    # Helper: append a slice from row j into current chunk buffers
    def _append_row_slice(j, local_start, local_end, bits_accum, weights_accum):
        # local indices w.r.t that row (0..bit_size)
        row = encoded_rows[j]
        # We can extend directly; Python slicing is fast in C.
        bits_accum.extend(row["bits"][local_start:local_end])
        weights_accum.extend(Weights[local_start:local_end])

    # Helper: compute partial integer of row j for local slice [ls, le)
    # Using mask on the original integer (O(1)).
    def _masked_partial_value(j, local_start, local_end):
        if local_start >= local_end:
            return 0
        v = int(encoded_rows[j]["original_number"])
        length = local_end - local_start
        pos_low = bit_size - local_end
        
        mask = ((1 << length) - 1) << pos_low
        return v & mask

    for chunk_start in range(0, total_bits, chunk_size):
        chunk_end = min(chunk_start + chunk_size, total_bits)

        bits_chunk = []
        weights_chunk = []
        nums_chunk = []

        # Which numbers overlap this chunk?
        start_num = chunk_start // bit_size
        end_num = (chunk_end - 1) // bit_size  # inclusive
        for j in range(start_num, end_num + 1):
            # Row j occupies [j*bit_size, (j+1)*bit_size) in the global stream
            block_start = j * bit_size
            block_end = block_start + bit_size

            ov0 = max(block_start, chunk_start)
            ov1 = min(block_end, chunk_end)
            if ov0 >= ov1:
                continue  # no overlap

            # Convert to local indices within row j
            local_start = ov0 - block_start
            local_end = ov1 - block_start  # exclusive

            # Append the overlapped bits & weights into the chunk
            _append_row_slice(j, local_start, local_end, bits_chunk, weights_chunk)

            # Compute partial integer via mask
            partial_val = _masked_partial_value(j, local_start, local_end)

            # Record where this row landed inside the chunk (chunk-local indices)
            chunk_local_start = len(bits_chunk) - (local_end - local_start)
            chunk_local_end = len(bits_chunk)  # exclusive

            nums_chunk.append({
                "index": j,
                "value": partial_val,
                "start": chunk_local_start,
                "end": chunk_local_end
            })

        # --- Pad this chunk to exactly chunk_size with zeros (bits=0, weight=0) ---
        deficit = chunk_size - len(bits_chunk)
        if deficit > 0:
            bits_chunk.extend([0] * deficit)
            weights_chunk.extend([0] * deficit)
        # -------------------------------------------------------------------------

        out.append({
            "sliced_message_bits": bits_chunk,
            "sliced_bit_weights": weights_chunk,
            "sliced_message_nums": nums_chunk
        })

    return out

# --- example usage ---
# if __name__ == "__main__":
#     # Suppose you already ran:
#     # rows = convert_to_binary([7, 5, 2], bit_size=3, order="msb")
#     rows = [
#         {"original_number": 7, "bits": [1,1,1]},  # 7 = 111 (msb->lsb)
#         {"original_number": 5, "bits": [1,0,1]},  # 5 = 101
#         {"original_number": 2, "bits": [0,1,0]},  # 2 = 010
#     ]
#     chunks = messageSliceBasedOnChunkSize(rows, chunk_size=5)
#     for c in chunks:
#         print(c)