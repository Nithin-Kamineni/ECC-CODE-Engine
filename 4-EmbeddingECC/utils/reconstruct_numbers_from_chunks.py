def reconstruct_numbers_from_chunks(chunks):
    """
    Rebuild the original numbers (and their MSB->LSB bit arrays) from
    messageSliceBasedOnChunkSize(...) output.

    Args:
        chunks: list of dicts, each with:
          - "sliced_bit_weights": list[int] (0..bit_size-1)
          - "sliced_message_nums": list[{"index": int, "value": int, "start": int, "end": int}]

    Returns:
        list[dict]: [
          {"original_number": int, "bits": list[int]}, ...
        ]
    """
    if not chunks:
        return []

    # Infer bit_size from any chunk's weights (max weight + 1)
    max_w = -1
    max_index = -1
    for ch in chunks:
        if "sliced_bit_weights" not in ch or "sliced_message_nums" not in ch:
            raise ValueError("Invalid chunk format.")
        if ch["sliced_bit_weights"]:
            mw = max(ch["sliced_bit_weights"])
            if mw > max_w:
                max_w = mw
        for rec in ch["sliced_message_nums"]:
            if "index" not in rec or "value" not in rec:
                raise ValueError("Invalid sliced_message_nums entry.")
            if rec["index"] > max_index:
                max_index = rec["index"]

    if max_w < 0 or max_index < 0:
        return []

    bit_size = max_w + 1
    num_values = max_index + 1

    # Accumulate masked partials by OR-ing them per index
    totals = [0] * num_values
    for ch in chunks:
        for rec in ch["sliced_message_nums"]:
            j = rec["index"]
            totals[j] |= int(rec["value"])

    # Produce MSB->LSB bit arrays for each reconstructed integer
    out = []
    for v in totals:
        bits = [ (v >> pos) & 1 for pos in range(bit_size - 1, -1, -1) ]  # MSB..LSB
        out.append({"original_number": v, "bits": bits})
    return out

# After your existing split:
# rows = [
#         {"original_number": 7, "bits": [1,1,1]},  # 7 = 111 (msb->lsb)
#         {"original_number": 5, "bits": [1,0,1]},  # 5 = 101
#         {"original_number": 2, "bits": [0,1,0]},  # 2 = 010
#     ]
# chunks = messageSliceBasedOnChunkSize(rows, chunk_size=5)
# for c in chunks:
#     print(c)

# print()
# restored = reconstruct_numbers_from_chunks(chunks)
# for r in restored:
#     print(r)