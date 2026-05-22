def convert_to_binary(nums, bit_size=8, order="msb"):
    """
    Convert each integer in `nums` to fixed-width binary (length = bit_size),
    without interleaving across numbers.

    Args:
        nums: iterable of non-negative integers
        bit_size: one of {4, 8, 16, 32, 64}
        order: "msb" -> emit bits MSB->LSB per number
               "lsb" -> emit bits LSB->MSB per number

    Returns:
        list[dict]: [
          {
            "original_number": <int>,
            "bits":    list[int],  # 0/1 of length bit_size (order respected)
          },
          ...
        ]
    """
    if bit_size not in (4, 8, 16, 32, 64):
        raise ValueError("bit_size must be one of {4, 8, 16, 32, 64}.")
    if order not in ("msb", "lsb"):
        raise ValueError("order must be 'msb' or 'lsb'.")

    max_val = (1 << bit_size) - 1
    vals = []
    for i, x in enumerate(nums):
        xi = int(x)
        if xi < 0 or xi > max_val:
            raise ValueError(f"nums[{i}]={x!r} out of range [0, {max_val}].")
        vals.append(xi)

    out = []
    if order == "msb":
        positions = range(bit_size - 1, -1, -1)   # e.g., 7..0
    else:
        positions = range(0, bit_size)            # e.g., 0..7

    for v in vals:
        bits = []
        for pos in positions:
            bits.append((v >> pos) & 1)
        out.append({
            "original_number": v,
            "bits": bits,
        })
    return out


# --- quick sanity check ---
# if __name__ == "__main__":
#     values = [10, 231, 52, 135]
#     message_bits = convert_to_binary(values, bit_size=8, order="msb")
#     for row in message_bits:
#         print(row)