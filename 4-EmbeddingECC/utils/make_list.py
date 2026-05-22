import numpy as np

def make_list(n: int, bits_quant: int = 256, seed = None) -> list[int]:
    """
    Vectorized replacement for make_list:
      - stratifies the 0..bits_quant-1 range into n buckets
      - samples ONE integer uniformly from each bucket
      - shuffles the result
    """
    if n <= 0:
        return []
    rng = np.random.default_rng(seed)

    # bucket edges (lo inclusive, hi inclusive)
    i = np.arange(n, dtype=np.int64)
    lo = (i * bits_quant) // n
    hi = ((i + 1) * bits_quant) // n - 1
    width = (hi - lo + 1)

    # sample per-bucket: floor(U * width) + lo  (works even when width varies)
    u = rng.random(n)
    samples = lo + np.floor(u * width).astype(np.int64)

    # shuffle
    rng.shuffle(samples)
    return samples.astype(int).tolist()