# How to Recover Original Weight Ordering from Pattern-Permuted Weights

## What was done

`prepare_patterns.py` found a hardware-friendly interleaver permutation for each
layer that spreads high-sensitivity weights across ECC groups. It then:

1. Saved the full weight tensor **in the permuted order** as `{layer}_weights_perm.npy`
   — this is what the hardware reads sequentially.
2. Saved the **permutation index array** as `{layer}_perm.npy`
   — `perm[i]` is the *original* flat index of the i-th element in permuted order.
   - Equivalently: `weights_perm[i] == weights_orig[perm[i]]`
3. Saved the **inverse permutation** as `{layer}_inv_perm.npy`
   — `inv_perm[orig_idx]` is where that weight appears in the permuted array.
   - Equivalently: `weights_orig[j] == weights_perm[inv_perm[j]]`
4. Saved `pattern_manifest.json` with paths and all metadata for every layer.

All files are in `0-Data/artifacts/patterns/`.

---

## How to go from permuted order → original order

```python
import numpy as np
import json

# Load the manifest
manifest = json.load(open("0-Data/artifacts/patterns/pattern_manifest.json"))

# Pick a layer
layer = "layer1.0.conv1.weight"
entry = manifest[layer]

# Load files
w_perm   = np.load(entry["weights_perm_file"])   # shape: (N,)  — hardware order
inv_perm = np.load(entry["inv_perm_file"])        # shape: (N,)  — reversal map

# Recover original flat ordering
w_orig_flat = w_perm[inv_perm]

# Reshape back to original tensor shape
w_orig = w_orig_flat.reshape(entry["shape"])
```

**One-liner check** (should be True for every weight):
```python
assert np.allclose(w_perm[inv_perm], w_orig_flat)
```

---

## How to go from original order → permuted order

```python
perm   = np.load(entry["perm_file"])      # shape: (N,)
w_flat = w_orig.flatten()
w_perm_reconstructed = w_flat[perm]       # == w_perm
```

---

## How to apply a processed permuted tensor back to the original model

After any downstream processing (e.g. ECC encoding/decoding, quantization) that
operates on the permuted `w_perm`, recover the corrected weights and reload them:

```python
import torch

# Recover original-order tensor
w_corrected_flat = processed_w_perm[inv_perm]
w_corrected = torch.tensor(w_corrected_flat).reshape(entry["shape"])

# Patch into model state dict
ckpt = torch.load(entry["model_path"], map_location="cpu")
sd   = ckpt.get("state_dict", ckpt)
sd[layer] = w_corrected
torch.save({"state_dict": sd}, "model_corrected.pth")
```

---

## Key invariant

```
weights_perm  = weights_orig[perm]         # permute
weights_orig  = weights_perm[inv_perm]     # invert
perm[inv_perm[i]] == i  for all i          # round-trip identity
```

---

## Manifest fields reference

| Field | Meaning |
|---|---|
| `layer` | Parameter name (matches PyTorch `model.named_parameters()` key) |
| `N` | Total number of weights in this layer (flattened) |
| `shape` | Original tensor shape (e.g. `[64, 3, 3, 3]` for a conv layer) |
| `best_family` | Interleaver type: `stride`, `block`, or `bit_reversal` |
| `best_param` | Family parameter (stride `s`, block rows `R`, or `null`) |
| `best_excess` | Total violation score after permutation (0 = perfect) |
| `best_violating_groups` | Groups still exceeding `max_sens` (0 = perfect) |
| `perm_file` | Absolute path to `{layer}_perm.npy` |
| `inv_perm_file` | Absolute path to `{layer}_inv_perm.npy` |
| `weights_perm_file` | Absolute path to `{layer}_weights_perm.npy` |
| `model_path` | Checkpoint used to extract original weights |
