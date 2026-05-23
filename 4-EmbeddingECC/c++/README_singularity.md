# C++ ECC Embedding — Singularity Build & Run Guide (HiPerGator)

## Overview

The C++ implementation uses only the standard C++17 library (no Python, no PyTorch).
It needs a container with `g++` and `make` to compile, then runs the binary directly.

---

## Step 1: Build the Singularity image

Run **once** on a HiPerGator login node. Two options:

### Option A — fakeroot (recommended, no sudo needed)

```bash
module load singularity
cd /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/4-EmbeddingECC/c++
singularity build --fakeroot ecc_cpp.sif ecc_embed.def
```

### Option B — Remote build (if fakeroot is not enabled for your account)

```bash
module load singularity
cd /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/4-EmbeddingECC/c++
singularity build --remote ecc_cpp.sif ecc_embed.def
# You will need a Sylabs account (free): https://cloud.sylabs.io
```

### Verify the image

```bash
singularity exec ecc_cpp.sif g++ --version
# Should print: g++ (Ubuntu ...) 11.x.x or newer
```

---

## Step 2: Compile the C++ binary

Run once (or after any code change). Compilation takes a few seconds.

```bash
CPP_DIR="/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/4-EmbeddingECC/c++"
singularity exec --bind /blue "${CPP_DIR}/ecc_cpp.sif" \
    make -C "${CPP_DIR}" -j4
# Binary created at: 4-EmbeddingECC/c++/ecc_embed_cpp
```

---

## Step 3: Run the binary

```bash
CPP_DIR="/blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/4-EmbeddingECC/c++"
singularity exec --bind /blue "${CPP_DIR}/ecc_cpp.sif" \
    "${CPP_DIR}/ecc_embed_cpp" \
        --dataset       CIFAR10 \
        --arch          resnet18 \
        --quant-bits    8 \
        --t-value       2 \
        --approach      search3 \
        --codeword      63 \
        --workers       24 \
        --patterns-dir  /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/0-Data/artifacts/patterns \
        --chunks-dir    /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/0-Data/artifacts/embeddedECC_Chunks
```

Or via SLURM (from the 4-EmbeddingECC directory):

```bash
cd /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/4-EmbeddingECC
EMBED_RUN_CPP=true EMBED_APPROACH=search3 sbatch run.sh
```

---

## run.sh auto-compilation

`run.sh` automatically compiles the binary if `ecc_embed_cpp` does not exist:

```bash
if [ ! -f "${CPP_BINARY}" ]; then
    singularity exec --bind /blue "${CPP_SIF}" make -C "${CPP_DIR}" -j4
fi
```

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `--fakeroot: permission denied` | Request fakeroot access: `rc-help@ufl.edu` or use `--remote` |
| `g++: command not found` | The image may not have built correctly — rebuild |
| Compilation fails (`filesystem` not found) | Ensure g++ ≥ 8; Ubuntu 22.04 image has g++ 11 which is fine |
| SLURM job crashes immediately | Check `logs/ecc-embed.*_%a.err` for the error message |

---

## File sizes (approximate)

| File | Size |
|---|---|
| `ecc_cpp.sif` | ~80 MB (Ubuntu 22.04 + build-essential) |
| `ecc_embed_cpp` | ~2 MB (optimised binary) |
| Per-layer JSONL chunks | Varies (typically 5–500 MB per layer depending on N) |
