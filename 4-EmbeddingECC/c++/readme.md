# Project context: ECC-CODE-Engine — how the C++ embedding path is built and run

Use this as project instructions (e.g. paste into the project's custom instructions, or save
as `CLAUDE.md` in the repo). It tells you how containers are created and run for this project
**now**, superseding any earlier instructions.

## The project still uses an Apptainer/Singularity container — only the build method changed

The C++ embedding code (`4-EmbeddingECC/c++/ecc_embed.cpp`) runs **inside an
Apptainer/Singularity image** (`ecc_cpp.sif`), on HiPerGator (UF Research Computing). This has
not changed. What changed is **how the image is created**.

**Do NOT** build the image from `ecc_embed.def`. **Do NOT** use `--fakeroot`. **Do NOT** use
`--remote`. All three are dead ends on this account:
- `--fakeroot` fails — the account has no `/etc/subuid` mapping.
- plain `singularity build` of a `.def` fails — requires root.
- `--remote` has been removed from current Apptainer/Singularity.
- `ecc_embed.def` is therefore **obsolete**; ignore it.

**Instead**, the image is obtained by pulling the prebuilt official `gcc` Docker image, which
already contains g++/gcc/make. Pulling a public Docker image is unprivileged and works on a
login node:

```bash
module load apptainer
cd /blue/rewetz/vkamineni/Projects/ECC-CODE-Engine/4-EmbeddingECC/c++
apptainer build ecc_cpp.sif docker://gcc:13
```

## Standard workflow

1. **Create image (one-time):** `apptainer build ecc_cpp.sif docker://gcc:13`
2. **Compile (after code changes):**
   `apptainer exec --bind /blue ecc_cpp.sif make -C <c++ dir> -j4` → produces `ecc_embed_cpp`
3. **Run directly:**
   `apptainer exec --bind /blue ecc_cpp.sif ./ecc_embed_cpp --dataset ... --approach search3 ...`
   (CPU-only — no `--nv` needed for the C++ path.)
4. **Run via SLURM:** from `4-EmbeddingECC/`, submit
   `EMBED_RUN_CPP=true EMBED_APPROACH=search3 sbatch run.sh`

## Key facts to respect when helping

- `run.sh` uses the C++ runner only when `EMBED_RUN_CPP=true` AND `EMBED_APPROACH` ∈
  {`search3`, `greedy`, `no`}. All other approaches (`parfit`, `parfix`, `replace`) use the
  Python runner `ecc_embed.py`, which runs in a **separate, unchanged** Python `.sif` (with
  `--nv`).
- `singularity` is symlinked to `apptainer` on HiPerGator; either command name works.
- Always build/run on `/blue` (or `/red`), never `/orange`.
- The C++ code currently targets **C++17 + pthreads** (and OpenMP if used). It needs no
  Python or PyTorch at runtime.

## Guidance for future container/build suggestions

- Prefer **pulling a prebuilt image** (`docker://...`) over building a privileged custom image.
  Only suggest a custom `.def` build if the user confirms they have obtained an `/etc/subuid`
  fakeroot mapping from RC support (`support.rc.ufl.edu`).
- If a new C++ dependency is needed that isn't in the `gcc` image, prefer (a) a different
  prebuilt base image that already includes it, or (b) static linking — over rebuilding a
  privileged image.
- When editing `run.sh`, keep both the C++ and Python branches intact; do not remove the
  Python path.