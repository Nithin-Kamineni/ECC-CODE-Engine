# Sensitivity-study — Part 1 (table)

Compares accuracy with vs without sensitivity-weighted ECC embedding, at two
BER (t) values, across all archs that have data. `0-Data` can only hold the
results of one `EMBED_SENSITIVITY` setting at a time, so this is a two-pass
procedure:

## Procedure

1. **No-sensitivity pass.** Set `EMBED_SENSITIVITY=false` and pick the archs
   you want (default coverage needs all 4):
   ```bash
   EMBED_SENSITIVITY=false EMBED_ARCHS="resnet18 resnet50 mobilenet_v2 efficientnet_b0" \
       bash EmbedAndEvaluate.sh
   ```
   Wait for stages 4-7 to finish.

2. **Capture the no-sensitivity snapshot:**
   ```bash
   cd Ablasian-Studies/Sensitivity-study/1-table
   bash run_capture.sh --mode no-sensitivity
   ```
   This writes `results/no_sensitivity_accuracies.json` and prints `[info]`
   saying to come back after the sensitivity pass — no table yet.

3. **Sensitivity pass.** Set `EMBED_SENSITIVITY=true`, re-run the same archs
   (this overwrites `0-Data` in place):
   ```bash
   EMBED_SENSITIVITY=true EMBED_ARCHS="resnet18 resnet50 mobilenet_v2 efficientnet_b0" \
       bash EmbedAndEvaluate.sh
   ```

4. **Capture the sensitivity snapshot — this also builds the table:**
   ```bash
   bash run_capture.sh --mode sensitivity
   ```
   Since `results/no_sensitivity_accuracies.json` already exists, this run
   also produces:
   - `results/comparison_table.md`
   - `results/comparison_table.json`

## Notes

- Default BERs compared: t=1 and t=6. Override with `--t-values 2 4` etc. on
  either `run_capture.sh` call (must match between both calls to land in the
  same table).
- Default dataset/qmode/bits/approach: IMAGENET / QAT / 8-bit / search3.
  Override with `--dataset`, `--qmode`, `--quant-bits`, `--approach`.
- Any arch/t-value missing an `accuracy.json` (e.g. resnet18/resnet50 before
  they've been run) is logged as `[skip]` and shown as `N/A` in the table —
  it does not stop the script.
- Re-running either step overwrites that mode's snapshot and regenerates the
  table from the latest data on both sides.
