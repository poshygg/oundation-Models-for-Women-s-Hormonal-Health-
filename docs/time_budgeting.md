# Time Budget Gate

The checkpoint between "found a technique via `ai/harness/papers`" and "start writing code." Its purpose: never end up with no time left after a single training run.

## When it applies
Any implementation that involves training, fine-tuning, or hyperparameter search. Skip it for inference-only code (serving, frontend wiring, post-processing).

## Procedure

### 1. Check remaining time
Pin down the current phase from `docs/24h_plan.md` first. Per-phase caps:

| Phase | Window | Single-run cap | Cumulative cap |
| --- | --- | --- | --- |
| 1 · MVP skeleton | 2h–8h | 30 min | – |
| 2 · Experiment & tuning | 8h–16h | 2 h | 6 h (parallel runs included) |
| 3 · Integration & demo | 16h–22h | no new training | – |
| 4 · Pitch rehearsal | 22h–24h | no new training | – |

### 2. Estimate the run time
A measured value beats a static table.
1. Run 5–10 steps of the real training loop and measure seconds per step, using the actual batch size and sequence length.
2. `total estimate = seconds_per_step × steps_per_epoch × epochs`.
3. If you can't measure yet, use the rough table below for a first pass, then re-verify with a real measurement as soon as you start coding.

**Rough table (RTX 4060 8GB, order-of-magnitude only — replace with a measurement)**

| Task | Rough time |
| --- | --- |
| CNN fine-tune (ResNet50/EffNet-B0, ~10k images, 5 epochs) | 15–40 min |
| LoRA/QLoRA SFT (7B, 4-bit, hundreds–thousands of samples, 1–2 epochs) | 30 min–2 h |
| Full fine-tune of a 7B-class model | not recommended locally — move to cloud |
| VLM head fine-tune (SigLIP/CLIP) | 10–30 min |
| Whisper small fine-tune | 20–40 min |
| WandB sweep (grid of 8 configs, no worktree parallelism) | per-run time × 8 |

### 3. Go / No-Go
Judge it with `scripts/time_probe.py`:

```powershell
python scripts\time_probe.py --steps-per-epoch 250 --epochs 3 --step-seconds 0.8 --phase 2
```

- Estimate ≤ cap → **GO**, start writing code
- Over the cap → **SCOPE DOWN**, apply these one at a time and re-estimate until it clears
  1. Shrink the dataset to a 10–20% subset
  2. Cut epochs / add an early-stopping criterion
  3. Downgrade to a smaller model (see the usage guide in `docs/model_registry.md`)
  4. Switch full fine-tune → LoRA/QLoRA
  5. Still over? Drop the training run entirely and fall back to zero-shot / a pretrained checkpoint (`docs/24h_plan.md` crisis-response cards)
- If the remaining hackathon time itself is shorter than the estimate: **NO-GO**, switch to an alternative immediately

### 4. Log it
Fill in the "Time budget" field in `memory/experiment_log.md` with estimate vs. actual. As measurements accumulate for a recurring task type, update the rough table above with the real numbers.
