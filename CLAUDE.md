# CLAUDE.md

Hack Nation MIT 2026 Summer — 24-hour hackathon, personal workspace (not team-shared). See `docs/hackathon_info.md` for the competition and `docs/24h_plan.md` for the full timeline.

## Language

- Code, comments, docs, commit messages: **English**. Pre-existing Korean docs stay as they are until translated on request.
- Chat responses to the user (explanations, summaries, questions): **Korean**. Only written artifacts in the repo are English — conversation stays in Korean.

## Build order — search papers, budget the time, then write code

Any feature that involves training, fine-tuning, or hyperparameter search goes through these three steps in order before code gets written. Skipping straight to code is the single riskiest habit on a 24-hour budget.

1. **Search prior work** — query the paper search harness in `ai/harness/papers` before picking a technique.
   ```
   python -m harness "<query>" --top 5
   python -m harness "" --by-capability <capability> --top 10
   ```
   Usage details: `ai/harness/papers/README.md`.

2. **Time budget gate** — estimate the run time per `docs/time_budgeting.md` and check it with `scripts/time_probe.py`. If it doesn't clear the gate, cut scope in the order documented there (smaller data subset → fewer epochs → smaller model → LoRA instead of full fine-tune → zero-shot fallback) and re-estimate.

3. **Write the code** — only after the gate passes, extend the `ml/pytorch`, `ml/tensorflow`, `ai/cnn`, or `ai/llm` skeletons. Follow `docs/workflow.md` for experiment tracking (branch + wandb run + config, 1:1:1), and log estimated vs. actual time in `memory/experiment_log.md` when done.

For benchmark or custom-task evaluation, check `ai/harness/README.md` first (lm-eval vs. inspect-ai vs. the custom harness).

## Model usage and token budget

This workspace drives Claude Code with two models: Claude Opus 4.8 (`claude-opus-4-8`) as the default, and Claude Fable 5 (`claude-fable-5`) reserved for genuinely hard blockers. Same spirit as the time-budget gate above — don't spend more than the problem needs.

- **Default to Opus 4.8** for everything — implementation, debugging, refactors, reviewing experiment code. It's half the price of Fable 5 ($5/$25 per MTok vs. $10/$50) and fast enough to stay interactive.
- **Escalate to Fable 5 only when stuck** — a bug that survived several Opus attempts, a genuinely hard architecture or algorithm decision, or a long autonomous multi-step run where its stronger long-horizon planning is worth the 2x cost. A single Fable 5 turn at high effort can run for several minutes, which costs wall-clock time as much as tokens — don't reach for it on routine work.
- **Keep context lean.** Don't paste full papers, logs, or datasets into chat — the paper harness already returns filtered top-k results instead of full text; use that. Use `/compact` when a session's history gets long, and offload broad exploratory reads to subagents so the main thread's context doesn't bloat.
- **Delegate mechanical subagent work at low effort.** Cheap, well-scoped subagent tasks (a targeted grep, a boilerplate file scaffold) don't need the same reasoning depth as the main implementation thread.

## Code comment style

- Comments use `##` only. Avoid single-`#` inline comments and long docstrings.
- Comment only on core classes, functions, and data structure definitions, and only where the WHY isn't obvious from the code. Skip comments that just restate WHAT the code does.
- No conversational references ("as requested", "per our discussion"), no citation-dropping ("per paper X"), no AI-generated-sounding narration.
- If a name already makes the intent clear, skip the comment entirely.

## Hardware defaults

Designed around a local RTX 4060 8GB. Moving to the cloud touches exactly two files — `.env` and `requirements/pytorch.txt` — procedure in `docs/portability_guide.md`.
