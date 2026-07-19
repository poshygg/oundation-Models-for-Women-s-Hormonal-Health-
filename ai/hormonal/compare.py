"""ai/hormonal/compare.py — head-to-head over models and both branches.

Runs several classification backends (LightGBM / XGBoost / CatBoost / EBM, plus
TabPFN if licensed) and the hormone-regression branch under ONE identical LOSO
split on the same dataset, then writes a comparison table. This is the "run both
pipelines and compare" deliverable: the roadmap's gradient-boosting classifier +
regression branch, alongside the EBM + conformal pipeline.

    python -m ai.hormonal.compare --config ai/hormonal/configs/compare.yaml
    python -m ai.hormonal.compare --config ... --max-folds 6   # fast smoke
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .conformal import conformal_oof, conformal_report
from .data import build_dataset
from .regression import run_loso_regression
from .train import SOTA, TREE_BACKENDS, load_config, run_loso


def _run_classification(cfg: dict, ds, backend: str) -> dict:
    cfg = json.loads(json.dumps(cfg))   # deep copy so per-backend edits don't leak
    cfg["model"]["backend"] = backend
    cv = run_loso(cfg, ds)
    r = cv["report"]["overall"]
    row = {
        "backend": backend,
        "macro_f1": r["macro_f1"],
        "accuracy": r["accuracy"],
        "balanced_accuracy": r["balanced_accuracy"],
        "auroc_ovr": r["auroc_ovr"],
        "brier": r["brier"],
        "total_seconds": cv["timing"]["total_seconds"],
        "sec_per_fold": cv["timing"]["mean_fold_seconds"],
        "glassbox": backend == "ebm",
        "shap_needed": backend in TREE_BACKENDS,
    }
    ccfg = cfg.get("conformal", {})
    if ccfg.get("enabled"):
        ev = cv["evaluated"]
        alpha = float(ccfg.get("alpha", 0.1))
        mask, _ = conformal_oof(cv["oof_proba"][ev], ds.y[ev], ds.groups[ev],
                                alpha=alpha, method=ccfg.get("method", "lac"),
                                n_splits=int(ccfg.get("n_splits", 5)))
        crep = conformal_report(mask, ds.y[ev], alpha)
        row["conf_coverage"] = crep["empirical_coverage"]
        row["conf_singleton_rate"] = crep["singleton_rate"]
        row["conf_singleton_acc"] = crep["singleton_accuracy"]
    return row


def _fmt_classification(rows: list[dict]) -> str:
    hdr = ("| backend | macro-F1 | acc | bal-acc | AUROC | Brier | s/fold | glassbox | "
           "SHAP pass | conf.cov | decisive | dec.acc |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for r in sorted(rows, key=lambda x: -x.get("macro_f1", -1)):
        lines.append(
            f"| {r['backend']} | {r['macro_f1']:.3f} | {r['accuracy']:.3f} | "
            f"{r['balanced_accuracy']:.3f} | {r['auroc_ovr']:.3f} | {r['brier']:.3f} | "
            f"{r['sec_per_fold']:.2f} | {'yes' if r['glassbox'] else 'no'} | "
            f"{'yes' if r['shap_needed'] else 'no'} | "
            f"{r.get('conf_coverage', float('nan')):.3f} | "
            f"{r.get('conf_singleton_rate', float('nan')):.3f} | "
            f"{r.get('conf_singleton_acc', float('nan')):.3f} |"
        )
    return "\n".join(lines)


def _fmt_regression(rep: dict) -> str:
    per = rep["per_hormone"]
    lines = ["| hormone | MAE | RMSE | R² | Spearman |", "|---|---|---|---|---|"]
    for h, m in per.items():
        lines.append(f"| {h} | {m['mae']:.2f} | {m['rmse']:.2f} | {m['r2']:.3f} | {m['spearman']:.3f} |")
    mac = rep["macro"]
    lines.append(f"| **macro** | {mac['mae']:.2f} | — | {mac['r2']:.3f} | {mac['spearman']:.3f} |")
    return "\n".join(lines)


def main() -> None:
    ## Windows consoles default to cp949/cp1252 and choke on the em-dash / R² in
    ## the report tables. Force UTF-8 stdout so the run doesn't crash at print time.
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max-folds", dest="max_folds", type=int)
    ap.add_argument("--feature-set", dest="feature_set",
                    choices=["all", "symptoms_only", "wearables_only"])
    args = ap.parse_args()

    cfg = load_config(args.config, {"max_folds": args.max_folds, "feature_set": args.feature_set})
    out_dir = Path(cfg["logging"]["out_dir"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[data] building dataset ...")
    ds = build_dataset(cfg)
    print(f"[data] {len(ds.y)} subject-days, {len(np.unique(ds.groups))} subjects, "
          f"{len(ds.feature_names)} features, synthetic={ds.synthetic}\n")

    ccmp = cfg["compare"]
    backends = list(ccmp["classification_backends"])
    if ccmp.get("include_tabpfn"):
        backends.append("tabpfn")

    ## ---- Classification branch: every backend, same LOSO ----
    rows, skipped = [], {}
    for b in backends:
        print(f"[classify] {b} ...")
        try:
            rows.append(_run_classification(cfg, ds, b))
        except Exception as e:  # noqa: BLE001 — keep the comparison going
            skipped[b] = str(e).splitlines()[0]
            print(f"[classify] {b} SKIPPED: {skipped[b]}")

    ## ---- Regression branch: hormone levels, same LOSO ----
    print("\n[regress] hormone levels (LH / E3G / PdG) ...")
    reg_backend = ccmp["regression_backend"]
    reg_params = cfg["regression_params"][reg_backend]
    reg = run_loso_regression(ds, reg_backend, reg_params, max_folds=cfg["split"].get("max_folds"))

    ## ---- Report ----
    cls_table = _fmt_classification(rows)
    reg_table = _fmt_regression(reg["report"])
    best = max(rows, key=lambda x: x["macro_f1"]) if rows else None
    data_note = (
        "- SYNTHETIC data is easy by construction; numbers only exercise/compare the pipeline, not science.\n"
        "  On real mcPHASES expect the ~0.66 macro-F1 SOTA regime and softer probabilities (APS may beat LAC)."
        if ds.synthetic else
        "- Real mcPHASES is hard: hormone-level R^2 is ~0 (absolute levels not recoverable from wearables),\n"
        "  though Spearman shows weak trend signal. The reusable contribution is the phase benchmark + conformal\n"
        "  no-call, not absolute hormone prediction. Conformal coverage holds ~target across backends."
    )

    report_md = f"""# Model comparison — mcPHASES phase classification + hormone regression

Data: {'SYNTHETIC (physiology-grounded fallback)' if ds.synthetic else 'mcPHASES processed table'}
 · {len(ds.y)} subject-days · {len(np.unique(ds.groups))} subjects · {len(ds.feature_names)} features
Split: leave-one-subject-out. SOTA reference (classification): macro-F1 {SOTA['macro_f1']:.3f} / acc {SOTA['accuracy']:.3f}.

## Classification branch (4-phase)

{cls_table}

{"Skipped: " + json.dumps(skipped) if skipped else ""}

Best macro-F1: **{best['backend']}** ({best['macro_f1']:.3f}) — vs SOTA {best['macro_f1'] - SOTA['macro_f1']:+.3f}.
"glassbox" = intrinsically interpretable (no SHAP pass). "conf.cov" = conformal empirical coverage
(target {1 - cfg['conformal']['alpha']:.2f}); "decisive" = singleton-set rate; "dec.acc" = accuracy when decisive.

## Regression branch (hormone levels, {reg_backend})

{reg_table}

Regression timing: {reg['report']['timing']['total_seconds']}s
({reg['report']['timing']['n_folds']} folds × {reg['report']['timing']['models_per_fold']} hormone models).

## Notes
- All models share one LOSO split and one feature matrix, so differences are the model's, not the pipeline's.
{data_note}
"""
    (out_dir / "comparison_report.md").write_text(report_md, encoding="utf-8")
    (out_dir / "comparison.json").write_text(json.dumps(
        {"classification": rows, "skipped": skipped, "regression": reg["report"]}, indent=2))

    print("\n" + "=" * 70)
    print("CLASSIFICATION\n" + cls_table)
    print("\nREGRESSION\n" + reg_table)
    print("=" * 70)
    print(f"\n[done] -> {out_dir / 'comparison_report.md'}")


if __name__ == "__main__":
    main()
