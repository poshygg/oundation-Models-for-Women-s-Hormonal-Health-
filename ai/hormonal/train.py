"""ai/hormonal/train.py — LOSO-CV gradient boosting for mcPHASES phase classification.

Build order was already cleared for this task: prior work is surveyed in
docs/challenges/05-womens-hormonal-health/research-notes.md (Layer 02), which
lands on gradient-boosted trees under a leave-one-subject-out split as both the
literature-backed and pragmatic choice for N~41 tabular multimodal data.

This trains it for real. On a bare machine with no mcPHASES download it runs
end-to-end on a physiology-grounded synthetic dataset (data.py), so the pipeline
is exercisable today; drop the processed table in place and the same command
trains on real data.

Run:
    python -m ai.hormonal.train --config ai/hormonal/configs/phase-clf-baseline.yaml
    python -m ai.hormonal.train --config ... --backend xgboost --max-folds 5
    python -m ai.hormonal.train --config ... --feature-set symptoms_only   # SOTA-input ablation
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GroupShuffleSplit

from .conformal import conformal_oof, conformal_report
from .data import PHASES, build_dataset
from .metrics import full_report

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    import wandb
except ImportError:
    wandb = None  # type: ignore

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore

N_CLASSES = len(PHASES)

## Backends with native SHAP TreeExplainer support. TabPFN is a neural
## in-context learner, so tree-based explainability does not apply to it.
TREE_BACKENDS = {"catboost", "xgboost", "lightgbm"}


## ----------------------------------------------------------------------------
## Model factory
## ----------------------------------------------------------------------------
def make_estimator(backend: str, params: dict):
    if backend == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(**params)
    if backend == "xgboost":
        from xgboost import XGBClassifier
        return XGBClassifier(**params, num_class=N_CLASSES, eval_metric="mlogloss")
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(**params, objective="multiclass", num_class=N_CLASSES, verbose=-1)
    if backend == "tabpfn":
        ## TabPFN v2: a pre-trained transformer that classifies by in-context
        ## learning. "fit" only stores the context; "predict" is a forward pass.
        ## No gradient training happens here.
        from tabpfn import TabPFNClassifier
        return TabPFNClassifier(**params)
    if backend == "ebm":
        ## Explainable Boosting Machine: a glassbox GAM. The fitted per-feature
        ## shape functions ARE the model, so global explanations are free — no
        ## SHAP pass needed (see fit_final_and_explain).
        from interpret.glassbox import ExplainableBoostingClassifier
        return ExplainableBoostingClassifier(**params)
    raise ValueError(f"unknown backend: {backend}")


def _proba_full(est, X: pd.DataFrame) -> np.ndarray:
    """predict_proba re-indexed to the canonical 0..N_CLASSES-1 column order,
    so folds missing a class still return a full-width probability matrix."""
    proba = np.asarray(est.predict_proba(X))
    classes = np.asarray(est.classes_).astype(int)
    if proba.shape[1] == N_CLASSES and list(classes) == list(range(N_CLASSES)):
        return proba
    full = np.zeros((proba.shape[0], N_CLASSES), dtype=float)
    for j, c in enumerate(classes):
        full[:, int(c)] = proba[:, j]
    return full


def fit_fold(backend: str, params: dict, calib_method: str,
             Xtr: pd.DataFrame, ytr: np.ndarray, gtr: np.ndarray, seed: int):
    """Train the base model, then optionally calibrate on a subject-disjoint
    held-in split (prefit). Falls back to the uncalibrated model if calibration
    is off or not fittable for this fold."""
    est = make_estimator(backend, params)

    if calib_method in (None, "none"):
        est.fit(Xtr, ytr)
        return est

    ## Hold out ~20% of TRAINING subjects to fit the calibrator (no leakage:
    ## the calibration subjects are still disjoint from the LOSO test subject).
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    fit_idx, cal_idx = next(gss.split(Xtr, ytr, groups=gtr))
    if len(np.unique(ytr[cal_idx])) < N_CLASSES or len(np.unique(ytr[fit_idx])) < N_CLASSES:
        est.fit(Xtr, ytr)
        return est

    est.fit(Xtr.iloc[fit_idx], ytr[fit_idx])
    try:
        calibrated = CalibratedClassifierCV(est, method=calib_method, cv="prefit")
        calibrated.fit(Xtr.iloc[cal_idx], ytr[cal_idx])
        return calibrated
    except (ValueError, TypeError):
        ## sklearn version without prefit support, or degenerate fold.
        est.fit(Xtr, ytr)
        return est


## ----------------------------------------------------------------------------
## LOSO cross-validation
## ----------------------------------------------------------------------------
def run_loso(cfg: dict, ds) -> dict:
    backend = cfg["model"]["backend"]
    params = dict(cfg["model"]["params"][backend])
    calib_method = cfg["calibration"]["method"]
    seed = int(cfg["experiment"]["seed"])
    max_folds = cfg["split"].get("max_folds")

    subjects = np.unique(ds.groups)
    if max_folds:
        subjects = subjects[: int(max_folds)]

    n = len(ds.y)
    oof_proba = np.full((n, N_CLASSES), np.nan)
    fold_manifest = []
    fold_times = []

    for i, test_subj in enumerate(subjects):
        test_mask = ds.groups == test_subj
        train_mask = ~test_mask
        if max_folds:  # restrict training to the retained subjects too
            train_mask &= np.isin(ds.groups, subjects)

        t0 = time.perf_counter()
        model = fit_fold(
            backend, params, calib_method,
            ds.X[train_mask], ds.y[train_mask], ds.groups[train_mask], seed,
        )
        oof_proba[test_mask] = _proba_full(model, ds.X[test_mask])
        dt = time.perf_counter() - t0
        fold_times.append(dt)
        fold_manifest.append({"fold": i, "test_subject": int(test_subj),
                              "n_test": int(test_mask.sum()), "seconds": round(dt, 3)})
        print(f"[fold {i+1}/{len(subjects)}] held-out subject {test_subj} "
              f"({int(test_mask.sum())} days)  {dt:.2f}s")

    evaluated = ~np.isnan(oof_proba).any(axis=1)
    report = full_report(
        ds.y[evaluated], oof_proba[evaluated], ds.regular[evaluated],
        min_confidence=float(cfg["nocall"]["min_confidence"]),
        nocall_enabled=bool(cfg["nocall"]["enabled"]),
    )

    ## Optional HSMM/HMM temporal smoothing of the OOF predictions — enforces the
    ## cyclic phase order + duration priors, reported alongside the raw classifier.
    scfg = cfg.get("smoothing", {})
    if scfg.get("enabled") and ds.segment is not None:
        from sklearn.metrics import accuracy_score, f1_score

        from .smoothing import smooth_proba
        labels = smooth_proba(oof_proba[evaluated], ds.segment[evaluated],
                              method=scfg.get("method", "hsmm"),
                              skip_prob=float(scfg.get("skip_prob", 0.02)))
        yt = ds.y[evaluated]
        report["smoothed"] = {
            "method": scfg.get("method", "hsmm"),
            "accuracy": float(accuracy_score(yt, labels)),
            "macro_f1": float(f1_score(yt, labels, average="macro", labels=np.arange(N_CLASSES))),
            "per_class_f1": {PHASES[k]: float(v) for k, v in enumerate(
                f1_score(yt, labels, average=None, labels=np.arange(N_CLASSES)))},
        }
    timing = {"total_seconds": round(float(np.sum(fold_times)), 2),
              "mean_fold_seconds": round(float(np.mean(fold_times)), 3),
              "n_folds": len(fold_times)}
    return {"report": report, "oof_proba": oof_proba, "evaluated": evaluated,
            "folds": fold_manifest, "subjects": subjects, "timing": timing}


## ----------------------------------------------------------------------------
## Final model + SHAP explainability
## ----------------------------------------------------------------------------
def fit_final_and_explain(cfg: dict, ds, out_dir: Path) -> dict:
    backend = cfg["model"]["backend"]
    params = dict(cfg["model"]["params"][backend])
    seed = int(cfg["experiment"]["seed"])

    final = make_estimator(backend, params)
    final.fit(ds.X, ds.y)

    ## Persist the model in the backend's native format.
    model_path = out_dir / f"final_{backend}.model"
    try:
        if backend == "catboost":
            final.save_model(str(model_path))
        elif backend == "xgboost":
            final.save_model(str(model_path.with_suffix(".json")))
        else:
            import joblib
            joblib.dump(final, model_path.with_suffix(".joblib"))
    except Exception as e:  # noqa: BLE001 — persistence is best-effort
        print(f"[warn] model save failed: {e}")

    ## Explainability. EBM's shape functions give importances for free; tree
    ## backends use SHAP; anything else (TabPFN) is skipped.
    importances = {}
    if backend == "ebm":
        importances = _ebm_importances(final, out_dir)
    elif cfg["explain"].get("shap", True):
        if backend in TREE_BACKENDS:
            importances = _shap_summary(final, ds, int(cfg["explain"]["shap_max_samples"]), seed, out_dir)
        else:
            print(f"[explain] SHAP TreeExplainer not applicable to '{backend}'; skipping.")
    return {"final_model_path": str(model_path), "shap_top": importances}


def _ebm_importances(final, out_dir: Path) -> dict:
    """Global term importances straight from the fitted EBM — no extra pass.
    Includes any pairwise interaction terms EBM selected."""
    names = list(final.term_names_)
    scores = np.asarray(final.term_importances())
    order = np.argsort(scores)[::-1]
    top = {names[i]: float(scores[i]) for i in order[:25]}
    (out_dir / "ebm_importances.json").write_text(json.dumps(top, indent=2))
    return top


def _shap_summary(final, ds, max_samples: int, seed: int, out_dir: Path) -> dict:
    try:
        import shap
    except ImportError:
        print("[warn] shap not installed; skipping explainability")
        return {}
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ds.X), size=min(max_samples, len(ds.X)), replace=False)
    Xs = ds.X.iloc[idx]
    try:
        explainer = shap.TreeExplainer(final)
        vals = explainer.shap_values(Xs)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] SHAP failed: {e}")
        return {}
    ## Mean |SHAP| aggregated across samples AND classes -> global feature
    ## importance. shap returns per-class arrays whose shape varies by backend
    ## ((n_features,) last for some, a (samples, features, classes) block for
    ## others), so identify the feature axis by matching its length rather than
    ## assuming a position.
    arr = np.abs(np.array(vals))
    n_features = len(ds.feature_names)
    feat_axis = next((ax for ax in range(arr.ndim) if arr.shape[ax] == n_features), arr.ndim - 1)
    reduce_axes = tuple(ax for ax in range(arr.ndim) if ax != feat_axis)
    mean_abs = np.asarray(arr.mean(axis=reduce_axes)).reshape(-1)
    order = np.argsort(mean_abs)[::-1]
    top = {ds.feature_names[i]: float(mean_abs[i]) for i in order[:25]}
    (out_dir / "shap_importance.json").write_text(json.dumps(top, indent=2))
    return top


## ----------------------------------------------------------------------------
## Reporting / artifacts
## ----------------------------------------------------------------------------
SOTA = {"macro_f1": 0.662, "accuracy": 0.676,
        "source": "CatBoost+HSMM, LOSO, self-report only (medRxiv 2026)"}


def write_model_card(cfg: dict, ds, cv: dict, extra: dict, out_dir: Path) -> None:
    r = cv["report"]["overall"]
    backend = cfg["model"]["backend"]
    delta = r["macro_f1"] - SOTA["macro_f1"]
    expl_source = {"ebm": "EBM shape-function term importances (glassbox, no extra pass)"}.get(
        backend, "mean |SHAP| (TreeExplainer)" if backend in TREE_BACKENDS else "n/a")
    conf = cv["report"].get("conformal")
    conf_block = (
        f"""## Conformal no-call ({cfg.get('conformal', {}).get('method', 'aps')}, alpha={conf['method_alpha']})
Distribution-free coverage guarantee, computed on OOF probabilities:
- empirical coverage: **{conf['empirical_coverage']:.3f}** (target {conf['target_coverage']})
- avg prediction-set size: {conf['avg_set_size']:.2f}
- decisive (singleton) rate: {conf['singleton_rate']:.3f}, accuracy when decisive: {conf['singleton_accuracy']:.3f}
- abstain (no-call) rate: {conf['abstain_rate']:.3f}
"""
        if conf else ""
    )
    card = f"""# mcPHASES 4-Phase Menstrual Cycle Classifier

**Challenge 05 — Foundation Models for Women's Hormonal Health (Layer 02).**
Reproducible, explainable classifier over multimodal daily signals (Fitbit + CGM
+ self-report symptoms), evaluated leave-one-subject-out with a conformal no-call.

- Backend: **{backend}**
- Feature set: **{cfg['features']['feature_set']}**  ({len(ds.feature_names)} features)
- Split: **leave-one-subject-out** over {len(cv['subjects'])} subjects
- Data: **{'SYNTHETIC (physiology-grounded fallback)' if ds.synthetic else 'mcPHASES processed table'}**
- Fit+predict time: {cv['timing']['total_seconds']}s over {cv['timing']['n_folds']} folds ({cv['timing']['mean_fold_seconds']}s/fold)

## Results (out-of-fold, LOSO)

| metric | ours | SOTA |
|---|---|---|
| macro-F1 | **{r['macro_f1']:.3f}** | {SOTA['macro_f1']:.3f} |
| accuracy | {r['accuracy']:.3f} | {SOTA['accuracy']:.3f} |
| balanced acc | {r['balanced_accuracy']:.3f} | — |
| AUROC (OvR) | {r['auroc_ovr']:.3f} | — |
| Brier | {r['brier']:.3f} | — |

macro-F1 vs SOTA: **{delta:+.3f}**. SOTA = {SOTA['source']}.

Per-class F1: {json.dumps(cv['report']['per_class_f1'])}

By cycle regularity: {json.dumps(cv['report'].get('by_regularity', {}))}

{conf_block}
## Top features ({expl_source})
{json.dumps(extra.get('shap_top', {}), indent=2)}

## Reproducibility
- seed: {cfg['experiment']['seed']}
- LOSO fold manifest: `loso_folds.json`
- out-of-fold predictions: `oof_predictions.parquet`
- config: `{cfg['experiment']['name']}.yaml`
- generated: {datetime.now(timezone.utc).isoformat()} on {platform.platform()}

## Intended use & limits
Research/education toward open hormonal-health infrastructure. **Not a medical
device.** N is small ({len(cv['subjects'])} subjects) and skewed; irregular-cycle
performance is reported separately and the model abstains (no-call) below the
confidence threshold rather than forcing a phase.
"""
    (out_dir / "MODEL_CARD.md").write_text(card, encoding="utf-8")


def save_artifacts(cfg: dict, ds, cv: dict, extra: dict, out_dir: Path) -> None:
    report_out = dict(cv["report"])
    report_out["timing"] = cv["timing"]
    report_out["backend"] = cfg["model"]["backend"]
    (out_dir / "metrics.json").write_text(json.dumps(report_out, indent=2))
    (out_dir / "loso_folds.json").write_text(json.dumps(cv["folds"], indent=2))
    ev = cv["evaluated"]
    oof = pd.DataFrame(cv["oof_proba"][ev], columns=[f"proba_{p}" for p in PHASES])
    oof.insert(0, "subject_id", ds.groups[ev])
    oof.insert(1, "y_true", ds.y[ev])
    oof.insert(2, "y_pred", cv["oof_proba"][ev].argmax(axis=1))
    try:
        oof.to_parquet(out_dir / "oof_predictions.parquet", index=False)
    except Exception:
        oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    write_model_card(cfg, ds, cv, extra, out_dir)


## ----------------------------------------------------------------------------
## Tracking
## ----------------------------------------------------------------------------
def init_tracking(cfg: dict):
    run = None
    if wandb and os.getenv("WANDB_API_KEY"):
        run = wandb.init(project=cfg["logging"]["wandb_project"],
                         name=cfg["experiment"]["name"],
                         tags=cfg["logging"].get("wandb_tags", []),
                         config=cfg)
    if mlflow:
        mlflow.set_tracking_uri("file:./experiments/mlruns")
        mlflow.set_experiment(cfg["logging"]["mlflow_experiment"])
        mlflow.start_run(run_name=cfg["experiment"]["name"])
    return run


def log_metrics(report: dict):
    flat = {}
    for k, v in report["overall"].items():
        flat[f"overall/{k}"] = v
    if "selective" in report:
        for k, v in report["selective"].items():
            flat[f"selective/{k}"] = v
    if wandb and wandb.run:
        wandb.log(flat)
    if mlflow and mlflow.active_run():
        mlflow.log_metrics({k: v for k, v in flat.items() if isinstance(v, (int, float)) and not np.isnan(v)})


def close_tracking():
    if wandb and wandb.run:
        wandb.finish()
    if mlflow and mlflow.active_run():
        mlflow.end_run()


## ----------------------------------------------------------------------------
## Entry point
## ----------------------------------------------------------------------------
def load_config(path: str, overrides: dict) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if overrides.get("backend"):
        cfg["model"]["backend"] = overrides["backend"]
    if overrides.get("feature_set"):
        cfg["features"]["feature_set"] = overrides["feature_set"]
    if overrides.get("max_folds") is not None:
        cfg["split"]["max_folds"] = overrides["max_folds"]
    return cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--backend", choices=["catboost", "xgboost", "lightgbm", "tabpfn", "ebm"])
    ap.add_argument("--feature-set", dest="feature_set",
                    choices=["all", "symptoms_only", "wearables_only"])
    ap.add_argument("--max-folds", dest="max_folds", type=int)
    ap.add_argument("--no-final", action="store_true", help="skip final-model fit + SHAP")
    args = ap.parse_args()

    cfg = load_config(args.config, vars(args))
    np.random.seed(int(cfg["experiment"]["seed"]))

    out_dir = Path(cfg["logging"]["out_dir"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[data] building dataset (feature_set={cfg['features']['feature_set']}) ...")
    ds = build_dataset(cfg)
    print(f"[data] {len(ds.y)} subject-days, {len(np.unique(ds.groups))} subjects, "
          f"{len(ds.feature_names)} features, synthetic={ds.synthetic}")

    init_tracking(cfg)
    print(f"[train] LOSO CV with {cfg['model']['backend']} ...")
    cv = run_loso(cfg, ds)

    r = cv["report"]["overall"]
    print(f"\n== LOSO result ==\n  macro-F1={r['macro_f1']:.3f} (SOTA {SOTA['macro_f1']:.3f}, "
          f"{r['macro_f1'] - SOTA['macro_f1']:+.3f})  acc={r['accuracy']:.3f}  "
          f"bal-acc={r['balanced_accuracy']:.3f}  AUROC={r['auroc_ovr']:.3f}  Brier={r['brier']:.3f}")
    if "selective" in cv["report"]:
        s = cv["report"]["selective"]
        print(f"  no-call: coverage={s['coverage']:.3f}  selective-acc={s['selective_accuracy']:.3f}")
    t = cv["timing"]
    print(f"  timing: {t['total_seconds']}s total over {t['n_folds']} folds "
          f"({t['mean_fold_seconds']}s/fold, fit+predict)")
    if "smoothed" in cv["report"]:
        s = cv["report"]["smoothed"]
        print(f"  +{s['method']} smoothing: acc={s['accuracy']:.3f} macro-F1={s['macro_f1']:.3f} "
              f"(raw macro-F1={r['macro_f1']:.3f})")

    ## Conformal no-call layer (distribution-free coverage guarantee) on the OOF
    ## probabilities. Replaces the ad-hoc confidence threshold when enabled.
    ccfg = cfg.get("conformal", {})
    if ccfg.get("enabled"):
        ev = cv["evaluated"]
        alpha = float(ccfg.get("alpha", 0.1))
        mask, qhats = conformal_oof(
            cv["oof_proba"][ev], ds.y[ev], ds.groups[ev],
            alpha=alpha, method=ccfg.get("method", "aps"),
            n_splits=int(ccfg.get("n_splits", 5)),
        )
        crep = conformal_report(mask, ds.y[ev], alpha)
        cv["report"]["conformal"] = crep
        print(f"  conformal[{ccfg.get('method','aps')}, alpha={alpha}]: "
              f"coverage={crep['empirical_coverage']:.3f} (target {crep['target_coverage']}), "
              f"avg set={crep['avg_set_size']:.2f}, decisive(singleton)={crep['singleton_rate']:.3f}, "
              f"singleton-acc={crep['singleton_accuracy']:.3f}")

    extra = {}
    if not args.no_final:
        print("[explain] fitting final model + SHAP ...")
        extra = fit_final_and_explain(cfg, ds, out_dir)

    save_artifacts(cfg, ds, cv, extra, out_dir)
    log_metrics(cv["report"])
    close_tracking()
    print(f"\n[done] artifacts -> {out_dir}")


if __name__ == "__main__":
    main()
