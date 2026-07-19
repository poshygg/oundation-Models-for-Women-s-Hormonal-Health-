"""ai/hormonal/explain.py — SHAP/conformal outputs -> clinical-context report.

Roadmap Block 3: translate the model's feature weights and one participant's
measured trends into a rigorous, human-readable physiological report. It consumes
the REAL pipeline artifacts already in data/processed/ (the B1 run):

  * shap_importance_B1.csv         -> global feature drivers
  * oof_B1_conformal.parquet       -> per-day predictions, probabilities, conformal set / no-call
  * mcphases_daily.parquet         -> per-day hormones (LH / estrogen / PdG), symptoms, wearables

It builds a structured evidence pack for one subject, then asks an LLM to write
the report. Provider order: OpenAI gpt-4o (the roadmap's OpenAI-credits path),
then Anthropic, then a deterministic template so a real report is produced even
with no API key. Every number in the report comes from the evidence pack — the
LLM is instructed to interpret, not invent.

    python -m ai.hormonal.explain --subject 1
    python -m ai.hormonal.explain --subject 1 --interval 2022 --provider openai
    python -m ai.hormonal.explain --subject 5 --provider template   # offline
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PHASES = ["Menstrual", "Follicular", "Fertility", "Luteal"]

## Short physiological gloss for feature families (substring match). Used to
## ground both the template report and the LLM prompt. Kept factual and hedged.
PHYSIOLOGY = {
    "flow_volume": "menstrual bleeding volume — the definitional signal of the menstrual phase",
    "flow_color": "menstrual bleeding color — co-varies with flow, marks menstruation",
    "cramps": "dysmenorrhea, typically peri-menstrual",
    "sorebreasts": "breast tenderness, classically luteal (progesterone-driven)",
    "bloating": "fluid retention, common in the luteal phase",
    "moodswing": "affective symptom, often luteal (premenstrual)",
    "foodcravings": "appetite change, often luteal",
    "resting_heart_rate": "resting HR rises in the luteal phase (progesterone thermogenesis)",
    "rhr": "resting HR rises in the luteal phase",
    "hr_bpm": "mean heart rate; elevated luteally",
    "hrv_rmssd": "parasympathetic HRV; tends to fall in the luteal phase",
    "hrv_high_frequency": "vagal HRV component; lower luteally",
    "hrv_low_frequency": "HRV low-frequency power",
    "breathing_rate": "respiratory rate; rises in the luteal phase",
    "resp": "respiratory rate; rises in the luteal phase",
    "temperature": "skin/body temperature rises ~0.3 C after ovulation (luteal)",
    "wrist_temp": "wrist temperature rises after ovulation",
    "cgm_glucose": "glucose level/variability; shifts with luteal insulin resistance",
    "glucose": "glucose dynamics shift across the cycle",
    "vo2": "cardiorespiratory fitness proxy",
    "sleep": "sleep architecture varies across the cycle",
    "estrogen": "estrogen (E3G) peaks late-follicular / peri-ovulation",
    "lh": "LH surges at ovulation",
    "pdg": "PdG (progesterone) peaks in the luteal phase",
}


def _gloss(feature: str) -> str:
    f = feature.lower()
    for key, txt in PHYSIOLOGY.items():
        if key in f:
            return txt
    return ""


def _zsuffix_note(feature: str) -> str:
    return " (z-scored within subject)" if feature.endswith("__z") else ""


## ----------------------------------------------------------------------------
## Evidence pack
## ----------------------------------------------------------------------------
def build_evidence(daily: pd.DataFrame, oof: pd.DataFrame, shap: pd.DataFrame,
                   subject: int, interval: int | None, top_k: int) -> dict:
    keys = ["id", "study_interval", "day_in_study"]
    df = daily.merge(oof, on=keys, suffixes=("", "_oof"), how="inner")
    sub = df[df["id"] == subject]
    if interval is not None:
        sub = sub[sub["study_interval"] == interval]
    if sub.empty:
        raise SystemExit(f"no rows for subject={subject} interval={interval}")

    phase_col = "phase" if "phase" in sub.columns else "phase_oof"
    y_true = sub[phase_col].astype(str)
    y_pred = sub["pred"].astype(str)
    valid = y_true.isin(PHASES)

    perf = {
        "n_days": int(len(sub)),
        "accuracy": float((y_pred[valid] == y_true[valid]).mean()) if valid.any() else float("nan"),
        "no_call_rate": float(sub["no_call"].mean()) if "no_call" in sub else float("nan"),
        "coverage": float(sub["covered"].mean()) if "covered" in sub else float("nan"),
        "avg_set_size": float(sub["set_size"].mean()) if "set_size" in sub else float("nan"),
        "intervals": sorted(int(x) for x in sub["study_interval"].unique()),
    }

    ## Phase makeup (actual vs predicted counts).
    phase_counts = {
        "actual": {p: int((y_true == p).sum()) for p in PHASES},
        "predicted": {p: int((y_pred == p).sum()) for p in PHASES},
    }

    ## Hormone picture by actual phase (drop hormones that are all-NaN for this
    ## subject — some intervals dropped assays).
    hormones = {}
    for h in ("lh", "estrogen", "pdg"):
        if h in sub.columns and sub[h].notna().any():
            by = sub.groupby(phase_col)[h].mean()
            by = {p: round(float(by[p]), 1) for p in PHASES if p in by.index and pd.notna(by[p])}
            if by:
                peak = max(by, key=by.get)
                hormones[h] = {"by_phase": by, "peak_phase": peak}

    ## Global drivers (SHAP), annotated with physiology.
    shap = shap.rename(columns={shap.columns[0]: "feature"})
    drivers = []
    for _, row in shap.sort_values("mean_abs_shap", ascending=False).head(top_k).iterrows():
        feat = str(row["feature"])
        drivers.append({
            "feature": feat,
            "mean_abs_shap": round(float(row["mean_abs_shap"]), 4),
            "physiology": _gloss(feat) + _zsuffix_note(feat),
        })

    return {
        "subject": subject,
        "interval": interval,
        "dataset": {"name": "mcPHASES", "phases": PHASES,
                    "note": "real, hormone-verified labels; leave-one-subject-out out-of-fold predictions"},
        "performance": perf,
        "phase_counts": phase_counts,
        "hormones": hormones,
        "global_drivers": drivers,
    }


def render_evidence_text(ev: dict) -> str:
    p = ev["performance"]
    lines = [
        f"SUBJECT {ev['subject']} (mcPHASES, intervals {p['intervals']}), {p['n_days']} days.",
        f"Model out-of-fold accuracy on this subject: {p['accuracy']:.3f}. "
        f"Conformal no-call rate: {p['no_call_rate']:.3f}; empirical coverage: {p['coverage']:.3f}; "
        f"average prediction-set size: {p['avg_set_size']:.2f}.",
        "",
        "PHASE COUNTS (actual vs predicted): "
        + ", ".join(f"{ph} {ev['phase_counts']['actual'][ph]}/{ev['phase_counts']['predicted'][ph]}"
                    for ph in PHASES),
        "",
        "HORMONE LEVELS BY ACTUAL PHASE:",
    ]
    if ev["hormones"]:
        for h, d in ev["hormones"].items():
            byp = ", ".join(f"{ph}={v}" for ph, v in d["by_phase"].items())
            lines.append(f"  {h}: {byp}  (peak: {d['peak_phase']})")
    else:
        lines.append("  (no hormone assays available for this subject/interval)")
    lines += ["", "TOP MODEL DRIVERS (global mean|SHAP|, higher = more influential):"]
    for d in ev["global_drivers"]:
        gloss = f" — {d['physiology']}" if d["physiology"] else ""
        lines.append(f"  {d['feature']}: {d['mean_abs_shap']}{gloss}")
    return "\n".join(lines)


## ----------------------------------------------------------------------------
## Report generation
## ----------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a reproductive-endocrinology research assistant. Write a rigorous, "
    "evidence-bound interpretation of a menstrual-phase classifier's outputs for ONE "
    "study participant, for a technical audience. Rules: ground every statement in the "
    "provided feature importances and measured trends; clearly separate MODEL BEHAVIOR "
    "(what drives predictions) from PHYSIOLOGY (what is biologically known); foreground "
    "uncertainty, including the model's own accuracy and its conformal no-call rate; do "
    "NOT give medical advice, diagnoses, or recommendations; do NOT invent any numbers "
    "beyond those provided. Keep it concise and structured."
)

USER_TEMPLATE = (
    "Write a clinical-context report with these sections:\n"
    "1. Summary (2-3 sentences)\n"
    "2. How the model reads this participant (map top drivers to physiology)\n"
    "3. Per-phase hormone picture\n"
    "4. Reliability & uncertainty (interpret accuracy, coverage, no-call)\n"
    "5. Caveats\n\n"
    "EVIDENCE:\n{evidence}\n"
)


def generate_report(ev: dict, provider: str, model: str | None) -> tuple[str, str]:
    evidence = render_evidence_text(ev)
    order = {"auto": ["openai", "anthropic", "template"],
             "openai": ["openai"], "anthropic": ["anthropic"], "template": ["template"]}[provider]
    for prov in order:
        try:
            if prov == "openai":
                return _openai_report(evidence, model or "gpt-4o"), "openai"
            if prov == "anthropic":
                return _anthropic_report(evidence, model or "claude-sonnet-5"), "anthropic"
            return _template_report(ev, evidence), "template"
        except Exception as e:  # noqa: BLE001 — fall through to the next provider
            print(f"[explain] provider '{prov}' unavailable: {str(e).splitlines()[0]}")
    return _template_report(ev, evidence), "template"


def _openai_report(evidence: str, model: str) -> str:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model, temperature=0.3,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": USER_TEMPLATE.format(evidence=evidence)}],
    )
    return resp.choices[0].message.content


def _anthropic_report(evidence: str, model: str) -> str:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model, max_tokens=1500, temperature=0.3, system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": USER_TEMPLATE.format(evidence=evidence)}],
    )
    return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")


def _template_report(ev: dict, evidence: str) -> str:
    """Deterministic, offline report assembled directly from the evidence pack —
    no LLM. Meaningful on its own and used as the fallback."""
    p = ev["performance"]
    top = ev["global_drivers"][:6]
    driver_lines = "\n".join(
        f"- **{d['feature']}** (|SHAP| {d['mean_abs_shap']})"
        + (f": {d['physiology']}" if d["physiology"] else "")
        for d in top
    )
    horm_lines = "\n".join(
        f"- **{h}** peaks in the **{d['peak_phase']}** phase — " + ", ".join(f"{ph}: {v}" for ph, v in d["by_phase"].items())
        for h, d in ev["hormones"].items()
    ) or "- No hormone assays available for this participant/interval."
    return f"""# Participant {ev['subject']} — model interpretation (template)

_Deterministic report from the evidence pack (no LLM). Set OPENAI_API_KEY or
ANTHROPIC_API_KEY for a narrative version._

## Summary
Over {p['n_days']} tracked days, the classifier reached **{p['accuracy']:.1%}** out-of-fold
accuracy for this participant. The conformal layer abstained (no-call) on
**{p['no_call_rate']:.1%}** of days while holding **{p['coverage']:.1%}** empirical coverage
(avg prediction-set size {p['avg_set_size']:.2f}), i.e. it declines to commit when the
phases are physiologically ambiguous rather than forcing a label.

## How the model reads this participant (top drivers → physiology)
{driver_lines}

## Per-phase hormone picture
{horm_lines}

## Reliability & uncertainty
Accuracy on this subject is {p['accuracy']:.3f}; the model is decisive on only
{1 - p['no_call_rate']:.1%} of days. Treat single-day phase calls as provisional —
the conformal set (avg size {p['avg_set_size']:.2f}) is the honest output.

## Caveats
Model behavior ≠ physiology; SHAP shows what drives predictions, not causation.
Not a medical device; no diagnosis or advice. Some intervals dropped hormone
assays, so per-phase hormone values may be missing.

---
### Evidence pack
```
{evidence}
```
"""


## ----------------------------------------------------------------------------
def main() -> None:
    ## Windows consoles default to cp949 here; keep unicode (em-dash, etc.) from
    ## crashing stdout while the file itself is always written UTF-8.
    try:
        import sys
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--daily", default="data/processed/mcphases_daily.parquet")
    ap.add_argument("--oof", default="data/processed/oof_B1_conformal.parquet")
    ap.add_argument("--shap", default="data/processed/shap_importance_B1.csv")
    ap.add_argument("--subject", type=int, required=True)
    ap.add_argument("--interval", type=int, default=None)
    ap.add_argument("--provider", default="auto", choices=["auto", "openai", "anthropic", "template"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--top-k", dest="top_k", type=int, default=15)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    daily = pd.read_parquet(args.daily)
    oof = pd.read_parquet(args.oof)
    shap = pd.read_csv(args.shap)

    ev = build_evidence(daily, oof, shap, args.subject, args.interval, args.top_k)
    report, used = generate_report(ev, args.provider, args.model)

    out_dir = Path("experiments/hormonal/reports")
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"subject_{args.subject}" + (f"_{args.interval}" if args.interval else "")
    out_path = Path(args.out) if args.out else out_dir / f"{stem}.md"
    out_path.write_text(report, encoding="utf-8")
    (out_dir / f"{stem}_evidence.json").write_text(json.dumps(ev, indent=2))

    print(f"[explain] provider used: {used}")
    print(f"[explain] report -> {out_path}")
    print("\n" + "=" * 70)
    print(report)


if __name__ == "__main__":
    main()
