from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore

## Per-phase caps in minutes — keep in sync with the table in docs/time_budgeting.md
PHASE_CAPS = {
    1: {"single_min": 30, "cumulative_min": None},
    2: {"single_min": 120, "cumulative_min": 360},
    3: {"single_min": 0, "cumulative_min": 0},
    4: {"single_min": 0, "cumulative_min": 0},
}

HACKATHON_HOURS = 24


def estimate_seconds(steps_per_epoch: float, epochs: float, step_seconds: float) -> float:
    return steps_per_epoch * epochs * step_seconds


def remaining_seconds(started_at: datetime, now: datetime) -> float:
    elapsed = (now - started_at).total_seconds()
    return HACKATHON_HOURS * 3600 - elapsed


def verdict(estimate_s: float, phase: int, remaining_s: float | None) -> str:
    cap = PHASE_CAPS[phase]
    cap_s = cap["single_min"] * 60
    lines = [
        f"Estimated: {estimate_s / 60:.1f} min  |  Phase {phase} single-run cap: {cap_s / 60:.0f} min"
    ]

    if remaining_s is not None:
        lines.append(f"Hackathon time remaining: {remaining_s / 3600:.1f} h")
        if estimate_s > remaining_s:
            lines.append("NO-GO — estimate exceeds remaining time. Switch to an alternative now.")
            return "\n".join(lines)

    if cap_s == 0:
        lines.append("NO-GO — no new training in this phase.")
    elif estimate_s <= cap_s:
        lines.append("GO — within the cap.")
    else:
        over_min = (estimate_s - cap_s) / 60
        lines.append(
            f"SCOPE DOWN — {over_min:.1f} min over the cap. "
            "Re-estimate in order: data subset -> fewer epochs -> smaller model -> switch to LoRA."
        )
    return "\n".join(lines)


def main() -> None:
    ## force utf-8 so this prints cleanly regardless of the host console codepage
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if load_dotenv:
        load_dotenv()

    parser = argparse.ArgumentParser(description="Time budget gate check before starting a training run")
    parser.add_argument("--steps-per-epoch", type=float, required=True)
    parser.add_argument("--epochs", type=float, required=True)
    parser.add_argument(
        "--step-seconds",
        type=float,
        required=True,
        help="average seconds per step, measured from 5-10 steps of the real training loop",
    )
    parser.add_argument("--phase", type=int, default=2, choices=sorted(PHASE_CAPS))
    parser.add_argument(
        "--started-at",
        type=str,
        default=os.getenv("HACKATHON_START") or None,
        help="hackathon start time, ISO format. Defaults to HACKATHON_START in .env",
    )
    args = parser.parse_args()

    estimate_s = estimate_seconds(args.steps_per_epoch, args.epochs, args.step_seconds)
    remaining_s = None
    if args.started_at:
        remaining_s = remaining_seconds(datetime.fromisoformat(args.started_at), datetime.now())

    print(verdict(estimate_s, args.phase, remaining_s))


if __name__ == "__main__":
    main()
