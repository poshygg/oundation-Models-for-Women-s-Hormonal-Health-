"""Shared utilities for the paper harness scripts."""
from __future__ import annotations

import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError as e:  # pragma: no cover
    print("[!] PyYAML 이 필요합니다: pip install -r requirements.txt", file=sys.stderr)
    raise

ROOT = Path(__file__).resolve().parents[1]  # papers/
CONFIG_PATH = ROOT / "configs" / "harness.yaml"


def load_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_logger(name: str, log_file: str | None = None) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s :: %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_file:
        log_path = ROOT / "logs" / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


_SLUG_RE = re.compile(r"[^\w\-]+", re.UNICODE)


def slugify(text: str, maxlen: int = 80) -> str:
    """ASCII 슬러그. 한글/특수문자 제거."""
    ascii_only = text.encode("ascii", "ignore").decode("ascii")
    ascii_only = ascii_only.replace(" ", "_")
    slug = _SLUG_RE.sub("", ascii_only).strip("_-")
    slug = re.sub(r"_+", "_", slug)
    return slug[:maxlen] or "paper"


def read_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
