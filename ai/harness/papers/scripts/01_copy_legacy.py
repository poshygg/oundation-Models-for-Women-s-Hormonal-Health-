"""01_copy_legacy.py — trading/papers/ (또는 configs/harness.yaml 의 legacy_source) 를
Hackathon_2026Summer/ai/harness/papers/legacy/ 로 복사.

- PDF/MD 파일만 복사, 원본 하위 폴더 구조 유지
- 이미 존재하면 skip (--overwrite 지정 시 덮어씀)
- --source 로 소스 경로 오버라이드 가능
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, get_logger, load_config  # noqa: E402

log = get_logger("copy_legacy", "01_copy_legacy.log")

VALID_EXT = {".pdf", ".md"}


def iter_source_files(src: Path):
    for p in src.rglob("*"):
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            yield p


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default=None, help="복사 소스 경로 오버라이드")
    parser.add_argument("--overwrite", action="store_true", help="기존 파일 덮어쓰기")
    args = parser.parse_args()

    cfg = load_config()
    source = Path(args.source) if args.source else Path(cfg["paths"]["legacy_source"])
    dest = ROOT / cfg["paths"]["legacy"]
    dest.mkdir(parents=True, exist_ok=True)

    log.info(f"소스: {source}")
    log.info(f"대상: {dest}")

    if not source.exists():
        log.error(f"소스 폴더가 존재하지 않습니다: {source}")
        log.error("--source 로 다른 경로를 지정하거나 configs/harness.yaml 의 legacy_source 를 수정하세요.")
        return 2

    files = list(iter_source_files(source))
    if not files:
        log.warning("복사할 PDF/MD 파일이 없습니다.")
        return 0

    copied = skipped = failed = 0
    for src_file in tqdm(files, desc="복사 중", unit="file"):
        rel = src_file.relative_to(source)
        dst_file = dest / rel
        dst_file.parent.mkdir(parents=True, exist_ok=True)

        if dst_file.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            shutil.copy2(src_file, dst_file)
            copied += 1
        except Exception as e:
            log.error(f"복사 실패 {src_file}: {e}")
            failed += 1

    log.info(f"완료 — 복사 {copied}, 스킵 {skipped}, 실패 {failed} (총 {len(files)})")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
