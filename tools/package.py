#!/usr/bin/env python3
"""Build the submission bundle: code.zip, plus a pre-flight check of every deliverable.

The submission asks for three files. Two of them are easy to get subtly wrong, so
this script verifies rather than assumes:

* **code.zip** - the runnable solution. The brief explicitly excludes virtualenvs,
  node_modules, build artifacts, the ``data/`` corpus and the ``dataset/`` folder,
  so those are filtered here and the resulting archive is listed back for review.
  Secrets are refused outright: if a ``.env`` or anything key-shaped would land in
  the archive, the build fails rather than shipping it.
* **output.csv** - checked against the full submission contract.
* **log.txt** - the transcript; its size and entry count are reported so a
  truncated or empty log is caught before upload rather than after.

    python tools/package.py
    python tools/package.py --out dist/code.zip
"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "code"))

# Everything the submission brief tells us to leave out, plus caches and secrets.
EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "env", "node_modules", "__pycache__", ".pytest_cache",
    ".mypy_cache", ".ruff_cache", ".cache", "runs", "dataset", "data", "dist",
    ".idea", ".vscode", ".DS_Store",
}
EXCLUDE_SUFFIXES = {".pyc", ".pyo", ".pyd", ".so", ".zip", ".log"}
EXCLUDE_NAMES = {".env", ".env.local", ".DS_Store"}

# What must be inside the archive for it to be runnable.
REQUIRED = (
    "code/main.py",
    "code/requirements.txt",
    "code/README.md",
    "code/router/pipeline.py",
    "code/router/schema.py",
    "code/judgments/expert_judgments.jsonl",
    "code/tests/test_router.py",
    "AGENTS.md",
)

_SECRET_SHAPES = (
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"sk-ant-[0-9A-Za-z_\-]{20,}"),
    re.compile(r"sk-[0-9A-Za-z]{32,}"),
    re.compile(r"gh[pousr]_[0-9A-Za-z]{20,}"),
)


def should_include(path: Path) -> bool:
    rel = path.relative_to(REPO)
    if any(part in EXCLUDE_DIRS for part in rel.parts):
        return False
    if path.name in EXCLUDE_NAMES or path.suffix in EXCLUDE_SUFFIXES:
        return False
    return path.is_file()


def scan_for_secrets(paths: list[Path]) -> list[str]:
    """Refuse to ship a key. Text files only; binaries are skipped."""
    findings: list[str] = []
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for pattern in _SECRET_SHAPES:
            if pattern.search(text):
                findings.append(str(path.relative_to(REPO)))
                break
    return findings


def build(out: Path) -> tuple[list[Path], int]:
    members = sorted(
        p for p in REPO.rglob("*")
        if should_include(p) and (p.parts[len(REPO.parts):][0] in {"code", "tools"}
                                  or p.parent == REPO)
    )
    leaked = scan_for_secrets(members)
    if leaked:
        raise SystemExit(
            "REFUSING TO PACKAGE - credential-shaped strings found in:\n  "
            + "\n  ".join(leaked)
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for path in members:
            zf.write(path, path.relative_to(REPO))
    return members, out.stat().st_size


def check_output_csv() -> bool:
    from main import _validate  # noqa: PLC0415
    from router.config import PATHS  # noqa: PLC0415
    from router.context_store import ContextStore  # noqa: PLC0415

    ok, problems = _validate(PATHS.output_csv, ContextStore.load())
    print(f"\n  output.csv       {'PASS' if ok else 'FAIL'}  {PATHS.output_csv}")
    for problem in problems[:10]:
        print(f"      - {problem}")
    return ok


def check_transcript() -> bool:
    log = Path.home() / "hackerrank_orchestrate_august26" / "log.txt"
    if not log.is_file():
        print(f"\n  transcript       FAIL  missing: {log}")
        return False
    text = log.read_text(encoding="utf-8")
    entries = text.count("\n## [")+ text.startswith("## [")
    has_agreement = "AGREEMENT RECORDED:" in text
    leaked = any(p.search(text) for p in _SECRET_SHAPES)
    ok = entries >= 3 and has_agreement and not leaked
    print(f"\n  transcript       {'PASS' if ok else 'FAIL'}  {log}")
    print(f"      {len(text):,} chars, {entries} entries, "
          f"agreement={'yes' if has_agreement else 'NO'}, "
          f"secrets={'LEAKED' if leaked else 'none'}")
    return ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "code.zip"))
    args = parser.parse_args(argv)

    out = Path(args.out)
    members, size = build(out)

    print(f"  code.zip         BUILT {out}  ({size / 1024:.0f} KB, {len(members)} files)")
    names = {str(p.relative_to(REPO)) for p in members}
    missing = [r for r in REQUIRED if r not in names]
    if missing:
        print("      MISSING REQUIRED ENTRIES:")
        for m in missing:
            print(f"      - {m}")
    else:
        print("      all required entries present")

    print("\n  archive contents:")
    for path in members:
        print(f"      {path.relative_to(REPO)}")

    csv_ok = check_output_csv()
    log_ok = check_transcript()

    ready = csv_ok and log_ok and not missing
    print("\n" + "=" * 66)
    print("  SUBMISSION READY" if ready else "  NOT READY - fix the failures above")
    print("=" * 66)
    return 0 if ready else 1


if __name__ == "__main__":
    sys.exit(main())
