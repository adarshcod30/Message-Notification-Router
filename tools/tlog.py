#!/usr/bin/env python3
"""Append-only transcript logger for HackerRank Orchestrate (August 2026).

Implements the log contract defined in AGENTS.md sections 2 and 5:

  * The log lives OUTSIDE the repo at ``$HOME/hackerrank_orchestrate_august26/log.txt``
    so it survives branch switches, worktrees, and cleanup.
  * Entries are append-only. Nothing already written is ever rewritten.
  * Secrets are redacted before anything touches disk.

Usage
-----
    python3 tools/tlog.py onboard   --agent claude-code --language py
    python3 tools/tlog.py session
    python3 tools/tlog.py turn      --title "..." --prompt-file p.txt --summary-file s.txt \
                                    --action "..." --action "..."
    python3 tools/tlog.py note      --title "..." --summary-file s.txt --action "..."

``turn`` is the AGENTS.md 5.2 per-user-turn entry. ``note`` uses the identical
block shape for engineering milestones recorded between user turns, so the
transcript reflects the real development narrative rather than a handful of
sparse checkpoints.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

LOG_DIR_NAME = "hackerrank_orchestrate_august26"
LOG_FILE_NAME = "log.txt"

# Submission deadline for the August 2026 edition (Asia/Kolkata).
CHALLENGE_END = datetime(2026, 8, 2, 18, 19, tzinfo=timezone(timedelta(hours=5, minutes=30)))

# Patterns scrubbed from every prompt/summary before it is written to disk.
# Ordered most-specific first so a narrow rule wins over the generic key= rule.
_SECRET_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED_GOOGLE_API_KEY]"),
    (re.compile(r"sk-ant-[0-9A-Za-z_\-]{20,}"), "[REDACTED_ANTHROPIC_API_KEY]"),
    (re.compile(r"sk-[0-9A-Za-z]{32,}"), "[REDACTED_OPENAI_API_KEY]"),
    (re.compile(r"gh[pousr]_[0-9A-Za-z]{20,}"), "[REDACTED_GITHUB_TOKEN]"),
    (re.compile(r"xox[abposr]-[0-9A-Za-z\-]{10,}"), "[REDACTED_SLACK_TOKEN]"),
    (re.compile(r"-{5}BEGIN[^-]*PRIVATE KEY-{5}.*?-{5}END[^-]*PRIVATE KEY-{5}", re.S), "[REDACTED_PRIVATE_KEY]"),
    # "Bearer <token>" / "Basic <blob>" use a bare space rather than : or =.
    (re.compile(r"(?i)\b(bearer|basic)\s+[A-Za-z0-9._\-+/=]{8,}"), r"\1 [REDACTED]"),
    (re.compile(r"(?i)\b(api[_-]?key|token|secret|password|passwd)\b\s*[:=]\s*\S+"), r"\1=[REDACTED]"),
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def log_path() -> Path:
    """Resolve the log path from the platform home directory (never hardcoded)."""
    return Path.home() / LOG_DIR_NAME / LOG_FILE_NAME


def redact(text: str) -> str:
    """Strip anything that looks like a credential. Applied to all free text."""
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def time_remaining() -> str:
    delta = CHALLENGE_END - datetime.now(timezone.utc)
    if delta.total_seconds() <= 0:
        return "0d 0h 0m (deadline passed)"
    return f"{delta.days}d {delta.seconds // 3600}h {(delta.seconds % 3600) // 60}m"


def _git(*args: str, cwd: Path) -> str:
    try:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=10, check=False
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def repo_root() -> Path:
    """Repo root, resolved from this file's location so it works from any cwd."""
    return Path(__file__).resolve().parent.parent


def branch(root: Path) -> str:
    return _git("rev-parse", "--abbrev-ref", "HEAD", cwd=root)


def append(block: str) -> None:
    """Append one UTF-8 block with \\n endings. Creates the parent dir if missing."""
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(block.rstrip("\n") + "\n\n")
    print(f"[tlog] appended {len(block)} chars -> {path}")


def context_block(agent: str, root: Path, parent: str) -> str:
    return (
        "Context:\n"
        f"tool={agent}\n"
        f"branch={branch(root)}\n"
        f"repo_root={root}\n"
        f"worktree=main\n"
        f"parent_agent={parent}\n"
    )


def read_text_arg(inline: str | None, from_file: str | None) -> str:
    """Read a text argument either inline or from a file (avoids shell quoting hell)."""
    if from_file:
        return Path(from_file).read_text(encoding="utf-8").strip()
    return (inline or "").strip()


# --------------------------------------------------------------------------- #
# Entry writers
# --------------------------------------------------------------------------- #

def cmd_onboard(args: argparse.Namespace) -> None:
    root = repo_root()
    path = log_path()
    if path.exists() and f"AGREEMENT RECORDED: {root}" in path.read_text(encoding="utf-8"):
        print("[tlog] agreement already recorded for this repo root; skipping")
        return
    append(
        f"## [{now_iso()}] ONBOARDING COMPLETE\n\n"
        f"AGREEMENT RECORDED: {root}\n"
        f"Agent: {args.agent}\n"
        f"Language: {args.language}\n"
        f"System Time: {now_iso()}\n"
        f"Time Remaining: {time_remaining()}\n"
    )


def cmd_session(args: argparse.Namespace) -> None:
    root = repo_root()
    append(
        f"## [{now_iso()}] SESSION START\n\n"
        f"Agent: {args.agent}\n"
        f"Repo Root: {root}\n"
        f"Branch: {branch(root)}\n"
        f"Worktree: main\n"
        f"Parent Agent: {args.parent}\n"
        f"Language: {args.language}\n"
        f"Time Remaining: {time_remaining()}\n"
    )


def _entry(args: argparse.Namespace, include_prompt: bool) -> None:
    root = repo_root()
    summary = redact(read_text_arg(args.summary, args.summary_file))
    actions = "\n".join(f"* {redact(a)}" for a in (args.action or [])) or "* (none)"

    body = f"## [{now_iso()}] {args.title[:80]}\n\n"
    if include_prompt:
        prompt = redact(read_text_arg(args.prompt, args.prompt_file))
        body += f"User Prompt (verbatim, secrets redacted):\n{prompt}\n\n"
    body += (
        f"Agent Response Summary:\n{summary}\n\n"
        f"Actions:\n{actions}\n\n"
        f"{context_block(args.agent, root, args.parent)}"
        f"Time Remaining: {time_remaining()}\n"
    )
    append(body)


def cmd_turn(args: argparse.Namespace) -> None:
    _entry(args, include_prompt=True)


def cmd_note(args: argparse.Namespace) -> None:
    _entry(args, include_prompt=False)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agent", default="claude-code", help="agent name recorded in Context")
    parser.add_argument("--parent", default="none", help="parent agent name, or 'none'")
    parser.add_argument("--language", default="py", help="js | ts | py | custom:<name>")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("onboard", help="record the AGENTS.md 3.4 agreement block").set_defaults(func=cmd_onboard)
    sub.add_parser("session", help="record an AGENTS.md 5.1 SESSION START entry").set_defaults(func=cmd_session)

    for name, help_text, func in (
        ("turn", "record an AGENTS.md 5.2 per-user-turn entry", cmd_turn),
        ("note", "record an engineering milestone between user turns", cmd_note),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--title", required=True)
        p.add_argument("--prompt")
        p.add_argument("--prompt-file")
        p.add_argument("--summary")
        p.add_argument("--summary-file")
        p.add_argument("--action", action="append", help="repeatable; one bullet per action")
        p.set_defaults(func=func)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
