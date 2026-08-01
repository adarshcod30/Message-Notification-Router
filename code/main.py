#!/usr/bin/env python3
"""Message Notification Router - command line entry point.

    python code/main.py run                 # route dataset/messages.csv -> output.csv
    python code/main.py evaluate            # score against the labelled sample rows
    python code/main.py ablate              # baseline vs full pipeline
    python code/main.py compare -m a -m b   # score across judge models
    python code/main.py validate            # check output.csv satisfies the spec

Run from the repository root. Configuration is environment-driven; see
``code/README.md`` or ``router/config.py`` for the full list.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Allow `python code/main.py` from the repo root without installing a package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from evaluation.evaluate import EvalResult, evaluate_samples, write_result  # noqa: E402
from evaluation.report import render_comparison, render_report  # noqa: E402
from router.config import PATHS  # noqa: E402
from router.context_store import ContextStore  # noqa: E402
from router.pipeline import RouterPipeline, write_audit, write_output  # noqa: E402
from router.schema import OUTPUT_COLUMNS  # noqa: E402

log = logging.getLogger("router")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # The HTTP layer is chatty at INFO and drowns the pipeline's own progress.
    logging.getLogger("urllib3").setLevel(logging.WARNING)


# --------------------------------------------------------------------------- #

def cmd_run(args: argparse.Namespace) -> int:
    store = ContextStore.load()
    log.info("dataset loaded: %s", json.dumps(store.stats))

    pipeline = RouterPipeline(store, use_llm=not args.no_llm)
    report = pipeline.run()

    output = write_output(report.decisions, Path(args.output) if args.output else None)
    audit = write_audit(report)

    log.info("run summary:\n%s", json.dumps(report.summary(), indent=2))
    log.info("wrote %s (%d rows)", output, len(report.decisions))
    log.info("wrote audit trail %s", audit)

    ok, problems = _validate(output, store)
    for problem in problems:
        log.error("VALIDATION: %s", problem)
    if ok:
        log.info("VALIDATION: output.csv satisfies the submission contract")
    return 0 if ok else 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    result = evaluate_samples(use_llm=not args.no_llm)
    path = write_result(result)
    print(render_report(result))
    log.info("wrote %s", path)
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    """Measure what each layer actually contributes, rather than assuming."""
    store = ContextStore.load()
    results: list[EvalResult] = []

    arms = [
        ("A. rules only (no LLM, no media)", {"ORCHESTRATE_USE_LLM": "0", "ORCHESTRATE_USE_MEDIA": "0"}),
        ("B. rules + media understanding", {"ORCHESTRATE_USE_LLM": "0", "ORCHESTRATE_USE_MEDIA": "1"}),
        ("C. LLM judge, no media", {"ORCHESTRATE_USE_LLM": "1", "ORCHESTRATE_USE_MEDIA": "0"}),
        ("D. full pipeline", {"ORCHESTRATE_USE_LLM": "1", "ORCHESTRATE_USE_MEDIA": "1"}),
    ]
    for name, env in arms:
        log.info("--- ablation arm: %s ---", name)
        with _env(env):
            _reload_config()
            results.append(
                evaluate_samples(use_llm=env["ORCHESTRATE_USE_LLM"] == "1", name=name, store=store)
            )
    _reload_config()

    print(render_comparison(results, title="Ablation: contribution of each layer"))
    path = PATHS.runs / "ablation.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    """Score the identical pipeline across several judge models."""
    store = ContextStore.load()
    results: list[EvalResult] = []
    for model in args.model:
        log.info("--- model: %s ---", model)
        with _env({"ORCHESTRATE_JUDGE_MODELS": model, "ORCHESTRATE_USE_LLM": "1"}):
            _reload_config()
            results.append(evaluate_samples(use_llm=True, name=model, store=store))
    _reload_config()

    print(render_comparison(results, title="Judge model comparison"))
    path = PATHS.runs / "model_comparison.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.as_dict() for r in results], indent=2), encoding="utf-8")
    log.info("wrote %s", path)
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    store = ContextStore.load()
    path = Path(args.output) if args.output else PATHS.output_csv
    ok, problems = _validate(path, store)
    for problem in problems:
        print(f"  FAIL  {problem}")
    print(
        "  PASS  output.csv satisfies the submission contract"
        if ok
        else f"\n{len(problems)} problem(s) found"
    )
    return 0 if ok else 1


# --------------------------------------------------------------------------- #

def _validate(path: Path, store: ContextStore) -> tuple[bool, list[str]]:
    """Check output.csv against the submission contract in problem_statement.md."""
    import csv

    from router.schema import Action, MessageType

    problems: list[str] = []
    if not path.is_file():
        return False, [f"{path} does not exist"]

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        header = list(rows[0].keys()) if rows else []

    if tuple(header) != OUTPUT_COLUMNS:
        problems.append(f"header is {header}, must be exactly {list(OUTPUT_COLUMNS)} in order")

    expected = [m.message_id for m in store.incoming]
    got = [r.get("message_id", "") for r in rows]
    if len(got) != len(expected):
        problems.append(f"{len(got)} rows, expected {len(expected)} (one per messages.csv row)")
    for mid in set(expected) - set(got):
        problems.append(f"missing prediction for {mid}")
    for mid in set(got) - set(expected):
        problems.append(f"prediction for unknown message_id {mid}")
    if len(set(got)) != len(got):
        problems.append("duplicate message_id rows present")

    actions = {a.value for a in Action}
    types = {t.value for t in MessageType}
    for row in rows:
        mid = row.get("message_id", "?")
        if row.get("action") not in actions:
            problems.append(f"{mid}: action {row.get('action')!r} not in {sorted(actions)}")
        if row.get("message_type") not in types:
            problems.append(f"{mid}: message_type {row.get('message_type')!r} not allowed")
        if not (row.get("reason") or "").strip():
            problems.append(f"{mid}: empty reason")
        try:
            confidence = float(row.get("confidence", ""))
            if not 0.0 <= confidence <= 1.0:
                problems.append(f"{mid}: confidence {confidence} outside [0, 1]")
        except (TypeError, ValueError):
            problems.append(f"{mid}: confidence {row.get('confidence')!r} is not a number")

        evidence = (row.get("evidence_message_ids") or "").strip()
        if not evidence:
            problems.append(f"{mid}: evidence_message_ids empty; must be ids or the literal 'none'")
        elif evidence != "none":
            for eid in evidence.split(";"):
                if eid.strip() not in store.history:
                    problems.append(f"{mid}: evidence id {eid!r} is not in message_history.csv")

    return not problems, problems


class _env:
    """Temporarily override environment variables."""

    def __init__(self, overrides: dict[str, str]) -> None:
        self.overrides = overrides
        self.saved: dict[str, str | None] = {}

    def __enter__(self) -> "_env":
        for key, value in self.overrides.items():
            self.saved[key] = os.environ.get(key)
            os.environ[key] = value
        return self

    def __exit__(self, *exc: object) -> None:
        for key, value in self.saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _reload_config() -> None:
    """Re-read env-driven config after an override.

    ``config`` snapshots the environment into frozen dataclasses at import time,
    which is what keeps a single run internally consistent. The ablation and
    comparison commands are the only places that need to change configuration
    mid-process, so they rebuild those singletons explicitly.
    """
    import router.config as cfg

    cfg.PATHS = cfg.Paths()
    cfg.MODELS = cfg.ModelConfig()
    cfg.ROUTER = cfg.RouterConfig()
    for module_name in (
        "router.pipeline", "router.judge", "router.media",
        "router.retrieval", "router.llm.provider", "router.arbiter",
    ):
        module = sys.modules.get(module_name)
        if module is None:
            continue
        for attr in ("PATHS", "MODELS", "ROUTER"):
            if hasattr(module, attr):
                setattr(module, attr, getattr(cfg, attr))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="WhatsApp message notification router",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="route dataset/messages.csv and write output.csv")
    run.add_argument("-o", "--output", help="output path (default: repo-root output.csv)")
    run.add_argument("--no-llm", action="store_true", help="deterministic baseline only")
    run.set_defaults(func=cmd_run)

    ev = sub.add_parser("evaluate", help="score against dataset/sample_messages.csv")
    ev.add_argument("--no-llm", action="store_true")
    ev.set_defaults(func=cmd_evaluate)

    sub.add_parser("ablate", help="measure each layer's contribution").set_defaults(func=cmd_ablate)

    cmp_ = sub.add_parser("compare", help="score the pipeline across judge models")
    cmp_.add_argument("-m", "--model", action="append", required=True, help="repeatable")
    cmp_.set_defaults(func=cmd_compare)

    val = sub.add_parser("validate", help="check output.csv against the submission contract")
    val.add_argument("-o", "--output")
    val.set_defaults(func=cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
