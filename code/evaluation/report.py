"""Human-readable rendering of evaluation results.

Terminal-friendly markdown, deliberately including the per-row error listing.
With n=30 labelled rows a single row is 3.3 percentage points, so the aggregate
alone is not enough to tune against - you have to look at what actually missed.
"""

from __future__ import annotations

from evaluation.evaluate import EvalResult


def _bar(value: float, width: int = 24) -> str:
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def render_report(result: EvalResult) -> str:
    lines: list[str] = [
        "",
        "=" * 78,
        f"  EVALUATION - {result.name}   (n={result.n} labelled rows)",
        "=" * 78,
        "",
        "  CRITERION                        SCORE",
        "  " + "-" * 62,
        f"  action accuracy                  {result.action.accuracy:6.1%}  {_bar(result.action.accuracy)}",
        f"  action macro-F1                  {result.action.macro_f1:6.1%}  {_bar(result.action.macro_f1)}",
        f"  message_type accuracy            {result.message_type.accuracy:6.1%}  {_bar(result.message_type.accuracy)}",
        f"  action AND type both correct     {result.both_correct:6.1%}  {_bar(result.both_correct)}",
        f"  reason exact match               {result.reason.exact_match:6.1%}  {_bar(result.reason.exact_match)}",
        f"  reason similarity to gold        {result.reason.mean_similarity:6.1%}  {_bar(result.reason.mean_similarity)}",
        f"  reason self-consistency          {result.reason.reason_action_consistency:6.1%}  {_bar(result.reason.reason_action_consistency)}",
        f"  evidence hit rate                {result.evidence.hit_rate:6.1%}  {_bar(result.evidence.hit_rate)}",
        f"  evidence cardinality match       {result.evidence.cardinality_match:6.1%}  {_bar(result.evidence.cardinality_match)}",
        f"  evidence ids valid               {result.evidence.valid_ids:6.1%}  {_bar(result.evidence.valid_ids)}",
        "",
        "  CONFIDENCE CALIBRATION",
        "  " + "-" * 62,
        f"  expected calibration error       {result.calibration.ece:.4f}   (lower is better)",
        f"  Brier score                      {result.calibration.brier:.4f}   (lower is better)",
        f"  mean confidence                  {result.calibration.mean_confidence:.3f}",
        f"    when action correct            {result.calibration.mean_when_correct:.3f}",
        f"    when action wrong              {result.calibration.mean_when_wrong:.3f}",
        f"  separation (correct - wrong)     {result.calibration.separation:+.4f}   (higher is better)",
    ]

    if result.action.confusion:
        labels = sorted(result.action.confusion)
        lines += ["", "  ACTION CONFUSION  (rows = gold, columns = predicted)", "  " + "-" * 62]
        lines.append("  " + " " * 10 + "".join(f"{l:>10}" for l in labels))
        for gold in labels:
            row = result.action.confusion[gold]
            cells = "".join(
                f"{row[p]:>10}" if gold != p else f"{'[' + str(row[p]) + ']':>10}" for p in labels
            )
            lines.append(f"  {gold:>10}{cells}")

    if result.message_type.per_class:
        lines += ["", "  MESSAGE TYPE - per class", "  " + "-" * 62,
                  f"  {'type':<18}{'prec':>8}{'recall':>8}{'F1':>8}{'n':>6}"]
        for label, m in sorted(result.message_type.per_class.items(), key=lambda kv: -kv[1]["support"]):
            if not m["support"]:
                continue
            lines.append(
                f"  {label:<18}{m['precision']:>8.2f}{m['recall']:>8.2f}{m['f1']:>8.2f}{m['support']:>6}"
            )

    if result.errors:
        lines += ["", f"  ERRORS  ({len(result.errors)} rows)", "  " + "-" * 62]
        for err in result.errors:
            action_note = (
                f"action {err['gold_action']} -> {err['pred_action']}"
                if err["gold_action"] != err["pred_action"] else "action OK"
            )
            type_note = (
                f"type {err['gold_type']} -> {err['pred_type']}"
                if err["gold_type"] != err["pred_type"] else "type OK"
            )
            lines.append(f"  {err['message_id']}  {action_note};  {type_note}")
            lines.append(f"      code={err['rationale_code']}  conf={err['confidence']}  by={err['decided_by']}")
            if err["text"]:
                lines.append(f"      text: {err['text']}")
            for note in err["notes"]:
                lines.append(f"      note: {note}")
    else:
        lines += ["", "  No action/type errors on the labelled rows."]

    if result.run_summary:
        usage = result.run_summary.get("llm_usage", {})
        lines += ["", "  RUN", "  " + "-" * 62,
                  f"  seconds                          {result.run_summary.get('seconds')}",
                  f"  decided by judge / baseline      {result.run_summary.get('decided_by_judge')} / "
                  f"{result.run_summary.get('decided_by_baseline')}",
                  f"  safety overrides applied         {result.run_summary.get('safety_overrides')}",
                  f"  judge vs baseline disagreements  {result.run_summary.get('judge_vs_baseline_disagreements')}"]
        # llm_usage mixes per-client dicts with scalar counters, so render each shape.
        for arm, stats in usage.items():
            if isinstance(stats, dict) and "calls" in stats:
                lines.append(
                    f"  {arm + ' calls / cache hits':<32} "
                    f"{stats.get('calls')} / {stats.get('cache_hits')}"
                )
            elif arm == "budget" and isinstance(stats, dict):
                lines.append(
                    f"  {'paid spend':<32} ${stats.get('spent_usd', 0):.4f} "
                    f"of ${stats.get('ceiling_usd', 0):.2f} ceiling "
                    f"({stats.get('paid_calls', 0)} calls)"
                )
            elif isinstance(stats, dict):
                lines.append(f"  {arm:<32} {stats}")
            else:
                lines.append(f"  {arm:<32} {stats}")

    lines += ["", "=" * 78, ""]
    return "\n".join(lines)


def render_comparison(results: list[EvalResult], title: str = "Comparison") -> str:
    """Side-by-side headline table across configurations."""
    if not results:
        return "(no results)"

    rows = [
        ("action acc", lambda r: f"{r.action.accuracy:.1%}"),
        ("action F1", lambda r: f"{r.action.macro_f1:.1%}"),
        ("type acc", lambda r: f"{r.message_type.accuracy:.1%}"),
        ("both", lambda r: f"{r.both_correct:.1%}"),
        ("reason exact", lambda r: f"{r.reason.exact_match:.1%}"),
        ("evidence hit", lambda r: f"{r.evidence.hit_rate:.1%}"),
        ("ECE", lambda r: f"{r.calibration.ece:.3f}"),
        ("Brier", lambda r: f"{r.calibration.brier:.3f}"),
        ("errors", lambda r: str(len(r.errors))),
    ]

    width = max(22, max(len(r.name) for r in results) + 2)
    lines = ["", "=" * 78, f"  {title}", "=" * 78, ""]
    lines.append("  " + " " * 16 + "".join(f"{r.name[:width - 2]:>{width}}" for r in results))
    lines.append("  " + "-" * (16 + width * len(results)))
    for label, fn in rows:
        lines.append("  " + f"{label:<16}" + "".join(f"{fn(r):>{width}}" for r in results))
    lines += ["", "=" * 78, ""]
    return "\n".join(lines)
