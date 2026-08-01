"""Evaluation harness: score the router against the labelled sample rows.

``dataset/sample_messages.csv`` carries 30 rows with the expected ``action``,
``message_type``, ``reason``, ``confidence`` and ``evidence_message_ids``. It is
the only ground truth available before submission, so it is what tuning is
measured against.

The harness supports three modes, all writing the same report shape:

* ``evaluate``  - score one configuration end to end.
* ``ablate``    - score the deterministic baseline against the full pipeline, so
                  the LLM's actual contribution is measured rather than assumed.
* ``compare``   - score the same pipeline across several judge models.

A caution the report states explicitly: n=30 is small. A three-row swing is four
percentage points of headline accuracy, so per-row error listings matter more
here than the aggregate, and the report prints every miss.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from router.config import PATHS
from router.context_store import ContextStore, Message
from router.pipeline import RouterPipeline
from router.schema import Decision

from .metrics import (
    CalibrationMetrics,
    ClassMetrics,
    EvidenceMetrics,
    ReasonMetrics,
    calibration_metrics,
    classification_metrics,
    evidence_metrics,
    reason_metrics,
)

log = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Full scorecard for one configuration."""

    name: str
    n: int
    action: ClassMetrics = field(default_factory=ClassMetrics)
    message_type: ClassMetrics = field(default_factory=ClassMetrics)
    reason: ReasonMetrics = field(default_factory=ReasonMetrics)
    evidence: EvidenceMetrics = field(default_factory=EvidenceMetrics)
    calibration: CalibrationMetrics = field(default_factory=CalibrationMetrics)
    both_correct: float = 0.0
    errors: list[dict] = field(default_factory=list)
    run_summary: dict = field(default_factory=dict)

    def headline(self) -> dict:
        return {
            "config": self.name,
            "n": self.n,
            "action_accuracy": self.action.accuracy,
            "action_macro_f1": self.action.macro_f1,
            "type_accuracy": self.message_type.accuracy,
            "action_and_type_correct": round(self.both_correct, 4),
            "reason_exact_match": self.reason.exact_match,
            "reason_similarity": self.reason.mean_similarity,
            "reason_self_consistency": self.reason.reason_action_consistency,
            "evidence_hit_rate": self.evidence.hit_rate,
            "evidence_none_recall": self.evidence.none_recall,
            "evidence_ids_valid": self.evidence.valid_ids,
            "confidence_ece": self.calibration.ece,
            "confidence_brier": self.calibration.brier,
            "confidence_separation": self.calibration.separation,
        }

    def as_dict(self) -> dict:
        return {
            "headline": self.headline(),
            "action": self.action.__dict__,
            "message_type": self.message_type.__dict__,
            "reason": self.reason.__dict__,
            "evidence": self.evidence.__dict__,
            "calibration": self.calibration.__dict__,
            "errors": self.errors,
            "run_summary": self.run_summary,
        }


def score(
    decisions: list[Decision],
    gold_rows: list[dict[str, str]],
    *,
    name: str = "pipeline",
    known_ids: set[str] | None = None,
    run_summary: dict | None = None,
) -> EvalResult:
    """Score predictions against labelled rows, matched by message_id."""
    by_id = {d.message_id: d for d in decisions}
    paired = [(row, by_id[row["message_id"]]) for row in gold_rows if row["message_id"] in by_id]
    missing = [r["message_id"] for r in gold_rows if r["message_id"] not in by_id]
    if missing:
        log.warning("%d labelled rows had no prediction: %s", len(missing), missing[:5])
    if not paired:
        return EvalResult(name=name, n=0)

    gold_actions = [r["action"].strip() for r, _ in paired]
    pred_actions = [d.action.value for _, d in paired]
    gold_types = [r["message_type"].strip() for r, _ in paired]
    pred_types = [d.message_type.value for _, d in paired]
    gold_reasons = [r["reason"].strip() for r, _ in paired]
    pred_reasons = [d.reason for _, d in paired]
    gold_evidence = [_split_ids(r["evidence_message_ids"]) for r, _ in paired]
    pred_evidence = [d.evidence_message_ids for _, d in paired]
    confidences = [d.confidence for _, d in paired]
    action_correct = [g == p for g, p in zip(gold_actions, pred_actions)]

    result = EvalResult(
        name=name,
        n=len(paired),
        action=classification_metrics(gold_actions, pred_actions),
        message_type=classification_metrics(gold_types, pred_types),
        reason=reason_metrics(gold_reasons, pred_reasons, pred_actions),
        evidence=evidence_metrics(gold_evidence, pred_evidence, known_ids),
        calibration=calibration_metrics(confidences, action_correct),
        both_correct=sum(
            ga == pa and gt == pt
            for ga, pa, gt, pt in zip(gold_actions, pred_actions, gold_types, pred_types)
        ) / len(paired),
        run_summary=run_summary or {},
    )

    for (row, decision), ok in zip(paired, action_correct):
        type_ok = row["message_type"].strip() == decision.message_type.value
        if ok and type_ok:
            continue
        result.errors.append({
            "message_id": row["message_id"],
            "gold_action": row["action"],
            "pred_action": decision.action.value,
            "gold_type": row["message_type"],
            "pred_type": decision.message_type.value,
            "rationale_code": decision.rationale_code,
            "confidence": decision.confidence,
            "decided_by": decision.source,
            "notes": decision.notes,
            "gold_reason": row["reason"],
            "pred_reason": decision.reason,
            "text": (row.get("message_text") or "")[:180].replace("\n", " "),
        })
    return result


def evaluate_samples(
    *, use_llm: bool = True, name: str | None = None, store: ContextStore | None = None
) -> EvalResult:
    """Run the pipeline over the labelled sample rows and score it."""
    store = store or ContextStore.load()
    if not store.samples:
        raise FileNotFoundError("dataset/sample_messages.csv not found; cannot evaluate")

    messages = [Message.from_row(row) for row in store.samples]
    pipeline = RouterPipeline(store, use_llm=use_llm)
    run = pipeline.run(messages)

    return score(
        run.decisions,
        store.samples,
        name=name or ("full pipeline (LLM judge)" if use_llm else "deterministic baseline"),
        known_ids=set(store.history),
        run_summary=run.summary(),
    )


def write_result(result: EvalResult, directory: Path | None = None, filename: str = "evaluation.json") -> Path:
    target = Path(directory) if directory else PATHS.runs
    target.mkdir(parents=True, exist_ok=True)
    path = target / filename
    path.write_text(json.dumps(result.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _split_ids(raw: str) -> list[str]:
    value = (raw or "").strip()
    if not value or value.lower() == "none":
        return []
    return [part.strip() for part in value.split(";") if part.strip()]
