"""Scoring metrics mirroring the five stated evaluation criteria.

The problem statement grades correctness of ``action``, correctness of
``message_type``, usefulness and consistency of ``reason``, relevance of
``evidence_message_ids``, and confidence calibration. This module implements one
metric family per criterion so tuning optimises what is actually scored, rather
than whatever is easiest to measure.

Two choices worth noting:

* **Macro-F1 alongside accuracy.** The label distribution is roughly balanced but
  the *errors* are not - over-muting is the failure mode a plain accuracy number
  hides, because muting everything still scores 40% here.
* **ECE and Brier together.** ECE asks whether stated confidence matches observed
  accuracy in aggregate; Brier penalises confident individual mistakes. A system
  that emits 0.85 for everything can look calibrated on one and bad on the other.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from router.signals import text_similarity


@dataclass
class ClassMetrics:
    """Per-class precision / recall / F1, plus the aggregates."""

    accuracy: float = 0.0
    macro_f1: float = 0.0
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    support: dict[str, int] = field(default_factory=dict)


def classification_metrics(gold: list[str], pred: list[str]) -> ClassMetrics:
    if len(gold) != len(pred):
        raise ValueError(f"length mismatch: {len(gold)} gold vs {len(pred)} predicted")

    labels = sorted(set(gold) | set(pred))
    confusion: dict[str, dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    for g, p in zip(gold, pred):
        confusion[g][p] += 1

    metrics = ClassMetrics(
        accuracy=sum(g == p for g, p in zip(gold, pred)) / len(gold) if gold else 0.0,
        confusion=confusion,
        support=dict(Counter(gold)),
    )

    f1s: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics.per_class[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": metrics.support.get(label, 0),
        }
        # Only classes actually present in the gold labels count toward macro-F1,
        # so a spurious extra prediction class cannot dilute the average.
        if metrics.support.get(label):
            f1s.append(f1)

    metrics.macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0
    metrics.accuracy = round(metrics.accuracy, 4)
    metrics.macro_f1 = round(metrics.macro_f1, 4)
    return metrics


@dataclass
class ReasonMetrics:
    exact_match: float = 0.0
    mean_similarity: float = 0.0
    distinct_reasons_used: int = 0
    reason_action_consistency: float = 0.0


def reason_metrics(
    gold_reasons: list[str],
    pred_reasons: list[str],
    pred_actions: list[str],
) -> ReasonMetrics:
    """Score reason quality.

    ``reason_action_consistency`` measures self-consistency rather than agreement
    with gold: does the same sentence always accompany the same action? A system
    that emits one wording for notify and another for mute is internally coherent
    even where it disagrees with the label, and incoherence is a real defect the
    similarity score alone would miss.
    """
    if not gold_reasons:
        return ReasonMetrics()

    exact = sum(g.strip() == p.strip() for g, p in zip(gold_reasons, pred_reasons))
    sims = [text_similarity(g, p) for g, p in zip(gold_reasons, pred_reasons)]

    by_reason: dict[str, set[str]] = defaultdict(set)
    for reason, action in zip(pred_reasons, pred_actions):
        by_reason[reason].add(action)
    consistent = sum(1 for actions in by_reason.values() if len(actions) == 1)

    return ReasonMetrics(
        exact_match=round(exact / len(gold_reasons), 4),
        mean_similarity=round(sum(sims) / len(sims), 4),
        distinct_reasons_used=len(by_reason),
        reason_action_consistency=round(consistent / len(by_reason), 4) if by_reason else 0.0,
    )


@dataclass
class EvidenceMetrics:
    hit_rate: float = 0.0          # any overlap with gold
    exact_set: float = 0.0         # identical id set
    precision: float = 0.0
    recall: float = 0.0
    cardinality_match: float = 0.0
    none_precision: float = 0.0
    none_recall: float = 0.0
    valid_ids: float = 0.0         # ids that actually exist in message_history


def evidence_metrics(
    gold: list[list[str]],
    pred: list[list[str]],
    known_ids: set[str] | None = None,
) -> EvidenceMetrics:
    if not gold:
        return EvidenceMetrics()

    hits = exact = card = 0
    precisions: list[float] = []
    recalls: list[float] = []
    gold_none = pred_none = both_none = 0
    total_ids = valid_ids = 0

    for g, p in zip(gold, pred):
        gset, pset = set(g), set(p)
        hits += bool(gset & pset)
        exact += gset == pset
        card += len(gset) == len(pset)
        if pset:
            precisions.append(len(gset & pset) / len(pset))
        if gset:
            recalls.append(len(gset & pset) / len(gset))
        if not gset:
            gold_none += 1
        if not pset:
            pred_none += 1
            both_none += not gset
        if known_ids is not None:
            total_ids += len(pset)
            valid_ids += sum(1 for i in pset if i in known_ids)

    n = len(gold)
    return EvidenceMetrics(
        hit_rate=round(hits / n, 4),
        exact_set=round(exact / n, 4),
        precision=round(sum(precisions) / len(precisions), 4) if precisions else 0.0,
        recall=round(sum(recalls) / len(recalls), 4) if recalls else 0.0,
        cardinality_match=round(card / n, 4),
        none_precision=round(both_none / pred_none, 4) if pred_none else 0.0,
        none_recall=round(both_none / gold_none, 4) if gold_none else 0.0,
        valid_ids=round(valid_ids / total_ids, 4) if total_ids else 1.0,
    )


@dataclass
class CalibrationMetrics:
    ece: float = 0.0               # expected calibration error
    brier: float = 0.0
    mean_confidence: float = 0.0
    mean_when_correct: float = 0.0
    mean_when_wrong: float = 0.0
    separation: float = 0.0        # correct minus wrong; >0 means confidence informs
    bins: list[dict] = field(default_factory=list)


def calibration_metrics(
    confidences: list[float], correct: list[bool], n_bins: int = 5
) -> CalibrationMetrics:
    """Confidence calibration.

    ``separation`` is the most actionable number here. The organizer's own
    confidences occupy a narrow 0.78-0.91 band, so absolute ECE is dominated by
    that offset. Whether confidence is *higher on rows we got right* is the
    property that makes the number useful at all.
    """
    if not confidences:
        return CalibrationMetrics()

    n = len(confidences)
    brier = sum((c - float(k)) ** 2 for c, k in zip(confidences, correct)) / n

    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for conf, ok in zip(confidences, correct):
        index = min(n_bins - 1, int(conf * n_bins))
        buckets[index].append((conf, ok))

    ece = 0.0
    bins: list[dict] = []
    for index in sorted(buckets):
        entries = buckets[index]
        avg_conf = sum(c for c, _ in entries) / len(entries)
        accuracy = sum(k for _, k in entries) / len(entries)
        ece += (len(entries) / n) * abs(avg_conf - accuracy)
        bins.append({
            "range": f"{index / n_bins:.1f}-{(index + 1) / n_bins:.1f}",
            "count": len(entries),
            "mean_confidence": round(avg_conf, 4),
            "accuracy": round(accuracy, 4),
            "gap": round(avg_conf - accuracy, 4),
        })

    right = [c for c, k in zip(confidences, correct) if k]
    wrong = [c for c, k in zip(confidences, correct) if not k]
    mean_right = sum(right) / len(right) if right else 0.0
    mean_wrong = sum(wrong) / len(wrong) if wrong else 0.0

    return CalibrationMetrics(
        ece=round(ece, 4),
        brier=round(brier, 4),
        mean_confidence=round(sum(confidences) / n, 4),
        mean_when_correct=round(mean_right, 4),
        mean_when_wrong=round(mean_wrong, 4),
        separation=round(mean_right - mean_wrong, 4),
        bins=bins,
    )
