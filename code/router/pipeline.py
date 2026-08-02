"""Pipeline orchestration: dataset in, decisions out.

Stage order:

    load context -> understand media -> extract signals -> retrieve candidates
        -> judge (LLM)  ─┐
        -> baseline      ├─> arbiter -> Decision
                        ─┘

The judge and the baseline produce the same ``Proposal`` shape and both flow
through the same arbiter, which is what makes the ablation comparison honest and
lets the pipeline substitute one for the other transparently when the API is
unavailable.

Every decision carries an audit record - the signals that fired, what each arm
proposed, and any override the arbiter applied - written alongside the CSV.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from . import baseline as baseline_router
from .arbiter import PolicyArbiter, Proposal
from .config import PATHS, ROUTER
from .context_store import ContextStore, Message
from .expert import ExpertJudge, LayeredJudge
from .judge import RoutingJudge
from .media import MediaAnalyzer, MediaUnderstanding
from .retrieval import EvidenceRetriever
from .schema import OUTPUT_COLUMNS, RATIONALE_BY_CODE, Decision
from .signals import SignalExtractor, SignalReport

log = logging.getLogger(__name__)


@dataclass
class RunReport:
    """Summary of one pipeline execution."""

    decisions: list[Decision] = field(default_factory=list)
    audit: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    judge_used: int = 0
    baseline_used: int = 0
    safety_overrides: int = 0
    arm_disagreements: int = 0
    media_analysed: int = 0
    llm_usage: dict = field(default_factory=dict)

    def summary(self) -> dict:
        from collections import Counter
        return {
            "messages": len(self.decisions),
            "seconds": round(self.seconds, 1),
            "actions": dict(Counter(d.action.value for d in self.decisions)),
            "message_types": dict(Counter(d.message_type.value for d in self.decisions)),
            "decided_by_judge": self.judge_used,
            "decided_by_baseline": self.baseline_used,
            "safety_overrides": self.safety_overrides,
            "judge_vs_baseline_disagreements": self.arm_disagreements,
            "media_analysed": self.media_analysed,
            "evidence_emitted": sum(1 for d in self.decisions if d.evidence_message_ids),
            "evidence_none": sum(1 for d in self.decisions if not d.evidence_message_ids),
            "llm_usage": self.llm_usage,
        }


class RouterPipeline:
    """End-to-end routing over a set of messages."""

    def __init__(
        self,
        store: ContextStore | None = None,
        *,
        use_llm: bool | None = None,
        judge_source: str | None = None,
    ) -> None:
        """``judge_source``: ``auto`` (expert artifact, then online), ``expert``,
        ``online`` (force the live model), or ``none`` (deterministic only)."""
        self.store = store or ContextStore.load()
        self.signals = SignalExtractor(self.store)
        self.retriever = EvidenceRetriever(self.store)
        self.arbiter = PolicyArbiter(self.store)
        self.media = MediaAnalyzer(self.store)

        self.use_llm = ROUTER.use_llm if use_llm is None else use_llm
        source = (judge_source or ROUTER.judge_source).strip().lower()
        self.judge_source = "none" if not self.use_llm else source
        self.judge: RoutingJudge | LayeredJudge | None = None

        if self.judge_source == "none":
            return

        expert = None
        if self.judge_source in {"auto", "expert"}:
            candidate = ExpertJudge()
            expert = candidate if candidate.available else None
            if expert is None and self.judge_source == "expert":
                log.warning("judge_source=expert but no judgments loaded; falling back to online")
                self.judge_source = "online"

        online = None
        if self.judge_source in {"auto", "online"}:
            candidate = RoutingJudge()
            online = candidate if candidate.available else None
            if online is None:
                log.warning("no API key configured; the online judge is unavailable")

        if expert is None and online is None:
            log.warning("no judge available; running the deterministic baseline only")
            self.use_llm = False
            self.judge_source = "none"
            return

        self.judge = LayeredJudge(expert, online)

    # ---------------- main entry ----------------

    def run(self, messages: list[Message] | None = None) -> RunReport:
        messages = messages if messages is not None else self.store.incoming
        report = RunReport()
        started = time.monotonic()

        if ROUTER.use_media and any(m.has_media for m in messages):
            results = self.media.analyse_all(messages)
            report.media_analysed = sum(1 for r in results.values() if r.ok)

        # Signals and retrieval are pure CPU work; do them all up front so the
        # (rate-limited) model calls are the only thing left to schedule.
        prepared = [self._prepare(m) for m in messages]

        if self.judge is not None:
            # Modest parallelism: the adaptive limiter serialises the actual
            # requests, so workers mostly overlap JSON handling and waiting.
            with ThreadPoolExecutor(max_workers=ROUTER.workers) as pool:
                proposals = list(pool.map(self._judge_one, prepared))
        else:
            proposals = [None] * len(prepared)

        for (signal_report, candidates, media), judged in zip(prepared, proposals):
            fallback = baseline_router.propose(signal_report, media)
            proposal = judged or fallback
            if judged is None:
                report.baseline_used += 1
            else:
                report.judge_used += 1
                if _action_of(judged) != _action_of(fallback):
                    report.arm_disagreements += 1

            decision = self.arbiter.finalise(signal_report, proposal, media)
            if any(n.startswith("safety override") or n.startswith("forced mute") for n in decision.notes):
                report.safety_overrides += 1

            report.decisions.append(decision)
            report.audit.append(
                self._audit_record(signal_report, candidates, media, judged, fallback, decision)
            )

        report.seconds = time.monotonic() - started
        usage = {}
        if self.judge is not None:
            client = getattr(self.judge, "client", None)
            if client is not None:
                usage["judge"] = client.stats.as_dict()
            if isinstance(self.judge, LayeredJudge):
                usage["served_by_expert"] = self.judge.served_by_expert
                usage["served_by_online"] = self.judge.served_by_online
                online = self.judge.online
                if online is not None:
                    # Spend is a first-class run output: a paid run should never
                    # finish without saying what it cost.
                    usage["budget"] = online.budget.as_dict()
                    served = getattr(online.client, "served", None)
                    if served:
                        usage["served_by_provider"] = dict(served)
        if ROUTER.use_media:
            usage["media"] = self.media.client.stats.as_dict()
        report.llm_usage = usage
        return report

    # ---------------- stages ----------------

    def _prepare(self, message: Message) -> tuple[SignalReport, list, MediaUnderstanding | None]:
        media = self.media.get(message.media_id) if message.has_media else None
        summary = media.render() if media is not None else ""
        signal_report = self.signals.extract(message, media_summary=summary)
        candidates = self.retriever.candidates(message)
        return signal_report, candidates, media

    def _judge_one(self, prepared: tuple) -> Proposal | None:
        signal_report, candidates, _media = prepared
        try:
            return self.judge.judge(signal_report, candidates) if self.judge else None
        except Exception:  # a single bad row must not abort a 110-message run
            log.exception("judge failed for %s; using baseline", signal_report.message.message_id)
            return None

    @staticmethod
    def _audit_record(
        report: SignalReport,
        candidates: list,
        media: MediaUnderstanding | None,
        judged: Proposal | None,
        fallback: Proposal,
        decision: Decision,
    ) -> dict:
        return {
            "message_id": report.message.message_id,
            "user_id": report.message.user_id,
            "conversation_type": report.message.conversation_type,
            "final": {
                "action": decision.action.value,
                "message_type": decision.message_type.value,
                "rationale_code": decision.rationale_code,
                "confidence": decision.confidence,
                "evidence": decision.evidence_message_ids,
                "source": decision.source,
                "notes": decision.notes,
            },
            "judge": None if judged is None else {
                "rationale_code": judged.rationale_code,
                "message_type": judged.message_type.value,
                "evidence": judged.evidence_message_ids,
                "key_factor": judged.key_factor,
                "agreement": judged.agreement,
            },
            "baseline": {
                "rationale_code": fallback.rationale_code,
                "message_type": fallback.message_type.value,
                "key_factor": fallback.key_factor,
            },
            "signals": {
                "risk_score": round(report.risk.score, 3),
                "risk_level": report.risk.level,
                "risk_reasons": report.risk.reasons,
                "families": report.risk.families,
                "impersonation": round(report.risk.impersonation, 3),
                "sender_first_contact": report.sender.is_first_contact,
                "sender_habitually_ignored": report.sender.is_habitually_ignored,
                "sender_is_admin": report.sender.is_group_admin,
                "group_muted": report.engagement.group_muted,
                "business_opted_out": report.engagement.business_opted_out,
                "repetition": {
                    "best_match": report.repetition.best_match_id,
                    "score": round(report.repetition.best_score, 3),
                    "outcome": report.repetition.match_outcome,
                },
            },
            "media": None if media is None else {
                "category": media.category,
                "summary": media.summary,
                "urgency": media.urgency,
                "risky": media.is_risky,
            },
            "evidence_candidates": [
                {"id": c.message.message_id, "affinity": c.affinity,
                 "similarity": round(c.similarity, 3), "outcome": c.outcome}
                for c in candidates[:5]
            ],
        }


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #

def write_output(decisions: list[Decision], path: Path | None = None) -> Path:
    """Write output.csv with exactly the required columns, in the required order."""
    target = Path(path) if path else PATHS.output_csv
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(OUTPUT_COLUMNS))
        writer.writeheader()
        for decision in decisions:
            writer.writerow(decision.to_row())
    return target


def write_audit(report: RunReport, directory: Path | None = None) -> Path:
    """Persist the full audit trail and run summary for inspection."""
    target_dir = Path(directory) if directory else PATHS.runs
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = {"summary": report.summary(), "decisions": report.audit}
    path = target_dir / "audit.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _action_of(proposal: Proposal) -> str:
    rationale = RATIONALE_BY_CODE.get(proposal.rationale_code)
    return rationale.action.value if rationale else "unknown"
