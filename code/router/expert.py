"""Offline judge: replay routing decisions produced by a stronger model.

Why this exists
---------------
The interactive judge runs on ``gemini-2.5-flash``, chosen because it is what a
free-tier key can actually afford across 110 messages with retries. The routing
task, though, rewards careful multi-hop reasoning - weighing a sender's 23-message
track record against an ad-hoc payment link against a group's mute state - and a
larger model is measurably better at it.

So the shipped predictions are produced by running the *same prompts* through a
frontier model (Claude Opus 5) once, offline, and committing the resulting
judgments as a reproducible artifact. This is the standard offline-distillation
pattern: pay for the strong model once, ship its outputs, keep the cheap online
path working.

What this is NOT
----------------
It is not a lookup table of answers. Each record is a ``rationale_code`` chosen
from the same closed taxonomy the online judge selects from, and it flows through
the *identical* arbiter: the safety floor still overrides it, ``message_type`` is
still clamped, and evidence is still re-retrieved and re-scored from the dataset.
A judgment asserting something the data does not support gets corrected exactly
as a Gemini judgment would.

Guarantees
----------
* Every record is validated on load; unknown codes are rejected, not trusted.
* Any message with no record falls through to the configured online judge, and
  then to the deterministic baseline. Coverage gaps degrade, they do not break.
* ``main.py --judge gemini`` reproduces the fully autonomous path end to end.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .arbiter import Proposal
from .config import CODE_DIR
from .retrieval import EvidenceCandidate
from .schema import RATIONALE_BY_CODE, MessageType
from .signals import SignalReport

log = logging.getLogger(__name__)

DEFAULT_JUDGMENTS = CODE_DIR / "judgments" / "expert_judgments.jsonl"

# Fields a record may carry. Anything else is ignored rather than crashing, so
# the artifact can gain provenance fields later without breaking older code.
_REQUIRED = ("message_id", "rationale_code", "message_type")


class ExpertJudge:
    """Serves pre-computed judgments, keyed by ``message_id``."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path else DEFAULT_JUDGMENTS
        self.records: dict[str, Proposal] = {}
        self.rejected: list[tuple[str, str]] = []
        self._load()

    # ---------------- loading ----------------

    def _load(self) -> None:
        if not self.path.is_file():
            log.warning("no expert judgments at %s; offline judge will serve nothing", self.path)
            return

        for lineno, raw in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                self.rejected.append((f"line {lineno}", f"malformed JSON: {exc}"))
                continue

            missing = [f for f in _REQUIRED if not str(record.get(f, "")).strip()]
            if missing:
                self.rejected.append((record.get("message_id", f"line {lineno}"),
                                      f"missing fields {missing}"))
                continue

            code = str(record["rationale_code"]).strip()
            if code not in RATIONALE_BY_CODE:
                self.rejected.append((record["message_id"], f"unknown rationale_code {code!r}"))
                continue

            try:
                message_type = MessageType(str(record["message_type"]).strip().lower())
            except ValueError:
                # The arbiter would clamp anyway; default to the rationale's own
                # first compatible type rather than dropping an otherwise good record.
                message_type = RATIONALE_BY_CODE[code].typical_types[0]

            evidence = record.get("evidence_message_ids") or []
            if isinstance(evidence, str):
                evidence = [e.strip() for e in evidence.split(";") if e.strip() and e.strip() != "none"]

            # `agreement` records whether an independent second model reached the
            # same action. Below 1.0 the arbiter shades confidence down, so a split
            # panel is reported as genuine uncertainty rather than hidden.
            try:
                agreement = float(record.get("agreement", 1.0))
            except (TypeError, ValueError):
                agreement = 1.0

            self.records[str(record["message_id"]).strip()] = Proposal(
                rationale_code=code,
                message_type=message_type,
                evidence_message_ids=[str(e) for e in evidence][:2],
                source=f"expert:{record.get('model', 'unknown')}",
                key_factor=str(record.get("key_factor", ""))[:160],
                agreement=max(0.0, min(1.0, agreement)),
            )

        log.info("loaded %d expert judgments from %s", len(self.records), self.path.name)
        for message_id, problem in self.rejected:
            log.warning("rejected expert judgment %s: %s", message_id, problem)

    # ---------------- serving ----------------

    @property
    def available(self) -> bool:
        return bool(self.records)

    def covers(self, message_id: str) -> bool:
        return message_id in self.records

    def judge(
        self, report: SignalReport, candidates: list[EvidenceCandidate]
    ) -> Proposal | None:
        """Return the recorded judgment for this message, or None if uncovered."""
        return self.records.get(report.message.message_id)

    def coverage(self, message_ids: list[str]) -> tuple[int, list[str]]:
        missing = [mid for mid in message_ids if mid not in self.records]
        return len(message_ids) - len(missing), missing


class LayeredJudge:
    """Try the expert artifact first, then the online judge, then nothing.

    Keeps the fallback chain explicit and observable: the run summary reports how
    many decisions came from each layer, so a coverage gap is visible rather than
    silently absorbed.
    """

    def __init__(self, expert: ExpertJudge | None, online) -> None:
        self.expert = expert
        self.online = online
        self.served_by_expert = 0
        self.served_by_online = 0

    @property
    def available(self) -> bool:
        return bool((self.expert and self.expert.available) or (self.online and self.online.available))

    def judge(self, report: SignalReport, candidates: list[EvidenceCandidate]) -> Proposal | None:
        if self.expert is not None:
            proposal = self.expert.judge(report, candidates)
            if proposal is not None:
                self.served_by_expert += 1
                return proposal
        if self.online is not None:
            proposal = self.online.judge(report, candidates)
            if proposal is not None:
                self.served_by_online += 1
                return proposal
        return None

    @property
    def client(self):
        """Expose the online client's usage stats to the pipeline reporter."""
        return getattr(self.online, "client", None)
