"""The LLM routing judge, with optional self-consistency voting.

The judge sees a quarantined copy of the message, the deterministic briefing, and
the retrieved evidence candidates. It returns a *rationale code* rather than free
text, so its output is a choice from a closed set that the arbiter can validate.

Self-consistency (``ORCHESTRATE_JUDGE_SAMPLES > 1``) draws N independent samples
and takes the majority action. It is off by default because the free-tier quota
is better spent covering all 110 messages once than covering a third of them
three times - but when it is on, the level of agreement flows into the confidence
rather than being thrown away.

Sample 1 is always greedy (temperature 0), so a single-sample run is fully
deterministic and reproducible from cache.
"""

from __future__ import annotations

import logging
from collections import Counter

from .arbiter import Proposal
from .config import MODELS, ROUTER
from .llm.prompts import JUDGE_SCHEMA, SYSTEM_PROMPT, build_user_prompt
from .llm.provider import GeminiClient
from .retrieval import EvidenceCandidate
from .schema import RATIONALE_BY_CODE, MessageType
from .signals import SignalReport

log = logging.getLogger(__name__)


class RoutingJudge:
    """Wraps the model call, schema validation, and ensemble voting."""

    def __init__(self, client: GeminiClient | None = None) -> None:
        self.client = client or GeminiClient(
            models=MODELS.judge_models,
            cache_enabled=ROUTER.use_cache,
            cache_namespace="judge",
        )

    @property
    def available(self) -> bool:
        return self.client.available

    def judge(
        self, report: SignalReport, candidates: list[EvidenceCandidate]
    ) -> Proposal | None:
        """Route one message. Returns None if the model could not be reached."""
        prompt = build_user_prompt(report, candidates)
        samples = max(1, ROUTER.judge_samples)
        proposals: list[Proposal] = []

        for index in range(samples):
            response = self.client.generate(
                prompt,
                system=SYSTEM_PROMPT,
                response_schema=JUDGE_SCHEMA,
                # Sample 0 greedy for determinism; later samples diversified so
                # the vote reflects genuine model uncertainty, not resampling noise.
                temperature=0.0 if index == 0 else ROUTER.ensemble_temperature,
                cache_salt=f"sample{index}",
            )
            if response is None:
                break
            parsed = self._parse(response.json(), report)
            if parsed is not None:
                proposals.append(parsed)

        if not proposals:
            return None
        if len(proposals) == 1:
            return proposals[0]
        return self._vote(proposals)

    # ---------------- internals ----------------

    def _parse(self, payload: dict | None, report: SignalReport) -> Proposal | None:
        if not isinstance(payload, dict):
            log.warning("%s: judge returned unparseable output", report.message.message_id)
            return None

        code = str(payload.get("rationale_code", "")).strip()
        if code not in RATIONALE_BY_CODE:
            log.warning("%s: judge returned unknown rationale_code %r", report.message.message_id, code)
            return None

        raw_type = str(payload.get("message_type", "")).strip().lower()
        try:
            message_type = MessageType(raw_type)
        except ValueError:
            # The arbiter clamps to the rationale's first compatible type anyway.
            message_type = RATIONALE_BY_CODE[code].typical_types[0]
            log.debug("%s: unknown message_type %r; using rationale default",
                      report.message.message_id, raw_type)

        evidence = payload.get("evidence_message_ids") or []
        if not isinstance(evidence, list):
            evidence = []

        return Proposal(
            rationale_code=code,
            message_type=message_type,
            evidence_message_ids=[str(e).strip() for e in evidence if str(e).strip()][:2],
            source="judge",
            key_factor=str(payload.get("key_factor", ""))[:160],
        )

    @staticmethod
    def _vote(proposals: list[Proposal]) -> Proposal:
        """Majority vote on the action, then on the specific rationale within it."""
        actions = Counter(RATIONALE_BY_CODE[p.rationale_code].action for p in proposals)
        winning_action, action_votes = actions.most_common(1)[0]

        in_majority = [
            p for p in proposals if RATIONALE_BY_CODE[p.rationale_code].action is winning_action
        ]
        codes = Counter(p.rationale_code for p in in_majority)
        winning_code = codes.most_common(1)[0][0]
        winner = next(p for p in in_majority if p.rationale_code == winning_code)

        # Union the evidence the agreeing samples cited; the retriever re-scores
        # it, so a wider proposal set costs nothing and can only help recall.
        evidence: list[str] = []
        for proposal in in_majority:
            for mid in proposal.evidence_message_ids:
                if mid not in evidence:
                    evidence.append(mid)

        return Proposal(
            rationale_code=winning_code,
            message_type=winner.message_type,
            evidence_message_ids=evidence[:2],
            source="judge",
            key_factor=winner.key_factor,
            agreement=action_votes / len(proposals),
        )
