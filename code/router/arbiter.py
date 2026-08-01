"""Policy arbitration: turn a proposed rationale into a final, consistent decision.

This is the last stage before ``output.csv``. It owns four responsibilities.

**Safety floor.** If the deterministic layer proved a message solicits credentials,
demands an advance fee, or attacks the router, the action is forced to ``mute``.
The override is deliberately *asymmetric*: when the judge has already chosen some
mute rationale, its choice stands, because the model has more context for
deciding *which* mute pattern applies. The override only fires when the judge
proposed notify or digest on something demonstrably unsafe. Rules set the floor;
the model supplies the nuance.

**Internal consistency.** ``message_type`` is clamped to a value the chosen
rationale can actually justify, so no row ever pairs "harmless greeting" with
``scam``.

**Rendering.** The rationale determines the reason sentence and the calibrated
confidence - not the model.

**Evidence.** Selection runs here, after the action is final, honouring the
rationale's evidence policy.
"""

from __future__ import annotations

from dataclasses import dataclass

from .context_store import ContextStore
from .media import MediaUnderstanding
from .retrieval import EvidenceRetriever
from .schema import (
    ACTION_CONFIDENCE_BANDS,
    FALLBACK_BY_ACTION,
    RATIONALE_BY_CODE,
    Action,
    Decision,
    MessageType,
    Rationale,
)
from .signals import SignalReport


@dataclass
class Proposal:
    """A routing suggestion from either the judge or the deterministic baseline."""

    rationale_code: str
    message_type: MessageType
    evidence_message_ids: list[str]
    source: str
    key_factor: str = ""
    agreement: float = 1.0  # fraction of ensemble samples that agreed


class PolicyArbiter:
    """Applies safety policy and renders the final decision."""

    def __init__(self, store: ContextStore) -> None:
        self.store = store
        self.retriever = EvidenceRetriever(store)

    def finalise(
        self,
        report: SignalReport,
        proposal: Proposal,
        media: MediaUnderstanding | None = None,
    ) -> Decision:
        notes: list[str] = []
        rationale = RATIONALE_BY_CODE.get(proposal.rationale_code)
        if rationale is None:
            # Unknown code: fall back to the safest rationale consistent with a
            # digest, which is the least harmful default for an unclassifiable row.
            rationale = RATIONALE_BY_CODE[FALLBACK_BY_ACTION[Action.DIGEST]]
            notes.append(f"unknown rationale_code {proposal.rationale_code!r}; fell back")

        rationale, override_note = self._apply_safety_floor(report, rationale, media)
        if override_note:
            notes.append(override_note)

        message_type = rationale.resolve_type(proposal.message_type)
        if message_type is not proposal.message_type:
            notes.append(
                f"message_type {proposal.message_type.value} incompatible with "
                f"{rationale.code}; clamped to {message_type.value}"
            )

        confidence = self._confidence(
            report, rationale, message_type, proposal, overridden=bool(override_note)
        )

        evidence = self.retriever.select(
            report.message,
            rationale.action,
            evidence_policy=rationale.evidence_policy,
            preselected=proposal.evidence_message_ids,
        )

        return Decision(
            message_id=report.message.message_id,
            action=rationale.action,
            message_type=message_type,
            reason=rationale.reason,
            confidence=confidence,
            evidence_message_ids=evidence,
            rationale_code=rationale.code,
            source=proposal.source,
            notes=notes,
        )

    # ---------------- safety ----------------

    def _apply_safety_floor(
        self,
        report: SignalReport,
        rationale: Rationale,
        media: MediaUnderstanding | None,
    ) -> tuple[Rationale, str]:
        """Force mute when the deterministic layer proved the message unsafe."""
        risk = report.risk
        media_risky = media is not None and media.is_risky

        unsafe = risk.solicits_credentials or risk.prompt_injection or risk.advance_fee or media_risky
        if not unsafe:
            # A heavily-reported impersonating business is unsafe even when the
            # copy itself is bland - the risk lives in the account, not the words.
            if risk.impersonation >= 0.75 and rationale.action is Action.NOTIFY:
                return (
                    RATIONALE_BY_CODE["MUTE_FAKE_SUPPORT_PRESSURE"],
                    "forced mute: sender is a high-confidence brand impersonator",
                )
            return rationale, ""

        # The judge already decided to suppress; trust its choice of pattern.
        if rationale.action is Action.MUTE:
            return rationale, ""

        forced = self._forced_mute_code(report, media)
        return (
            RATIONALE_BY_CODE[forced],
            f"safety override: {rationale.code} -> {forced} "
            f"({'; '.join(risk.reasons[:2]) or 'unsafe media content'})",
        )

    def _forced_mute_code(self, report: SignalReport, media: MediaUnderstanding | None) -> str:
        """Pick the most specific mute rationale for a proven-unsafe message."""
        risk = report.risk

        # Injection is the most specific finding and has its own labelled pattern.
        if risk.prompt_injection or (media is not None and media.contains_router_instruction):
            return "MUTE_PROMPT_INJECTION"

        credentials = risk.solicits_credentials or (media is not None and media.asks_for_credentials)
        if credentials:
            # No prior contact at all, opening with a credential demand.
            if report.sender.is_first_contact:
                return "MUTE_FIRST_CONTACT_SENSITIVE_ASK"
            # Impersonated support desk leaning on suspension pressure, with no
            # link to click - the sample separates this from the link-driven flow.
            if risk.families.get("account_threat") and not risk.families.get("suspicious_link"):
                return "MUTE_FAKE_SUPPORT_PRESSURE"
            return "MUTE_CREDENTIAL_PHISH"

        if risk.advance_fee or (media is not None and media.asks_for_payment):
            return "MUTE_ADVANCE_FEE_FRAUD"

        return "MUTE_FAKE_SUPPORT_PRESSURE"

    # ---------------- confidence ----------------

    def _confidence(
        self,
        report: SignalReport,
        rationale: Rationale,
        message_type: MessageType,
        proposal: Proposal,
        *,
        overridden: bool,
    ) -> float:
        """Calibrated confidence: the rationale's base, nudged by real uncertainty.

        Adjustments are small and stay inside the action's observed band. They
        exist so the number carries information - a row where three ensemble
        samples disagreed should not look as certain as one where they agreed.
        """
        confidence = rationale.confidence_for(message_type)

        # The sample carries DIGEST_TRUSTED_NO_URGENCY at 0.82 in group traffic
        # and 0.80 one-to-one; a private message has less surrounding context.
        if (
            rationale.code == "DIGEST_TRUSTED_NO_URGENCY"
            and report.message.conversation_type == "personal"
        ):
            confidence -= 0.02

        # Genuine ensemble disagreement is genuine uncertainty.
        if proposal.agreement < 1.0:
            confidence -= 0.02 if proposal.agreement >= 0.6 else 0.04

        # The two stages disagreeing is real uncertainty. The safety floor still
        # wins the decision, but the confidence should not pretend otherwise.
        if overridden:
            confidence -= 0.02

        low, high = ACTION_CONFIDENCE_BANDS[rationale.action]
        return round(min(high, max(low, confidence)), 2)
