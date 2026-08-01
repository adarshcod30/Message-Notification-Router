"""Evidence retrieval: find the historical precedent that justifies a decision.

``evidence_message_ids`` is graded on whether the cited ids "point to relevant
historical messages". Measuring the labelled sample showed that relevance here is
**behavioural, not lexical**: ranking a user's history by text similarity puts the
organizer's chosen evidence first in only 8 of 28 rows, whereas the reaction
recorded against that evidence agrees with the action in 26 of 28.

    notify  <- a past message this user opened AND replied to
    digest  <- a past message this user opened but did not reply to
    mute    <- a past message this user dismissed, muted, or reported

So evidence is not "the most similar thing that happened before". It is *the
precedent that shows how this user treats messages like this one*. Retrieval
therefore runs after the action is chosen and conditions on it.

Scoring blends four terms, in descending weight:

1. **Structural affinity** - same sender, business, or group.
2. **Outcome consistency** - does the recorded reaction support this action.
3. **Topical similarity** - lexical overlap, still useful as a tiebreak.
4. **Recency** - newer precedents beat older ones, all else equal.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import ROUTER
from .context_store import ContextStore, Message, MessageEvent
from .schema import Action
from .signals import text_similarity


# How well each recorded outcome supports each action, in [-1, 1].
# Derived from the outcome/action agreement measured on the labelled sample.
_OUTCOME_FIT: dict[Action, dict[str, float]] = {
    Action.NOTIFY: {"engaged": 1.00, "seen": 0.25, "rejected": -0.60, "reported": -0.90},
    Action.DIGEST: {"engaged": 0.35, "seen": 1.00, "rejected": -0.10, "reported": -0.70},
    Action.MUTE:   {"engaged": -0.70, "seen": 0.10, "rejected": 1.00, "reported": 1.00},
}


# Minimum structural affinity for a candidate to count as a precedent at all.
# 0.28 is "same conversation type"; anything below that is coincidence, and the
# labelled sample's first abstention row tops out at 0.12.
_MIN_AFFINITY = 0.25


def classify_outcome(event: MessageEvent | None) -> str:
    """Bucket a recorded reaction into one of four behavioural outcomes."""
    if event is None:
        return "unknown"
    if event.message_reported:
        return "reported"
    if event.notification_dismissed or event.muted_after_message:
        return "rejected"
    if event.message_replied:
        return "engaged"
    if event.message_opened:
        return "seen"
    return "rejected"


@dataclass
class EvidenceCandidate:
    """One scored historical message, with the breakdown kept for the audit trail."""

    message: Message
    score: float
    outcome: str
    outcome_text: str
    affinity: str
    similarity: float

    def brief(self, limit: int = 150) -> str:
        """One-line rendering for the judge prompt."""
        text = (self.message.message_text or "(voice note, no text)").replace("\n", " ")
        if len(text) > limit:
            text = text[: limit - 1] + "…"
        return (
            f"{self.message.message_id} [{self.affinity}; user {self.outcome_text}] "
            f'"{text}"'
        )


class EvidenceRetriever:
    """Selects supporting evidence from the receiving user's own history."""

    def __init__(self, store: ContextStore) -> None:
        self.store = store

    # ---------------- candidate generation ----------------

    def candidates(self, message: Message, limit: int | None = None) -> list[EvidenceCandidate]:
        """Rank this user's history by action-independent relevance.

        Used to build the judge prompt, before an action exists. Structural
        affinity and topical similarity only.
        """
        limit = limit or ROUTER.evidence_candidates
        scored: list[EvidenceCandidate] = []

        for prior in self.store.user_history(message.user_id):
            affinity, affinity_score = self._affinity(message, prior)
            similarity = (
                text_similarity(message.message_text, prior.message_text)
                if message.message_text and prior.message_text
                else 0.0
            )
            # A same-sender precedent with no lexical overlap is still highly
            # relevant; an unrelated message that shares stock phrasing is not.
            base = 0.62 * affinity_score + 0.38 * similarity
            if base <= 0.0:
                continue
            event = self.store.event(message.user_id, prior.message_id)
            scored.append(
                EvidenceCandidate(
                    message=prior,
                    score=base + 0.04 * self._recency(message, prior),
                    outcome=classify_outcome(event),
                    outcome_text=event.describe() if event else "no recorded reaction",
                    affinity=affinity,
                    similarity=similarity,
                )
            )

        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:limit]

    # ---------------- final selection ----------------

    def select(
        self,
        message: Message,
        action: Action,
        *,
        evidence_policy: str = "single",
        preselected: list[str] | None = None,
    ) -> list[str]:
        """Choose the evidence ids to emit for a decided action.

        ``evidence_policy`` comes from the chosen rationale and is authoritative.
        A rationale whose sentence asserts "this is the first message from the
        sender" must not cite that sender's history, however good the match -
        the emitted reason and the emitted evidence have to agree.

        ``preselected`` are ids the judge proposed. They are honoured when they
        exist and are genuinely this user's history, but they are re-scored
        alongside everything else rather than trusted blindly - a model will
        happily cite a plausible-looking id it never actually read.
        """
        if evidence_policy == "none":
            return []

        pool = self.candidates(message, limit=max(ROUTER.evidence_candidates, 12))
        if not pool:
            return []

        by_id = {c.message.message_id: c for c in pool}
        for mid in preselected or []:
            if mid in by_id or mid not in self.store.history:
                continue
            prior = self.store.history[mid]
            # Only admissible if it really is this user's own history.
            if prior.user_id != message.user_id:
                continue
            affinity, affinity_score = self._affinity(message, prior)
            event = self.store.event(message.user_id, mid)
            by_id[mid] = EvidenceCandidate(
                message=prior,
                score=0.62 * affinity_score
                + 0.38 * text_similarity(message.message_text, prior.message_text),
                outcome=classify_outcome(event),
                outcome_text=event.describe() if event else "no recorded reaction",
                affinity=affinity,
                similarity=text_similarity(message.message_text, prior.message_text),
            )

        proposed = set(preselected or [])
        fit = _OUTCOME_FIT[action]
        ranked = sorted(
            by_id.values(),
            key=lambda c: (
                c.score
                + 0.55 * fit.get(c.outcome, 0.0)
                # A small nudge toward what the judge picked, enough to break a
                # tie in its favour without letting it override a better match.
                + (0.10 if c.message.message_id in proposed else 0.0)
            ),
            reverse=True,
        )

        wanted = 2 if evidence_policy == "pair" else 1
        chosen: list[EvidenceCandidate] = []
        for candidate in ranked:
            if candidate.score < ROUTER.evidence_min_score:
                continue
            # Structural floor: a message that merely shares a conversation type
            # is not a precedent, it is a coincidence. Abstaining beats padding.
            if self._affinity(message, candidate.message)[1] < _MIN_AFFINITY:
                continue
            # Never cite a precedent whose recorded behaviour contradicts the
            # decision - that is worse than citing nothing.
            if fit.get(candidate.outcome, 0.0) < 0.0:
                continue
            chosen.append(candidate)
            if len(chosen) >= wanted:
                break

        return [c.message.message_id for c in chosen[: ROUTER.evidence_max_emitted]]

    # ---------------- scoring parts ----------------

    def _affinity(self, message: Message, prior: Message) -> tuple[str, float]:
        """Structural closeness of a historical message to the incoming one."""
        if message.business_id and prior.business_id == message.business_id:
            return "same business", 1.0
        if message.sender_user_id and prior.sender_user_id == message.sender_user_id:
            if message.group_id and prior.group_id == message.group_id:
                return "same sender, same group", 1.0
            return "same sender", 0.92
        if message.group_id and prior.group_id == message.group_id:
            return "same group", 0.70
        if (
            message.business_id
            and prior.business_id
            and self._same_brand(message.business_id, prior.business_id)
        ):
            return "same brand, different account", 0.55
        if prior.conversation_type == message.conversation_type:
            return "same conversation type", 0.28
        return "same user history", 0.12

    def _same_brand(self, a: str, b: str) -> bool:
        ba, bb = self.store.business(a), self.store.business(b)
        return bool(ba and bb and ba.brand_name and ba.brand_name == bb.brand_name)

    @staticmethod
    def _recency(message: Message, prior: Message) -> float:
        """1.0 for same-day, decaying to 0 across roughly three months."""
        now, then = message.timestamp, prior.timestamp
        if now is None or then is None:
            return 0.0
        days = abs((now - then).days)
        return max(0.0, 1.0 - days / 90.0)
