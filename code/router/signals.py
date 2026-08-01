"""Deterministic feature extraction: who sent this, how risky is it, does the user care.

Everything here is computed without a model call. It serves three purposes:

* **Hard safety.** Credential solicitation and prompt injection get detected
  deterministically so the arbiter can override the judge. Safety that depends
  on a sampled token is not safety.
* **Prompt evidence.** ``SignalReport.describe()`` renders these findings as the
  factual briefing the judge reasons over, so the model is told what the data
  says rather than left to infer it from raw CSV rows.
* **Standalone routing.** ``baseline.py`` routes from these features alone, which
  gives the pipeline a working fallback when the API is unavailable and gives the
  evaluation an honest no-LLM ablation arm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from .context_store import (
    BusinessAccount,
    ContextStore,
    GroupMembership,
    Message,
    UserBusinessLink,
)
from .lexicon import detect_script, extract_domains, match_families, mentions_user, normalise

# Sender relationship to the receiving user, in descending order of trust.
TRUSTED_GROUP_TYPES = {"family", "extended_family", "coworker", "school_group", "caregiving"}
BROADCAST_GROUP_TYPES = {"marketplace", "local_food", "real_estate", "investment_tips", "society"}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]{3,}", normalise(text)))


def text_similarity(a: str, b: str) -> float:
    """Blend token-set Jaccard with a character-level ratio.

    Jaccard alone misses reworded duplicates; SequenceMatcher alone is fooled by
    shared boilerplate. Together they separate "the same offer resent" from "a
    different message that happens to share a greeting", which is exactly the
    distinction repetition-fatigue routing depends on.
    """
    if not a or not b:
        return 0.0
    ta, tb = _tokens(a), _tokens(b)
    jaccard = len(ta & tb) / len(ta | tb) if ta | tb else 0.0
    ratio = SequenceMatcher(None, normalise(a)[:600], normalise(b)[:600]).ratio()
    return 0.6 * jaccard + 0.4 * ratio


@dataclass
class SenderProfile:
    """Resolved identity and standing of whoever sent this message."""

    kind: str                       # user | business | unknown
    label: str                      # display string used in the prompt
    is_group_admin: bool = False
    group_role: str = ""
    relationship: str = "unknown"   # first_contact | known | trusted | broadcast
    prior_messages: int = 0
    prior_opened: int = 0
    prior_replied: int = 0
    prior_dismissed: int = 0
    prior_muted: int = 0
    prior_reported: int = 0

    @property
    def is_first_contact(self) -> bool:
        return self.prior_messages == 0

    @property
    def engagement_rate(self) -> float:
        if not self.prior_messages:
            return 0.0
        return (self.prior_opened + self.prior_replied) / (2 * self.prior_messages)

    @property
    def rejection_rate(self) -> float:
        if not self.prior_messages:
            return 0.0
        return (self.prior_dismissed + self.prior_muted + self.prior_reported) / (3 * self.prior_messages)

    @property
    def is_habitually_ignored(self) -> bool:
        """User has consistently turned this sender away before."""
        return self.prior_messages >= 2 and self.prior_dismissed >= 2 and self.prior_opened == 0

    @property
    def has_been_reported(self) -> bool:
        return self.prior_reported > 0


@dataclass
class RiskAssessment:
    """Composite safety verdict with the reasons that produced it."""

    score: float = 0.0
    families: dict[str, int] = field(default_factory=dict)
    impersonation: float = 0.0
    reasons: list[str] = field(default_factory=list)

    # Hard flags. These are strong enough to override a model decision.
    solicits_credentials: bool = False
    prompt_injection: bool = False
    advance_fee: bool = False
    chain_forward: bool = False

    @property
    def is_unsafe(self) -> bool:
        """Clear safety risk: mute regardless of the user's usual engagement."""
        return self.solicits_credentials or self.prompt_injection or self.advance_fee or self.score >= 0.65

    @property
    def level(self) -> str:
        if self.score >= 0.65:
            return "high"
        if self.score >= 0.35:
            return "medium"
        if self.score > 0.0:
            return "low"
        return "none"


@dataclass
class EngagementProfile:
    """How this user treats this conversation and their notifications generally."""

    group_muted: bool = False
    group_engagement: float = 0.0
    group_type: str = ""
    group_name: str = ""
    group_is_high_volume: bool = False
    in_quiet_hours: bool = False
    daily_notifications: float = 0.0
    daily_dismissals: float = 0.0
    business_opted_out: bool = False
    business_fatigued: bool = False
    business_transactional: bool = False
    business_reason: str = ""


@dataclass
class RepetitionSignal:
    """Evidence that near-identical content already reached this user."""

    best_match_id: str = ""
    best_score: float = 0.0
    match_outcome: str = ""
    near_duplicate_count: int = 0

    @property
    def is_repeat(self) -> bool:
        return self.best_score >= 0.62

    @property
    def repeat_was_rejected(self) -> bool:
        return self.is_repeat and self.match_outcome in {
            "dismissed without opening",
            "dismissed, then muted the sender",
            "muted the sender afterwards",
            "reported as unsafe",
            "ignored",
        }


@dataclass
class SignalReport:
    """Everything the deterministic layer knows about one incoming message."""

    message: Message
    sender: SenderProfile
    risk: RiskAssessment
    engagement: EngagementProfile
    repetition: RepetitionSignal
    directly_mentions_user: bool = False
    has_urgency: bool = False
    has_action_request: bool = False
    disclaims_urgency: bool = False
    uses_official_channel: bool = False
    is_marketing: bool = False
    is_greeting: bool = False
    script: str = "latin"
    domains: list[str] = field(default_factory=list)
    media_summary: str = ""

    def describe(self) -> str:
        """Render the factual briefing the judge reasons over.

        Written as prose findings rather than a feature dump: the model uses
        natural-language context far more reliably than it uses bare floats.
        """
        m = self.message
        lines: list[str] = []

        lines.append(f"Conversation: {m.conversation_type}")
        if self.engagement.group_name:
            lines.append(
                f"Group: {self.engagement.group_name} (type={self.engagement.group_type}, "
                f"{'high-volume' if self.engagement.group_is_high_volume else 'normal volume'})"
            )
            lines.append(
                f"User's relationship to group: "
                f"{'MUTED by user' if self.engagement.group_muted else 'not muted'}, "
                f"engagement {self.engagement.group_engagement:.0%}"
            )

        lines.append(f"Sender: {self.sender.label}")
        if self.sender.group_role:
            lines.append(
                f"Sender's role in this group: {self.sender.group_role}"
                + ("  (NOTE: admin role alone does not make a payment demand safe)"
                   if self.sender.is_group_admin else "")
            )
        if self.sender.is_first_contact:
            lines.append("Sender history with this user: NO prior messages on record (first contact)")
        else:
            lines.append(
                f"Sender history with this user: {self.sender.prior_messages} prior message(s) - "
                f"{self.sender.prior_opened} opened, {self.sender.prior_replied} replied, "
                f"{self.sender.prior_dismissed} dismissed, {self.sender.prior_muted} muted, "
                f"{self.sender.prior_reported} reported"
            )
            if self.sender.is_habitually_ignored:
                lines.append("  -> This user has consistently ignored or dismissed this sender.")
            if self.sender.has_been_reported:
                lines.append("  -> This user has previously REPORTED this sender as unsafe.")

        if self.engagement.business_reason:
            lines.append(f"Business relationship: {self.engagement.business_reason}")
        if self.engagement.business_opted_out:
            lines.append("  -> User has OPTED OUT of promotions from this business.")
        if self.engagement.business_fatigued:
            lines.append("  -> User dismisses most messages from this business.")
        if self.engagement.business_transactional:
            lines.append("  -> User has a genuine recent order/booking/payment with this business.")

        if m.forwarded_count:
            lines.append(f"Forward count: {m.forwarded_count} (heavily forwarded content)")
        if self.directly_mentions_user:
            lines.append(f"The message directly @-mentions {m.user_id}.")
        if self.script not in {"latin", "none"}:
            lines.append(f"Language/script: {self.script}")

        if self.risk.families:
            lines.append("Content signals detected:")
            from .lexicon import family_label
            for name, count in sorted(self.risk.families.items(), key=lambda kv: -kv[1]):
                lines.append(f"  - {family_label(name)} ({count} pattern(s))")
        if self.domains:
            lines.append(f"Domains in text: {', '.join(self.domains)}")
        if self.risk.reasons:
            lines.append("Risk findings:")
            lines.extend(f"  - {r}" for r in self.risk.reasons)
        lines.append(f"Composite risk: {self.risk.level} ({self.risk.score:.2f})")

        if self.repetition.best_match_id:
            lines.append(
                f"Closest prior message to this user: {self.repetition.best_match_id} "
                f"(similarity {self.repetition.best_score:.2f}); user {self.repetition.match_outcome}"
            )
            if self.repetition.near_duplicate_count > 1:
                lines.append(
                    f"  -> {self.repetition.near_duplicate_count} near-duplicates already reached this user."
                )

        if self.engagement.in_quiet_hours:
            lines.append("Arrived during the user's do-not-disturb window.")
        lines.append(
            f"User's notification load: {self.engagement.daily_notifications:.1f}/day sent, "
            f"{self.engagement.daily_dismissals:.1f}/day dismissed"
        )
        if self.media_summary:
            lines.append(f"Media content:\n{self.media_summary}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #

class SignalExtractor:
    """Builds a :class:`SignalReport` for each incoming message."""

    def __init__(self, store: ContextStore) -> None:
        self.store = store

    def extract(self, message: Message, media_summary: str = "") -> SignalReport:
        families = match_families(message.message_text)
        sender = self._sender_profile(message)
        engagement = self._engagement(message)
        risk = self._risk(message, families, sender, engagement)
        repetition = self._repetition(message)

        return SignalReport(
            message=message,
            sender=sender,
            risk=risk,
            engagement=engagement,
            repetition=repetition,
            directly_mentions_user=mentions_user(message.message_text, message.user_id),
            has_urgency="urgency" in families,
            has_action_request="action_request" in families,
            disclaims_urgency="no_action_marker" in families,
            uses_official_channel="official_channel" in families,
            is_marketing="marketing" in families,
            is_greeting="greeting" in families,
            script=detect_script(message.message_text),
            domains=extract_domains(message.message_text),
            media_summary=media_summary,
        )

    # ---------------- sender ----------------

    def _sender_profile(self, message: Message) -> SenderProfile:
        store = self.store
        if message.business_id:
            business = store.business(message.business_id)
            history = store.business_history(message.user_id, message.business_id)
            profile = self._reaction_counts(message.user_id, history)
            profile.kind = "business"
            profile.label = self._business_label(business)
            link = store.business_link(message.user_id, message.business_id)
            profile.relationship = "known" if link else "first_contact"
            return profile

        if message.sender_user_id:
            history = store.sender_history(message.user_id, message.sender_user_id)
            profile = self._reaction_counts(message.user_id, history)
            profile.kind = "user"
            profile.label = f"user {message.sender_user_id}"
            membership = store.membership(message.group_id, message.sender_user_id)
            if membership is not None:
                profile.is_group_admin = membership.is_admin
                profile.group_role = membership.role
            group = store.group(message.group_id)
            if group and group.group_type in TRUSTED_GROUP_TYPES:
                profile.relationship = "trusted" if profile.prior_messages else "known"
            elif group and group.group_type in BROADCAST_GROUP_TYPES:
                profile.relationship = "broadcast"
            elif profile.prior_messages:
                profile.relationship = "known"
            else:
                profile.relationship = "first_contact"
            return profile

        return SenderProfile(kind="unknown", label="unidentified sender")

    def _reaction_counts(self, user_id: str, history: list[Message]) -> SenderProfile:
        counts = self.store.reaction_profile(user_id, history)
        return SenderProfile(
            kind="",
            label="",
            prior_messages=counts["total"],
            prior_opened=counts["opened"],
            prior_replied=counts["replied"],
            prior_dismissed=counts["dismissed"],
            prior_muted=counts["muted"],
            prior_reported=counts["reported"],
        )

    @staticmethod
    def _business_label(business: BusinessAccount | None) -> str:
        if business is None:
            return "unknown business account"
        bits = [
            f"business '{business.display_name}' (brand={business.brand_name}, "
            f"category={business.category})",
            "VERIFIED" if business.verified else "NOT verified",
            f"account age {business.account_age_days}d",
            f"sends from {business.domain_used_by_sender or 'no domain'}"
            + (f" but official domain is {business.official_domain}" if business.domain_mismatch else ""),
            f"{business.user_reports_30d} user reports in 30d "
            f"({business.report_rate_per_1k:.1f} per 1k messages)",
        ]
        return "; ".join(bits)

    # ---------------- engagement ----------------

    def _engagement(self, message: Message) -> EngagementProfile:
        store = self.store
        profile = EngagementProfile()

        user = store.user(message.user_id)
        if user is not None:
            profile.in_quiet_hours = user.in_quiet_hours(message.timestamp)
        profile.daily_notifications, profile.daily_dismissals = store.notification_load(message.user_id)

        group = store.group(message.group_id)
        if group is not None:
            profile.group_type = group.group_type
            profile.group_name = group.group_name
            profile.group_is_high_volume = group.is_high_volume
            membership: GroupMembership | None = store.membership(message.group_id, message.user_id)
            if membership is not None:
                profile.group_muted = membership.group_muted_by_user
                profile.group_engagement = membership.engagement

        link: UserBusinessLink | None = store.business_link(message.user_id, message.business_id)
        if link is not None:
            profile.business_opted_out = link.opted_out
            profile.business_fatigued = link.is_fatigued
            profile.business_transactional = link.is_transactional
            profile.business_reason = (
                f"{link.why_user_knows_account}, last activity {link.last_activity_at or 'unknown'}, "
                f"{link.activity_count_180d} activities in 180d, "
                f"{link.messages_opened_30d} opened / {link.messages_dismissed_30d} dismissed in 30d"
            )
        elif message.business_id:
            profile.business_reason = "no prior relationship with this business on record"
        return profile

    # ---------------- risk ----------------

    def _risk(
        self,
        message: Message,
        families: dict[str, int],
        sender: SenderProfile,
        engagement: EngagementProfile,
    ) -> RiskAssessment:
        risk = RiskAssessment(families=dict(families))
        score = 0.0

        if families.get("credential_solicitation"):
            risk.solicits_credentials = True
            score += 0.55
            risk.reasons.append("solicits an OTP, PIN, password or card detail")

        if families.get("prompt_injection"):
            risk.prompt_injection = True
            score += 0.50
            risk.reasons.append(
                "contains text addressed to the notification system itself, attempting to "
                "dictate the routing decision"
            )

        if families.get("advance_fee"):
            risk.advance_fee = True
            score += 0.40
            risk.reasons.append("demands an upfront fee, token amount or QR payment before delivery")

        if families.get("account_threat"):
            score += 0.20
            risk.reasons.append("threatens account suspension or a closing window")

        if families.get("prize_lure"):
            score += 0.25
            risk.reasons.append("claims the user won or was selected for a reward")

        if families.get("suspicious_link"):
            score += 0.20
            risk.reasons.append("directs the user to a shortener or lookalike domain")

        if families.get("health_misinfo"):
            score += 0.20
            risk.reasons.append("promotes unverified medical advice")

        if families.get("chain_forward"):
            risk.chain_forward = True
            score += 0.15
            risk.reasons.append("uses chain-letter framing to pressure onward forwarding")

        # Forwarding depth. Legitimate admins occasionally relay a notice, so this
        # is a nudge rather than a verdict on its own.
        if message.forwarded_count >= 8:
            score += 0.15
            risk.reasons.append(f"forwarded {message.forwarded_count} times")
        elif message.forwarded_count >= 4:
            score += 0.08

        # Business impersonation.
        business = self.store.business(message.business_id)
        if business is not None:
            risk.impersonation = business.impersonation_score
            if risk.impersonation >= 0.6:
                score += 0.35
                risk.reasons.append(
                    f"sender impersonates {business.brand_name}: unverified, "
                    f"{business.account_age_days}d-old account on {business.domain_used_by_sender}"
                )
            elif risk.impersonation >= 0.4:
                score += 0.15
                risk.reasons.append("business account shows several impersonation indicators")

        # First contact that opens with a sensitive demand is the classic setup.
        if sender.is_first_contact and (risk.solicits_credentials or risk.advance_fee):
            score += 0.20
            risk.reasons.append("no prior contact with this sender, yet it already demands money or a code")

        # The user's own past verdict on this sender is strong evidence.
        if sender.has_been_reported:
            score += 0.20
            risk.reasons.append("this user has previously reported this sender")

        # A message can only be reassuring if it is not simultaneously extracting.
        if families.get("reassurance") and not (risk.solicits_credentials or risk.advance_fee):
            score -= 0.10
            risk.reasons.append("explicitly warns the user against sharing credentials or paying via links")

        if families.get("official_channel") and not risk.advance_fee:
            score -= 0.10
            risk.reasons.append("routes the user through an official app or office rather than an ad-hoc link")

        risk.score = max(0.0, min(1.0, score))
        return risk

    # ---------------- repetition ----------------

    def _repetition(self, message: Message) -> RepetitionSignal:
        """Find the closest thing this user has already received."""
        signal = RepetitionSignal()
        if not message.message_text:
            # Voice notes carry no text; repetition is judged from the sender's
            # own track record instead, which _sender_profile already captures.
            return signal

        best_score = 0.0
        best: Message | None = None
        near_duplicates = 0

        for prior in self.store.user_history(message.user_id):
            if not prior.message_text:
                continue
            score = text_similarity(message.message_text, prior.message_text)
            if score >= 0.62:
                near_duplicates += 1
            if score > best_score:
                best_score, best = score, prior

        if best is not None:
            signal.best_match_id = best.message_id
            signal.best_score = best_score
            signal.near_duplicate_count = near_duplicates
            event = self.store.event(message.user_id, best.message_id)
            signal.match_outcome = event.describe() if event else "no recorded reaction"
        return signal
