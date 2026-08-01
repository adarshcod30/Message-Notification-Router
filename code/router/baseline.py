"""Zero-LLM routing from the deterministic signals alone.

This module earns its place three times over:

* **Availability.** The run happens against a rate-limited free-tier key. If quota
  runs out mid-pass the pipeline still emits a complete, defensible
  ``output.csv`` instead of a partial file or a crash.
* **Honest ablation.** ``main.py ablate`` scores this arm against the full
  pipeline, which is how we can state what the LLM is actually contributing
  rather than assuming it helps.
* **A second opinion.** Where baseline and judge disagree, that disagreement is
  logged - it is the fastest way to find prompt bugs and genuinely hard rows.

The rules are ordered most-decisive first and read as a cascade. Each returns a
rationale code from the same taxonomy the judge selects from, so both arms are
directly comparable and both flow through the same arbiter.
"""

from __future__ import annotations

from .media import MediaUnderstanding
from .schema import Action, MessageType
from .signals import SignalReport

# Group types where a same-day operational notice genuinely warrants interrupting.
_OPERATIONAL_GROUPS = {"society", "school_group", "college_faculty", "safety"}
_WORK_GROUPS = {"coworker"}
_FAMILY_GROUPS = {"family", "extended_family", "caregiving"}


def route(report: SignalReport, media: MediaUnderstanding | None = None) -> tuple[str, MessageType, str]:
    """Return ``(rationale_code, message_type, key_factor)`` from signals alone."""
    message = report.message
    risk = report.risk
    sender = report.sender
    engagement = report.engagement
    text_urgency = report.has_urgency and not report.disclaims_urgency
    media_urgent = media is not None and media.urgency in {"high", "medium"}

    # ---------------- 1. Safety, in descending specificity ----------------

    if risk.prompt_injection or (media is not None and media.contains_router_instruction):
        return "MUTE_PROMPT_INJECTION", MessageType.SCAM, "text instructs the router itself"

    credentials = risk.solicits_credentials or (media is not None and media.asks_for_credentials)
    if credentials:
        if sender.is_first_contact:
            return (
                "MUTE_FIRST_CONTACT_SENSITIVE_ASK", MessageType.SCAM,
                "first contact opens with a credential demand",
            )
        if risk.families.get("account_threat") and not risk.families.get("suspicious_link"):
            return (
                "MUTE_FAKE_SUPPORT_PRESSURE", MessageType.SCAM,
                "fake support desk plus suspension pressure",
            )
        return "MUTE_CREDENTIAL_PHISH", MessageType.SCAM, "solicits an OTP or account credential"

    if risk.advance_fee or (media is not None and media.asks_for_payment and media.shows_qr_or_link):
        return (
            "MUTE_ADVANCE_FEE_FRAUD", MessageType.SCAM,
            "demands payment up front before delivery",
        )

    if risk.families.get("prize_lure"):
        return "MUTE_ADVANCE_FEE_FRAUD", MessageType.SCAM, "unsolicited prize or reward claim"

    if risk.impersonation >= 0.6:
        return (
            "MUTE_FAKE_SUPPORT_PRESSURE", MessageType.SCAM,
            "business account is impersonating a known brand",
        )

    # ---------------- 2. Noise the user has already rejected ----------------

    if risk.chain_forward or (message.forwarded_count >= 8 and (report.is_greeting or risk.families.get("health_misinfo"))):
        mtype = MessageType.GREETING if report.is_greeting else MessageType.FORWARD
        if sender.is_habitually_ignored:
            return "MUTE_REPEAT_FORWARD_SENDER", mtype, "known forwarder this user ignores"
        return "MUTE_CHAIN_FORWARD_NOISE", mtype, f"chain forward, forwarded {message.forwarded_count}x"

    if sender.is_habitually_ignored and not text_urgency and not report.directly_mentions_user:
        mtype = MessageType.GREETING if report.is_greeting else (
            MessageType.PROMOTION if report.is_marketing else MessageType.FORWARD
        )
        return "MUTE_REPEAT_FORWARD_SENDER", mtype, "user has ignored every prior message from this sender"

    if report.repetition.repeat_was_rejected:
        return (
            "MUTE_HISTORICALLY_IGNORED",
            MessageType.PROMOTION if report.is_marketing else MessageType.FORWARD,
            f"near-duplicate of {report.repetition.best_match_id}, which the user rejected",
        )

    # ---------------- 3. Business traffic ----------------

    if message.business_id:
        return _route_business(report, media)

    # ---------------- 4. Direct, personal urgency ----------------

    if report.directly_mentions_user or message.conversation_type == "personal":
        return _route_direct(report, media, text_urgency or media_urgent)

    # ---------------- 5. Group traffic ----------------

    return _route_group(report, media, text_urgency or media_urgent)


# --------------------------------------------------------------------------- #

def _route_business(report: SignalReport, media: MediaUnderstanding | None) -> tuple[str, MessageType, str]:
    engagement = report.engagement
    marketing = report.is_marketing or (media is not None and media.category == "promotional_poster")

    if marketing:
        if engagement.business_opted_out or engagement.business_fatigued:
            return (
                "MUTE_MARKETING_OPTED_OUT",
                MessageType.PROMOTION if engagement.business_reason else MessageType.SPAM,
                "user opted out of or routinely dismisses this sender's marketing",
            )
        if not engagement.business_reason or engagement.business_reason.startswith("no prior"):
            return (
                "MUTE_UNSOLICITED_BULK_PROMO", MessageType.SPAM,
                "cold marketing from a business the user has no relationship with",
            )
        return (
            "DIGEST_OPTED_IN_PROMOTION", MessageType.PROMOTION,
            "promotional, but the user still allows promotions from this business",
        )

    # Transactional traffic from a business the user genuinely deals with.
    if engagement.business_transactional and report.risk.score < 0.3:
        reason = engagement.business_reason
        if any(k in reason for k in ("booking", "appointment")):
            return (
                "NOTIFY_BUSINESS_BOOKING_REMINDER", MessageType.EVENT,
                "verified business reminder matching a real booking",
            )
        if report.has_urgency:
            return (
                "NOTIFY_BUSINESS_ORDER_UPDATE", MessageType.BUSINESS_UPDATE,
                "verified business update matching a recent order",
            )

    return (
        "DIGEST_VERIFIED_BUSINESS_NON_URGENT", MessageType.BUSINESS_UPDATE,
        "legitimate business message with nothing due",
    )


def _route_direct(
    report: SignalReport, media: MediaUnderstanding | None, urgent: bool
) -> tuple[str, MessageType, str]:
    sender = report.sender
    group_type = report.engagement.group_type

    if sender.is_first_contact and report.message.conversation_type == "personal":
        return (
            "DIGEST_UNFAMILIAR_BUT_SAFE", MessageType.UNKNOWN,
            "unknown sender, but no urgency or payment pressure",
        )

    if urgent and report.has_action_request:
        if group_type in _WORK_GROUPS or (
            sender.prior_replied >= 2 and group_type not in _FAMILY_GROUPS
            and report.message.conversation_type != "personal"
        ):
            return (
                "NOTIFY_WORK_DEADLINE", MessageType.URGENT,
                "work context with a deadline the user is blocking",
            )
        return (
            "NOTIFY_CLOSE_CONTACT_URGENT", MessageType.URGENT,
            "close contact needs an answer within the hour",
        )

    if report.directly_mentions_user and report.has_action_request:
        return (
            "NOTIFY_DIRECT_REQUEST", MessageType.PERSONAL,
            "message names this user and asks them to act",
        )

    if report.is_greeting:
        return "DIGEST_HARMLESS_GREETING", MessageType.GREETING, "greeting with no action attached"

    if report.disclaims_urgency:
        return (
            "DIGEST_TRUSTED_NO_URGENCY", MessageType.PERSONAL,
            "trusted sender, explicitly nothing needed now",
        )

    return "DIGEST_CASUAL_CHAT", MessageType.PERSONAL, "safe personal chat with no deadline"


def _route_group(
    report: SignalReport, media: MediaUnderstanding | None, urgent: bool
) -> tuple[str, MessageType, str]:
    sender = report.sender
    engagement = report.engagement
    group_type = engagement.group_type

    if urgent and sender.is_group_admin:
        if group_type in {"school_group", "college_faculty"}:
            return (
                "NOTIFY_SCHOOL_SAME_DAY", MessageType.EVENT,
                "school admin posted a same-day operational change",
            )
        if group_type in _OPERATIONAL_GROUPS:
            # A genuine dues reminder through an official channel is payment,
            # not a generic alert.
            if report.uses_official_channel and "payment" in report.message.message_text.lower()[:200]:
                return (
                    "NOTIFY_LEGITIMATE_PAYMENT_DEADLINE", MessageType.PAYMENT,
                    "admin payment deadline via an official channel",
                )
            return (
                "NOTIFY_ADMIN_TIME_SENSITIVE", MessageType.URGENT,
                "trusted admin posted a time-sensitive notice",
            )

    if urgent and group_type in _WORK_GROUPS:
        return (
            "NOTIFY_WORK_DEADLINE", MessageType.URGENT,
            "work group message with an immediate dependency",
        )

    if report.is_greeting:
        if sender.is_habitually_ignored or report.message.forwarded_count >= 4:
            return "MUTE_REPEAT_FORWARD_SENDER", MessageType.GREETING, "repeat forwarder"
        return "DIGEST_HARMLESS_GREETING", MessageType.GREETING, "harmless group greeting"

    if report.is_marketing or engagement.group_type in {"marketplace", "local_food", "real_estate"}:
        if report.repetition.is_repeat:
            return (
                "MUTE_HISTORICALLY_IGNORED", MessageType.PROMOTION,
                "this listing already reached the user",
            )
        return (
            "DIGEST_RELEVANT_OFFER", MessageType.PROMOTION,
            "marketplace listing, potentially useful but not urgent",
        )

    # A muted group still lets a direct mention through, handled earlier; anything
    # else in a muted or low-engagement group is digest at best.
    if engagement.group_muted or engagement.group_engagement < 0.35:
        return (
            "DIGEST_USEFUL_GROUP_INFO", MessageType.EVENT,
            "muted or low-engagement group; nothing needs interrupting",
        )

    if sender.is_group_admin or group_type in _OPERATIONAL_GROUPS:
        return (
            "DIGEST_USEFUL_GROUP_INFO", MessageType.EVENT,
            "useful group notice with a deadline days away",
        )

    return "DIGEST_CASUAL_CHAT", MessageType.PERSONAL, "ordinary group chatter"


def propose(report: SignalReport, media: MediaUnderstanding | None = None):
    """Adapt :func:`route` to the arbiter's ``Proposal`` shape."""
    from .arbiter import Proposal

    code, message_type, key_factor = route(report, media)
    return Proposal(
        rationale_code=code,
        message_type=message_type,
        evidence_message_ids=[],
        source="baseline",
        key_factor=key_factor,
    )
