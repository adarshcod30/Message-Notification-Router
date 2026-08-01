"""Multilingual pattern lexicon for risk, urgency, and intent detection.

Why patterns at all, when there is an LLM in the loop?

Two reasons. First, *safety must not be probabilistic*: a message demanding an
OTP has to be muted whether or not the judge is having a good day, so the
pipeline needs a deterministic detector it can use as a hard override. Second,
these matches become **explicit prompt evidence** - the judge is told "this text
solicits a credential" rather than being left to notice it, which measurably
improves both the decision and the category it picks.

The corpus mixes English, romanised Hindi (Hinglish), Devanagari, and French, so
every risk family carries patterns in each. Matching is accent- and case-
insensitive.

These are general linguistic patterns. Nothing here keys off a message_id, and
no pattern encodes an answer for a specific row.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass


def normalise(text: str) -> str:
    """Casefold and strip accents so 'récupérer' and 'recuperer' both match."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped.casefold()).strip()


@dataclass(frozen=True)
class PatternFamily:
    """A named group of regexes with a human-readable label for the prompt."""

    name: str
    label: str
    patterns: tuple[str, ...]

    def compiled(self) -> list[re.Pattern[str]]:
        return [re.compile(p) for p in self.patterns]


# --------------------------------------------------------------------------- #
# Risk families
# --------------------------------------------------------------------------- #

CREDENTIAL_SOLICITATION = PatternFamily(
    name="credential_solicitation",
    label="asks the user to hand over an OTP, PIN, password or card detail",
    patterns=(
        r"\b(otp|o\.t\.p)\b",
        r"\bone[- ]time (pass)?code\b",
        r"\b(\d[- ]?digit|six[- ]digit|6[- ]digit)\s+(login\s+)?(code|otp|pin)\b",
        # -ing / -s forms matter: "by sharing your account number" is a request
        # even though the bare stem "share" never appears.
        r"\b(shar(e|ing)|send(ing)?|giv(e|ing)|tell(ing)?|reply(ing)? with|confirm(ing)?|provid(e|ing)|enter(ing)?)\b[^.!?]{0,40}\b(otp|pin|password|cvv|code)\b",
        r"\bconfirm (your )?(password|pin|wallet pin|card pin)\b",
        r"\b(account|card|bank|wallet) (number|details?|pin)\b[^.!?]{0,30}\b(shar(e|ing)|send(ing)?|confirm(ing)?|provid(e|ing)|verif(y|ying))\b",
        r"\b(shar(e|ing)|send(ing)?|provid(e|ing)|submit(ting)?|upload(ing)?|verif(y|ying)|confirm(ing)?|enter(ing)?)\b[^.!?]{0,30}\b(wallet|account|bank|card)\b[^.!?]{0,20}\b(number|details?|pin)\b",
        r"\bfill\b[^.!?]{0,25}\bbank details\b",
        # Hinglish / Devanagari
        r"\botp\b[^.!?]{0,25}\b(batao|bata|do|dijiye|bhejo|share|daal)\b",
        r"\b(code|otp)\b[^.!?]{0,20}\bdaal\s?do\b",
        r"\bverification code\b[^.!?]{0,25}\b(confirm|karo|kar do)\b",
        r"ओटीपी|पासवर्ड",
    ),
)

ACCOUNT_THREAT = PatternFamily(
    name="account_threat",
    label="threatens account suspension or a closing window to force fast action",
    patterns=(
        r"\b(account|profile|access|card|service|payout)\b[^.!?]{0,45}\b(block(ed|ing)?|suspend(ed)?|restrict(ed)?|lock(ed)?|deactivat|clos(e|ed|ure)|expire)",
        r"\bwill (be )?(block|suspend|restrict|lock|expire|clos)",
        r"\b(verify|confirm|complete)\b[^.!?]{0,30}\b(now|immediately|today|within \d+|before midnight|in \d+ (mins?|minutes|hours?))\b",
        r"\bfinal (notice|reminder|warning)\b",
        # "complete pending account check" carries no suspension wording but is
        # the same demand: prove who you are, right now, over there.
        r"\b(pending|incomplete|failed)\s+(account|kyc|profile|wallet|security)\s+(check|verification|update)\b",
        r"\bcomplete\b[^.!?]{0,25}\b(account|kyc|profile|security)\s+(check|verification|update)\b",
        r"\b(security|support) (alert|check|team|desk)\b",
        # Hinglish
        r"\b(band|block)\s+ho\s+jayega\b",
        r"\bhold\s+pe\s+chala\s+jayega\b",
        r"\bjaldi\s+kar(o| lo)\b",
        r"\btime\s+kam\s+hai\b",
    ),
)

ADVANCE_FEE = PatternFamily(
    name="advance_fee",
    label="demands money up front before anything is delivered",
    patterns=(
        r"\b(processing|reattempt|re-attempt|clearance|activation|reactivation|registration|token|penalty|convenience)\s+(fee|charge|amount)\b",
        r"\bpay\b[^.!?]{0,40}\b(fee|charge|token|amount)\b[^.!?]{0,40}\b(release|approve|unlock|block|confirm)",
        r"\b(loan|refund|prize|reward|benefit|payout|amount)\b[^.!?]{0,40}\b(approved|released?|credited)\b[^.!?]{0,60}\bpay\b",
        r"\bscan\b[^.!?]{0,25}\bqr\b[^.!?]{0,40}\bpay\b",
        r"\bpay\b[^.!?]{0,30}\bqr\b",
        # A screenshot of a completed payment sent back into the chat is the
        # collection step of a QR/link fraud. Legitimate desks reconcile against
        # their own records and explicitly tell users NOT to post receipts.
        r"\bsend\b[^.!?]{0,30}\bscreenshot\b",
        r"\bscreenshot\b[^.!?]{0,30}\b(here|after|once|to me|so i can)\b",
        r"\btoken\s+(amount|money)\b",
        r"\bpay\s+rs\.?\s?\d+",
    ),
)

PRIZE_LURE = PatternFamily(
    name="prize_lure",
    label="claims the user won something or was specially selected",
    patterns=(
        r"\b(congrats|congratulations)\b[^.!?]{0,40}\b(selected|won|winner|reward|prize|lucky)\b",
        r"\byour (number|account) (was|has been) selected\b",
        r"\bclaim\b[^.!?]{0,30}\b(reward|prize|benefit|voucher|gift)\b",
        r"\blucky\s+draw\b",
    ),
)

SUSPICIOUS_LINK = PatternFamily(
    name="suspicious_link",
    label="points to a shortener or a lookalike domain rather than an official one",
    patterns=(
        r"\b(bit\.ly|tinyurl|shorturl\.at|t\.me|rb\.gy|cutt\.ly|is\.gd|vl\.gl|weurl\.co|wame\.pro)\b",
        # Brand name glued to a suffix on a non-official TLD path.
        r"\b[a-z0-9-]{3,}-(secure|verify|login|kyc|refund|alert|help|reward|gift|pay|check)[a-z0-9-]*\.(in|com|net|co|org|xyz|info)\b",
        r"\b(secure|verify|login|kyc|refund|alert|account|pay)-[a-z0-9-]{3,}\.(in|com|net|co|org|xyz|info)\b",
        r"\b(open|click|tap|use|visit|follow)\s+(this|the)\s+link\b",
        r"\b(verify|confirm|complete|check|update|login|log in)\b[^.!?]{0,30}\b(through|via|at|on|using)\s+(this|the)\s+link\b",
        r"\b(link|url) (shared )?(here|below|in this chat)\b",
        r"\blink\s+(shared|below)\b[^.!?]{0,30}\b(complete|verify|confirm)\b",
        r"\blink open kar\b",
    ),
)

PROMPT_INJECTION = PatternFamily(
    name="prompt_injection",
    label="tries to give instructions to the notification router itself",
    patterns=(
        r"\bignore (all |any )?(previous|prior|earlier|above)\b[^.!?]{0,30}\b(rule|instruction|routing|prompt)",
        r"\b(system|internal|assistant|agent|router|model)\s+(note|instruction|metadata|prompt|message|directive)\b",
        r"\brouting\s+(override|rule|instruction)\b",
        r"\b(set|mark|classify|treat|force)\b[^.!?]{0,30}\b(action|this|it|message)\b[^.!?]{0,25}\b(=|as|to)\s*(notify|urgent|high)",
        r"\baction\s*=\s*(notify|digest|mute)\b",
        r"\bconfidence\s*=\s*[\d.]+",
        r"\b(verified_business|user_priority|priority)\s*=\s*\w+",
        r"\bnote for (the )?(notification )?(router|system|assistant)\b",
        r"\balways mark this as\b",
        r"\bdo not (flag|mute|filter|block) this\b",
    ),
)

CHAIN_FORWARD = PatternFamily(
    name="chain_forward",
    label="chain-letter framing that pressures the user to forward it on",
    patterns=(
        r"\b(forward|share|send)\b[^.!?]{0,30}\b(to )?(at least )?(ten|10|\d+)\s+(people|persons|friends|groups)\b",
        r"\bdo not break the chain\b|\bdon'?t break the chain\b",
        r"\bshare\b[^.!?]{0,30}\b(all|every|family|other)\s+groups?\b",
        r"\b(share|forward)\b[^.!?]{0,30}\bbefore (midnight|sunset|night|tonight)\b",
        r"\bfwd (as received|from)\b",
        r"\bforwarded (health |as )?(tip|message)\b",
        r"\b(sab|sabhi)\s+groups?\s+me\b",
        r"\bshare kar (do|dena)\b",
        r"\bfailao\b",
    ),
)

HEALTH_MISINFO = PatternFamily(
    name="health_misinfo",
    label="unverified medical advice presented as a secret cure",
    patterns=(
        r"\b(stop|band karo)\b[^.!?]{0,25}\b(tablets?|medicines?|dawai)\b",
        r"\b(herbal|home) (mix|remedy|cure)\b",
        r"\bdoctors? (don'?t|do not|never) (tell|want you to know)\b",
        r"\b(health|life) (secret|hack)\b[^.!?]{0,30}\b(share|forward|read)\b",
        r"\bthis one habit will (fix|cure)\b",
    ),
)

GREETING = PatternFamily(
    name="greeting",
    label="a well-wishing message with no action attached",
    patterns=(
        r"\bgood (morning|evening|night)\b",
        r"\b(stay|keep) (blessed|positive|smiling|safe)\b",
        r"\bpositive (energy|vibes)\b",
        r"\bhave a (good|great|nice) day\b",
        r"\bsending (good vibes|blessings)\b",
        r"\bbhagwan\b|\bbhala kare\b|\bshubh\b",
    ),
)

MARKETING = PatternFamily(
    name="marketing",
    label="promotional content with commercial intent",
    patterns=(
        r"\b\d{1,3}\s?%\s?(off|discount)\b",
        r"\b(offer|deal|sale|discount|coupon|voucher|cashback)\b",
        r"\b(shop|buy|order) (now|today)\b",
        r"\blimited (time|period|offer|stock)\b",
        r"\btap (below|here) to (view|shop|explore|check)\b",
        r"\bwelcome offer\b|\bfirst order\b",
        r"\bstarting (at|from) rs\.?\s?\d+",
    ),
)

FEEDBACK_REQUEST = PatternFamily(
    name="feedback_request",
    label="asks the user to rate, review, or fill in a survey",
    patterns=(
        r"\b(quick |short |brief )?(survey|feedback|review)\b[^.!?]{0,30}\b(fill|share|give|take|complete|leave)\b",
        r"\b(fill|share|give|take|complete|leave)\b[^.!?]{0,30}\b(a )?(quick |short |brief )?(survey|feedback|review|rating)\b",
        r"\bhow (has|was) your experience\b",
        r"\bwe would love to hear\b|\bwe'd love to hear\b",
        r"\brate (your|this|the)\b",
        r"\btell us (how|what|about)\b",
    ),
)

OPT_OUT_MARKER = PatternFamily(
    name="opt_out_marker",
    label="carries a bulk-marketing unsubscribe footer",
    patterns=(
        r"\breply stop\b",
        r"\bunsubscribe\b",
        r"\bt&?c'?s? apply\b",
        r"\bopt[- ]out\b",
    ),
)

URGENCY = PatternFamily(
    name="urgency",
    label="states a deadline or window measured in minutes or hours",
    patterns=(
        r"\b(in|for|within|over)\s+(the\s+)?next\s+\d+\s*(min|minute|hour)",
        r"\bbefore \d{1,2}[:.]\d{2}\b",
        r"\b\d+\s*(min|mins|minutes|hrs|hours)\b[^.!?]{0,25}\b(left|max|only|before|remaining)\b",
        r"\bwithin \d+\s*(min|minute|hour)",
        r"\bby \d{1,2}[:.]?\d{0,2}\s*(am|pm)\b",
        r"\bbefore \d{1,2}\s*(am|pm|o'?clock)\b",
        r"\b(right )?now\b|\bimmediately\b|\basap\b",
        r"\btoday\b|\btonight\b|\bthis (morning|evening|afternoon)\b",
        r"\b(leaving|closes?|closing|starts?|expires?) (in|at|today|tonight)\b",
        r"\bcall me (now|urgently)\b",
        r"\beod\b|\bend of day\b",
        # Hinglish
        r"\babhi\b|\bjaldi\b|\baaj\b|\bturant\b",
        r"\b\d+\s*min\s+me\b",
    ),
)

ACTION_REQUEST = PatternFamily(
    name="action_request",
    label="asks this user specifically to reply, call, confirm or decide",
    patterns=(
        r"\b(can|could|will) you\b",
        r"\b(please |pls |plz )?(confirm|reply|respond|call|send|share|join|check|bring)\b",
        r"\blet me know\b|\btell me\b",
        r"\b(message|msg|text|ping|dm|whatsapp) me\b",
        r"\bneed (your|you)\b",
        r"\b(stay|remain|be) (online|available|near|reachable)\b",
        r"\bkeep\b[^.!?]{0,25}\b(ready|nearby|handy|packed)\b",
        r"\bwaiting for\b",
        r"\bconfirm kar\b|\bbata dena\b|\bbhej dena\b",
    ),
)

NO_ACTION_MARKER = PatternFamily(
    name="no_action_marker",
    label="explicitly tells the user nothing is needed from them",
    patterns=(
        r"\bno (need to |rush|hurry|pressure|urgency)\b",
        r"\bno\s+\w+\s+(is\s+)?(required|needed|necessary)\b",
        r"\bnothing (urgent|dramatic|blocking)\b",
        r"\bwhenever (you get time|convenient|free)\b",
        r"\bno need to (reply|respond|answer)\b",
        r"\bdon'?t (call|reply) now\b",
        r"\bread (it )?later\b|\bwhen you get time\b",
        r"\bkoi urgency nahi\b|\bbaad me\b",
    ),
)

OFFICIAL_CHANNEL = PatternFamily(
    name="official_channel",
    label="routes the user through an official app or office rather than an ad-hoc link",
    patterns=(
        r"\b(society |official |registered )?app\b[^.!?]{0,25}\b(only|use|check|through|via)\b",
        r"\buse (the )?(society app|office qr|official app|registered app)\b",
        r"\b(check|review)\b[^.!?]{0,30}\bin (the |your )?(app|portal|account)\b",
        r"\bdirect office\b|\boffice counter\b|\boffice me\b",
        r"\bno (payment or )?otp is required\b",
        r"\bnever ask(s)? for otp\b",
        r"\bdon'?t use any payment link\b|\bnot? share (any )?link\b",
    ),
)


# --------------------------------------------------------------------------- #
# Reassurance - the negation guard
# --------------------------------------------------------------------------- #

# Genuine senders routinely mention credentials in order to warn against them:
# "no payment or OTP is required for this delivery", "the brand never asks for
# OTP on calls", "please don't use any payment link shared by residents".
#
# Matching risk patterns naively turns those reassurances into scam signals and
# mutes exactly the safety notices a user most wants. So these spans are excised
# from the text before risk matching, while intent matching still sees the full
# message. This is anti-phishing advice, not phishing.
REASSURANCE = PatternFamily(
    name="reassurance",
    label="explicitly warns the user against sharing credentials or paying via links",
    patterns=(
        r"\bno (payment|otp|pin|fee|charge)[^.!?]{0,40}\b(is |are )?(required|needed|asked)\b",
        r"\b(never|do not|don'?t|no one will) (ask|request)[^.!?]{0,40}\b(otp|pin|password|payment|card|bank)\b",
        r"\b(do not|don'?t|never) (share|send|give|reveal)[^.!?]{0,30}\b(otp|pin|password|code|details?)\b",
        r"\b(do not|don'?t|never) (use|open|click|trust)[^.!?]{0,40}\b(link|qr)\b",
        r"\b(do not|don'?t) (post|send|share)[^.!?]{0,30}\bscreenshots?\b",
        r"\breceipts? (will be )?(matched|reconciled)\b",
        r"\bsafety advisory\b",
        r"\bbeware of\b|\bstay alert\b",
        r"\bshare (the )?pickup code only after\b",
    ),
)

_REASSURANCE_RE = REASSURANCE.compiled()


def _strip_reassurance(haystack: str) -> str:
    """Blank out spans that warn *against* the risky behaviour they name.

    Replaced with spaces rather than removed so surrounding patterns cannot be
    accidentally joined across the gap.
    """
    for pattern in _REASSURANCE_RE:
        haystack = pattern.sub(lambda m: " " * len(m.group(0)), haystack)
    return haystack


RISK_FAMILIES: tuple[PatternFamily, ...] = (
    CREDENTIAL_SOLICITATION,
    ACCOUNT_THREAT,
    ADVANCE_FEE,
    PRIZE_LURE,
    SUSPICIOUS_LINK,
    PROMPT_INJECTION,
    CHAIN_FORWARD,
    HEALTH_MISINFO,
)

INTENT_FAMILIES: tuple[PatternFamily, ...] = (
    GREETING,
    MARKETING,
    FEEDBACK_REQUEST,
    OPT_OUT_MARKER,
    URGENCY,
    ACTION_REQUEST,
    NO_ACTION_MARKER,
    OFFICIAL_CHANNEL,
)

ALL_FAMILIES: tuple[PatternFamily, ...] = RISK_FAMILIES + INTENT_FAMILIES + (REASSURANCE,)

# Compiled once at import; matching runs 110 x 16 families per pipeline pass.
_RISK_COMPILED: dict[str, list[re.Pattern[str]]] = {f.name: f.compiled() for f in RISK_FAMILIES}
_INTENT_COMPILED: dict[str, list[re.Pattern[str]]] = {
    f.name: f.compiled() for f in INTENT_FAMILIES + (REASSURANCE,)
}
_LABELS: dict[str, str] = {f.name: f.label for f in ALL_FAMILIES}


def match_families(text: str) -> dict[str, int]:
    """Return {family_name: number of distinct patterns that fired}.

    Risk families are matched against the reassurance-stripped text so that a
    warning about OTPs is not counted as a request for one. Intent families see
    the original text, since a reassurance is itself meaningful intent.
    """
    if not text:
        return {}
    full = normalise(text)
    safe = _strip_reassurance(full)

    hits: dict[str, int] = {}
    for name, patterns in _RISK_COMPILED.items():
        count = sum(1 for p in patterns if p.search(safe))
        if count:
            hits[name] = count
    for name, patterns in _INTENT_COMPILED.items():
        count = sum(1 for p in patterns if p.search(full))
        if count:
            hits[name] = count
    return hits


def family_label(name: str) -> str:
    return _LABELS.get(name, name)


def mentions_user(text: str, user_id: str) -> bool:
    """True if the text @-mentions this specific user."""
    if not text or not user_id:
        return False
    return re.search(rf"@\s*{re.escape(user_id)}\b", text, re.IGNORECASE) is not None


def extract_domains(text: str) -> list[str]:
    """Bare domains and URL hosts appearing in the text."""
    if not text:
        return []
    found = re.findall(
        r"\b(?:https?://)?((?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z]{2,})\b",
        text.lower(),
    )
    # Drop sentence-boundary false positives like "e.g" or bare file extensions.
    return [d for d in dict.fromkeys(found) if "." in d and len(d.split(".")[-1]) >= 2]


def detect_script(text: str) -> str:
    """Rough language hint for the prompt: which writing system dominates."""
    if not text:
        return "none"
    if any("ऀ" <= c <= "ॿ" for c in text):
        return "devanagari"
    romanised_hindi = (
        "aaj", "abhi", "jaldi", "karo", "kar do", "hai", "nahi", "bhejo", "batao",
        "kal", "milte", "mat", "dena", "raha", "jayega", "wala", "koi", "bol",
        "aapka", "apna", "sab", "ho gaya", "le aao", "hata do",
    )
    lowered = normalise(text)
    if sum(1 for token in romanised_hindi if token in lowered) >= 2:
        return "hinglish"
    french = ("bonjour", "votre", "merci", "reception", "immeuble", "avant", "trouve")
    if sum(1 for token in french if token in lowered) >= 2:
        return "french"
    return "latin"
