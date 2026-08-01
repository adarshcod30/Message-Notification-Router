"""Loads the dataset into one indexed, in-memory context graph.

The thirteen CSVs are a small relational database with implicit foreign keys.
Routing quality depends far more on *resolving* those keys correctly than on
prompt wording, so this module does the joins once, up front, and exposes typed
accessors instead of letting every downstream stage re-parse CSV rows.

Everything is read-only after construction.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from functools import cached_property
from pathlib import Path

from .config import PATHS


# --------------------------------------------------------------------------- #
# Parsing helpers - the CSVs use empty strings for nulls throughout
# --------------------------------------------------------------------------- #

def _int(value: str | None, default: int = 0) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _bool(value: str | None) -> bool:
    return str(value).strip() in {"1", "true", "True", "yes"}


def _text(value: str | None) -> str:
    return (value or "").strip()


def _dt(value: str | None) -> datetime | None:
    raw = _text(value)
    if not raw:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"required dataset file missing: {path}\n"
            "Run from the repo root, or set ORCHESTRATE_DATASET_DIR."
        )
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [{k: (v or "") for k, v in row.items() if k is not None} for row in csv.DictReader(handle)]


# --------------------------------------------------------------------------- #
# Entities
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Message:
    """A message, incoming or historical. Same schema in both files."""

    message_id: str
    user_id: str
    conversation_type: str
    group_id: str
    business_id: str
    sender_user_id: str
    created_at: str
    message_text: str
    media_type: str
    media_id: str
    forwarded_count: int

    @property
    def timestamp(self) -> datetime | None:
        return _dt(self.created_at)

    @property
    def has_media(self) -> bool:
        return bool(self.media_type and self.media_id)

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "Message":
        return cls(
            message_id=_text(row.get("message_id")),
            user_id=_text(row.get("user_id")),
            conversation_type=_text(row.get("conversation_type")),
            group_id=_text(row.get("group_id")),
            business_id=_text(row.get("business_id")),
            sender_user_id=_text(row.get("sender_user_id")),
            created_at=_text(row.get("created_at")),
            message_text=(row.get("message_text") or "").strip(),
            media_type=_text(row.get("media_type")),
            media_id=_text(row.get("media_id")),
            forwarded_count=_int(row.get("forwarded_count")),
        )


@dataclass(frozen=True)
class User:
    user_id: str
    do_not_disturb_window: str
    messages_opened_30d: int
    messages_replied_30d: int
    notifications_dismissed_30d: int
    messages_reported_30d: int

    @property
    def dnd_range(self) -> tuple[int, int] | None:
        """DND window as (start_minute, end_minute) since midnight, or None."""
        raw = self.do_not_disturb_window
        if "-" not in raw:
            return None
        start, _, end = raw.partition("-")
        try:
            sh, sm = (int(p) for p in start.strip().split(":"))
            eh, em = (int(p) for p in end.strip().split(":"))
        except ValueError:
            return None
        return sh * 60 + sm, eh * 60 + em

    def in_quiet_hours(self, when: datetime | None) -> bool:
        """True if ``when`` falls inside the DND window (which wraps midnight)."""
        rng = self.dnd_range
        if rng is None or when is None:
            return False
        start, end = rng
        minute = when.hour * 60 + when.minute
        if start > end:  # window wraps midnight, e.g. 22:00-07:00
            return minute >= start or minute < end
        return start <= minute < end

    @property
    def dismissal_rate(self) -> float:
        """How reflexively this user dismisses notifications overall."""
        total = self.messages_opened_30d + self.notifications_dismissed_30d
        return self.notifications_dismissed_30d / total if total else 0.0


@dataclass(frozen=True)
class Group:
    group_id: str
    group_name: str
    group_type: str
    member_count: int
    admin_count: int
    created_at: str
    messages_30d: int

    @property
    def is_high_volume(self) -> bool:
        return self.messages_30d >= 400 or self.member_count >= 150


@dataclass(frozen=True)
class GroupMembership:
    group_id: str
    user_id: str
    role: str
    joined_at: str
    messages_sent_30d: int
    messages_read_30d: int
    replies_sent_30d: int
    notifications_dismissed_30d: int
    group_muted_by_user: bool

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def engagement(self) -> float:
        """How much this user actually engages with this group, in [0, 1]."""
        signal = self.messages_read_30d + 2 * self.replies_sent_30d
        noise = self.notifications_dismissed_30d
        return signal / (signal + noise) if signal + noise else 0.0


@dataclass(frozen=True)
class BusinessAccount:
    business_id: str
    display_name: str
    brand_name: str
    category: str
    verified: bool
    official_domain: str
    domain_used_by_sender: str
    account_age_days: int
    messages_sent_30d: int
    user_reports_30d: int
    domain_used_by_sender_age_days: int

    @property
    def domain_mismatch(self) -> bool:
        """Sender is using a domain that is not the brand's official one.

        On its own this is weak: legitimate verified brands route marketing
        through link shorteners and tracking domains. It only becomes damning
        alongside an unverified account, a young domain, or heavy reporting.
        """
        return bool(
            self.official_domain
            and self.domain_used_by_sender
            and self.official_domain != self.domain_used_by_sender
        )

    @property
    def report_rate_per_1k(self) -> float:
        return 1000 * self.user_reports_30d / self.messages_sent_30d if self.messages_sent_30d else 0.0

    @property
    def impersonation_score(self) -> float:
        """0..1 likelihood this account is impersonating the brand it claims.

        The dataset's fake accounts share a consistent fingerprint: they borrow a
        real ``brand_name``, are unverified, were registered three to five weeks
        ago on a lookalike domain that is younger still, and are reported heavily
        relative to volume. Genuine brands fail at most one of these tests.
        """
        score = 0.0
        if not self.verified:
            score += 0.30
        if self.domain_mismatch:
            score += 0.25
        if self.account_age_days and self.account_age_days < 60:
            score += 0.20
        if self.domain_used_by_sender_age_days and self.domain_used_by_sender_age_days < 45:
            score += 0.15
        if self.report_rate_per_1k >= 25:
            score += 0.20
        elif self.report_rate_per_1k >= 8:
            score += 0.10
        # Borrowing a known brand name while unverified is the sharpest single tell.
        if not self.verified and self.brand_name and self.brand_name.lower() != "unknown":
            if self.brand_name.lower() not in self.display_name.lower() or self.domain_mismatch:
                score += 0.10
        return min(1.0, score)


@dataclass(frozen=True)
class UserBusinessLink:
    user_id: str
    business_id: str
    why_user_knows_account: str
    last_activity_at: str
    allows_promotions: bool
    promotions_opted_out_at: str
    activity_count_180d: int
    messages_opened_30d: int
    messages_dismissed_30d: int
    messages_replied_30d: int
    last_reply_at: str

    @property
    def opted_out(self) -> bool:
        return bool(self.promotions_opted_out_at) or not self.allows_promotions

    @property
    def is_transactional(self) -> bool:
        """User has a real order/booking/payment relationship, not just a signup.

        Matched on stems rather than whole words: the dataset writes these as
        ``ride_booked_today``, ``prescription_refill`` and ``recent_return_pickup``,
        none of which contain the noun forms a naive check looks for.
        """
        reason = self.why_user_knows_account
        return any(k in reason for k in (
            "order", "deliver", "book", "pay", "purchas", "appointment",
            "refill", "pickup", "prescription", "reserv", "ride", "subscription_active",
        ))

    @property
    def dismissal_ratio(self) -> float:
        total = self.messages_opened_30d + self.messages_dismissed_30d
        return self.messages_dismissed_30d / total if total else 0.0

    @property
    def is_fatigued(self) -> bool:
        """User is visibly tired of this sender's messages."""
        return self.messages_dismissed_30d >= 3 and self.dismissal_ratio >= 0.6


@dataclass(frozen=True)
class MessageEvent:
    """How one user reacted to one historical message."""

    user_id: str
    message_id: str
    message_opened: bool
    message_replied: bool
    reaction_time_minutes: int | None
    notification_dismissed: bool
    muted_after_message: bool
    message_reported: bool

    @property
    def was_welcome(self) -> bool:
        return self.message_opened and not self.notification_dismissed

    @property
    def was_rejected(self) -> bool:
        return self.notification_dismissed or self.muted_after_message or self.message_reported

    def describe(self) -> str:
        """Compact human-readable summary, used inside the judge prompt."""
        if self.message_reported:
            return "reported as unsafe"
        if self.muted_after_message and self.notification_dismissed:
            return "dismissed, then muted the sender"
        if self.muted_after_message:
            return "muted the sender afterwards"
        if self.notification_dismissed:
            return "dismissed without opening"
        if self.message_replied:
            rt = f" in {self.reaction_time_minutes} min" if self.reaction_time_minutes is not None else ""
            return f"opened and replied{rt}"
        if self.message_opened:
            return "opened but did not reply"
        return "ignored"


# --------------------------------------------------------------------------- #
# The store
# --------------------------------------------------------------------------- #

@dataclass
class ContextStore:
    """Indexed access to every dataset relation."""

    users: dict[str, User] = field(default_factory=dict)
    groups: dict[str, Group] = field(default_factory=dict)
    memberships: dict[tuple[str, str], GroupMembership] = field(default_factory=dict)
    businesses: dict[str, BusinessAccount] = field(default_factory=dict)
    user_business: dict[tuple[str, str], UserBusinessLink] = field(default_factory=dict)
    history: dict[str, Message] = field(default_factory=dict)
    events: dict[tuple[str, str], MessageEvent] = field(default_factory=dict)
    images: dict[str, str] = field(default_factory=dict)
    voice_notes: dict[str, str] = field(default_factory=dict)
    daily_load: dict[str, list[tuple[str, int, int]]] = field(default_factory=dict)
    incoming: list[Message] = field(default_factory=list)
    samples: list[dict[str, str]] = field(default_factory=list)
    # Directory the store was loaded from; media paths resolve against it so an
    # ORCHESTRATE_DATASET_DIR override or a test fixture stays self-consistent.
    dataset_dir: Path = field(default_factory=lambda: PATHS.dataset)

    # --- derived indices, built in _index() ---
    history_by_user: dict[str, list[Message]] = field(default_factory=lambda: defaultdict(list))
    history_by_user_sender: dict[tuple[str, str], list[Message]] = field(default_factory=lambda: defaultdict(list))
    history_by_user_business: dict[tuple[str, str], list[Message]] = field(default_factory=lambda: defaultdict(list))
    history_by_user_group: dict[tuple[str, str], list[Message]] = field(default_factory=lambda: defaultdict(list))
    members_by_group: dict[str, list[GroupMembership]] = field(default_factory=lambda: defaultdict(list))

    # ---------------- construction ----------------

    @classmethod
    def load(cls, dataset_dir: Path | None = None) -> "ContextStore":
        base = Path(dataset_dir) if dataset_dir else PATHS.dataset
        store = cls(dataset_dir=base)

        for row in _read(base / "users.csv"):
            store.users[_text(row["user_id"])] = User(
                user_id=_text(row["user_id"]),
                do_not_disturb_window=_text(row["do_not_disturb_window"]),
                messages_opened_30d=_int(row["messages_opened_30d"]),
                messages_replied_30d=_int(row["messages_replied_30d"]),
                notifications_dismissed_30d=_int(row["notifications_dismissed_30d"]),
                messages_reported_30d=_int(row["messages_reported_30d"]),
            )

        for row in _read(base / "groups.csv"):
            store.groups[_text(row["group_id"])] = Group(
                group_id=_text(row["group_id"]),
                group_name=_text(row["group_name"]),
                group_type=_text(row["group_type"]),
                member_count=_int(row["member_count"]),
                admin_count=_int(row["admin_count"]),
                created_at=_text(row["created_at"]),
                messages_30d=_int(row["messages_30d"]),
            )

        for row in _read(base / "group_members.csv"):
            membership = GroupMembership(
                group_id=_text(row["group_id"]),
                user_id=_text(row["user_id"]),
                role=_text(row["role"]),
                joined_at=_text(row["joined_at"]),
                messages_sent_30d=_int(row["messages_sent_30d"]),
                messages_read_30d=_int(row["messages_read_30d"]),
                replies_sent_30d=_int(row["replies_sent_30d"]),
                notifications_dismissed_30d=_int(row["notifications_dismissed_30d"]),
                group_muted_by_user=_bool(row["group_muted_by_user"]),
            )
            store.memberships[(membership.group_id, membership.user_id)] = membership

        for row in _read(base / "business_accounts.csv"):
            store.businesses[_text(row["business_id"])] = BusinessAccount(
                business_id=_text(row["business_id"]),
                display_name=_text(row["display_name"]),
                brand_name=_text(row["brand_name"]),
                category=_text(row["category"]),
                verified=_bool(row["verified"]),
                official_domain=_text(row["official_domain"]),
                domain_used_by_sender=_text(row["domain_used_by_sender"]),
                account_age_days=_int(row["account_age_days"]),
                messages_sent_30d=_int(row["messages_sent_30d"]),
                user_reports_30d=_int(row["user_reports_30d"]),
                domain_used_by_sender_age_days=_int(row["domain_used_by_sender_age_days"]),
            )

        for row in _read(base / "user_business_history.csv"):
            link = UserBusinessLink(
                user_id=_text(row["user_id"]),
                business_id=_text(row["business_id"]),
                why_user_knows_account=_text(row["why_user_knows_account"]),
                last_activity_at=_text(row["last_activity_at"]),
                allows_promotions=_bool(row["allows_promotions"]),
                promotions_opted_out_at=_text(row["promotions_opted_out_at"]),
                activity_count_180d=_int(row["activity_count_180d"]),
                messages_opened_30d=_int(row["messages_opened_30d"]),
                messages_dismissed_30d=_int(row["messages_dismissed_30d"]),
                messages_replied_30d=_int(row["messages_replied_30d"]),
                last_reply_at=_text(row["last_reply_at"]),
            )
            store.user_business[(link.user_id, link.business_id)] = link

        for row in _read(base / "message_history.csv"):
            msg = Message.from_row(row)
            store.history[msg.message_id] = msg

        for row in _read(base / "message_events.csv"):
            rt = _text(row["reaction_time_minutes"])
            event = MessageEvent(
                user_id=_text(row["user_id"]),
                message_id=_text(row["message_id"]),
                message_opened=_bool(row["message_opened"]),
                message_replied=_bool(row["message_replied"]),
                reaction_time_minutes=_int(rt) if rt else None,
                notification_dismissed=_bool(row["notification_dismissed"]),
                muted_after_message=_bool(row["muted_after_message"]),
                message_reported=_bool(row["message_reported"]),
            )
            store.events[(event.user_id, event.message_id)] = event

        for row in _read(base / "images.csv"):
            store.images[_text(row["image_id"])] = _text(row["file_path"])
        for row in _read(base / "voice_notes.csv"):
            store.voice_notes[_text(row["voice_note_id"])] = _text(row["file_path"])

        for row in _read(base / "daily_notification_summary.csv"):
            store.daily_load.setdefault(_text(row["user_id"]), []).append(
                (_text(row["date"]), _int(row["notifications_sent"]), _int(row["notifications_dismissed"]))
            )

        store.incoming = [Message.from_row(r) for r in _read(base / "messages.csv")]

        sample_path = base / "sample_messages.csv"
        if sample_path.is_file():
            store.samples = _read(sample_path)

        store._index()
        return store

    def _index(self) -> None:
        for msg in self.history.values():
            self.history_by_user[msg.user_id].append(msg)
            if msg.sender_user_id:
                self.history_by_user_sender[(msg.user_id, msg.sender_user_id)].append(msg)
            if msg.business_id:
                self.history_by_user_business[(msg.user_id, msg.business_id)].append(msg)
            if msg.group_id:
                self.history_by_user_group[(msg.user_id, msg.group_id)].append(msg)
        for membership in self.memberships.values():
            self.members_by_group[membership.group_id].append(membership)

        # Newest first everywhere, so "most recent precedent" is just [0].
        def key(m: Message) -> str:
            return m.created_at
        for bucket in (
            self.history_by_user,
            self.history_by_user_sender,
            self.history_by_user_business,
            self.history_by_user_group,
        ):
            for messages in bucket.values():
                messages.sort(key=key, reverse=True)

    # ---------------- accessors ----------------

    def user(self, user_id: str) -> User | None:
        return self.users.get(user_id)

    def group(self, group_id: str) -> Group | None:
        return self.groups.get(group_id) if group_id else None

    def membership(self, group_id: str, user_id: str) -> GroupMembership | None:
        return self.memberships.get((group_id, user_id)) if group_id and user_id else None

    def business(self, business_id: str) -> BusinessAccount | None:
        return self.businesses.get(business_id) if business_id else None

    def business_link(self, user_id: str, business_id: str) -> UserBusinessLink | None:
        return self.user_business.get((user_id, business_id)) if business_id else None

    def event(self, user_id: str, message_id: str) -> MessageEvent | None:
        return self.events.get((user_id, message_id))

    def media_path(self, media_type: str, media_id: str) -> Path | None:
        """Absolute path to a media file, or None if unknown/missing on disk."""
        table = self.images if media_type == "image" else self.voice_notes if media_type == "voice" else None
        if table is None:
            return None
        rel = table.get(media_id)
        if not rel:
            return None
        path = self.dataset_dir / rel
        return path if path.is_file() else None

    def sender_history(self, user_id: str, sender_user_id: str) -> list[Message]:
        return self.history_by_user_sender.get((user_id, sender_user_id), [])

    def business_history(self, user_id: str, business_id: str) -> list[Message]:
        return self.history_by_user_business.get((user_id, business_id), [])

    def user_history(self, user_id: str) -> list[Message]:
        return self.history_by_user.get(user_id, [])

    def notification_load(self, user_id: str) -> tuple[float, float]:
        """(mean notifications/day, mean dismissals/day) over the summary window."""
        rows = self.daily_load.get(user_id) or []
        if not rows:
            return 0.0, 0.0
        return (
            sum(r[1] for r in rows) / len(rows),
            sum(r[2] for r in rows) / len(rows),
        )

    def reaction_profile(self, user_id: str, messages: list[Message]) -> dict[str, int]:
        """Aggregate how this user reacted across a set of historical messages."""
        profile = {"total": 0, "opened": 0, "replied": 0, "dismissed": 0, "muted": 0, "reported": 0}
        for msg in messages:
            event = self.event(user_id, msg.message_id)
            if event is None:
                continue
            profile["total"] += 1
            profile["opened"] += event.message_opened
            profile["replied"] += event.message_replied
            profile["dismissed"] += event.notification_dismissed
            profile["muted"] += event.muted_after_message
            profile["reported"] += event.message_reported
        return profile

    @cached_property
    def stats(self) -> dict[str, int]:
        return {
            "users": len(self.users),
            "groups": len(self.groups),
            "memberships": len(self.memberships),
            "businesses": len(self.businesses),
            "user_business_links": len(self.user_business),
            "history_messages": len(self.history),
            "message_events": len(self.events),
            "images": len(self.images),
            "voice_notes": len(self.voice_notes),
            "incoming_messages": len(self.incoming),
            "labelled_samples": len(self.samples),
        }
