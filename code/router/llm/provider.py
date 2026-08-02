"""LLM access layer: caching, adaptive rate limiting, backoff, model fallback.

The whole run happens against a free-tier key with per-minute *and* per-day
request ceilings that are not published per model and change over time. Three
design choices follow from that:

1. **Cache first.** Every response is persisted to disk keyed by a hash of the
   exact request. Re-running the pipeline after a prompt tweak only re-queries
   the calls that actually changed, and a completed run costs zero quota to
   reproduce. This is also what makes the submission deterministic.

2. **Adapt, do not assume.** The limiter starts at a conservative guess and
   *lowers its own ceiling* whenever the API returns 429, rather than trusting a
   hardcoded RPM that will be wrong.

3. **Degrade, never die.** On persistent failure the client walks a model
   fallback chain, and only then gives up - returning ``None`` so the caller can
   fall back to the deterministic router instead of crashing the run.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

from ..config import MODELS, PATHS

log = logging.getLogger(__name__)

# Response bodies that indicate a transient condition worth retrying.
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass
class LLMResponse:
    """One model reply plus the metadata needed for auditing and costing."""

    text: str
    model: str
    provider: str
    cached: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    attempts: int = 1

    def json(self) -> dict | None:
        """Parse the reply as JSON, tolerating markdown fences and stray prose."""
        raw = self.text.strip()
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1] if raw.count("```") >= 2 else raw.strip("`")
            if raw.lstrip().startswith("json"):
                raw = raw.lstrip()[4:]
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        # Fall back to the outermost brace-balanced object, which recovers
        # replies wrapped in commentary despite the JSON mime type.
        start = raw.find("{")
        if start == -1:
            return None
        depth = 0
        in_string = escape = False
        for i, ch in enumerate(raw[start:], start):
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue
            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(raw[start : i + 1])
                    except json.JSONDecodeError:
                        return None
        return None


@dataclass
class UsageStats:
    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    rate_limit_hits: int = 0
    failures: int = 0
    prompt_tokens: int = 0
    output_tokens: int = 0
    thinking_tokens: int = 0
    by_model: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "live_calls": self.calls - self.cache_hits,
            "retries": self.retries,
            "rate_limit_hits": self.rate_limit_hits,
            "failures": self.failures,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
            "by_model": dict(self.by_model),
        }


class AdaptiveRateLimiter:
    """Thread-safe minimum-interval limiter that tightens itself on 429.

    A plain token bucket needs a correct RPM up front. We do not have one, so
    this starts optimistic and multiplies its spacing by 1.5 on every rate-limit
    response (capped at 20x). It never speeds back up within a run - under a
    daily quota, creeping back up just burns the remaining budget on retries.
    """

    def __init__(self, requests_per_minute: int) -> None:
        self._base_interval = 60.0 / max(1, requests_per_minute)
        self._interval = self._base_interval
        self._next_free = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next_free - now)
            self._next_free = max(now, self._next_free) + self._interval
        if wait > 0:
            time.sleep(wait)

    def penalise(self) -> None:
        with self._lock:
            self._interval = min(self._interval * 1.5, self._base_interval * 20)
            log.debug("rate limiter backed off to %.2fs between calls", self._interval)

    @property
    def interval(self) -> float:
        return self._interval


class ResponseCache:
    """Content-addressed disk cache. One JSON file per distinct request."""

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = root
        self.enabled = enabled
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(payload: dict) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]

    def _path(self, key: str) -> Path:
        # Shard by first two hex chars to keep directory listings small.
        return self.root / key[:2] / f"{key}.json"

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        path = self._path(key)
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def put(self, key: str, value: dict) -> None:
        if not self.enabled:
            return
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)  # atomic, so a killed run never leaves a torn cache entry


class GeminiClient:
    """Gemini REST client with caching, retry, fallback, and usage accounting."""

    provider = "gemini"

    def __init__(
        self,
        models: tuple[str, ...] | None = None,
        cache_enabled: bool = True,
        cache_namespace: str = "gemini",
        max_attempts: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.models = tuple(m.strip() for m in (models or MODELS.judge_models) if m.strip())
        if not self.models:
            raise ValueError("at least one model must be configured")
        self.max_attempts = max_attempts or MODELS.max_attempts
        self.timeout_seconds = timeout_seconds or MODELS.timeout_seconds
        self.cache = ResponseCache(PATHS.cache / cache_namespace, cache_enabled)
        self.limiter = AdaptiveRateLimiter(MODELS.requests_per_minute)
        self.stats = UsageStats()
        # Circuit breaker. A daily-quota exhaustion is not a transient spike: every
        # subsequent call will fail the same way. Without this the run re-discovers
        # the outage once per message, turning a graceful degradation into a crawl.
        self._consecutive_failures = 0
        self._circuit_open = False
        self._session = requests.Session()
        self._lock = threading.Lock()

    # ---------------- public API ----------------

    @property
    def available(self) -> bool:
        return bool(MODELS.api_key) and not self._circuit_open

    @property
    def circuit_open(self) -> bool:
        """True once the client has given up on the API for this run."""
        return self._circuit_open

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        media: list[Path] | None = None,
        response_schema: dict | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,
        cache_salt: str = "",
    ) -> LLMResponse | None:
        """Generate one completion, or None if every model and retry failed.

        ``cache_salt`` distinguishes otherwise-identical requests that must not
        share a cache entry - self-consistency samples, for example.
        """
        parts: list[dict] = [{"text": prompt}]
        for path in media or []:
            part = _inline_media(path)
            if part:
                parts.append(part)

        body: dict = {
            "contents": [{"role": "user", "parts": parts}],
            "generationConfig": {
                "temperature": MODELS.temperature if temperature is None else temperature,
                "maxOutputTokens": max_output_tokens or MODELS.max_output_tokens,
            },
        }
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        if response_schema is not None:
            body["generationConfig"]["responseMimeType"] = "application/json"
            body["generationConfig"]["responseSchema"] = response_schema

        budget = MODELS.thinking_budget if thinking_budget is None else thinking_budget
        if budget >= 0:
            body["generationConfig"]["thinkingConfig"] = {"thinkingBudget": budget}

        cache_key = ResponseCache.key({"body": body, "salt": cache_salt, "models": self.models})
        # Cache is still served with the breaker open: replaying a stored answer
        # costs no quota and is exactly what we want during an outage.
        if (hit := self.cache.get(cache_key)) is not None:
            with self._lock:
                self.stats.calls += 1
                self.stats.cache_hits += 1
            return LLMResponse(
                text=hit["text"], model=hit.get("model", "cached"),
                provider=self.provider, cached=True,
            )

        if self._circuit_open:
            with self._lock:
                self.stats.failures += 1
            return None

        for model in self.models:
            # A daily-quota wall applies to the whole project, so walking the rest
            # of the fallback chain after hitting it just burns wall-clock.
            if self._circuit_open:
                break
            response = self._call_with_retry(model, body)
            if response is not None:
                self.cache.put(cache_key, {"text": response.text, "model": model})
                with self._lock:
                    self._consecutive_failures = 0
                return response
            log.warning("model %s exhausted its retries; falling back", model)

        with self._lock:
            self.stats.failures += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= MODELS.circuit_breaker_threshold and not self._circuit_open:
                self._circuit_open = True
                log.error(
                    "every model failed %d times in a row - opening the circuit and "
                    "serving the rest of this run from cache and the deterministic router",
                    self._consecutive_failures,
                )
        return None

    # ---------------- internals ----------------

    def _call_with_retry(self, model: str, body: dict) -> LLMResponse | None:
        url = f"{MODELS.api_base}/models/{model}:generateContent"
        headers = {"content-type": "application/json", "x-goog-api-key": MODELS.api_key}

        for attempt in range(1, self.max_attempts + 1):
            if self._circuit_open:
                return None
            self.limiter.acquire()
            try:
                raw = self._session.post(
                    url, headers=headers, json=body, timeout=self.timeout_seconds
                )
            except requests.RequestException as exc:
                log.warning("%s attempt %d transport error: %s", model, attempt, exc)
                self._sleep_backoff(attempt)
                continue

            if raw.status_code == 200:
                parsed = self._parse(raw.json(), model, attempt)
                if parsed is not None:
                    return parsed
                # A 200 with no usable text (usually MAX_TOKENS spent entirely on
                # thinking) is retryable, but only with more room to answer.
                log.warning("%s attempt %d returned no text; raising output budget", model, attempt)
                body["generationConfig"]["maxOutputTokens"] = int(
                    body["generationConfig"]["maxOutputTokens"] * 1.5
                )
                self._sleep_backoff(attempt)
                continue

            if raw.status_code in _RETRY_STATUS:
                if raw.status_code == 429:
                    self.limiter.penalise()
                    with self._lock:
                        self.stats.rate_limit_hits += 1
                    if _is_daily_quota_exhausted(raw):
                        log.error("daily free-tier quota exhausted; no retry can help")
                        with self._lock:
                            self._circuit_open = True
                        return None
                self._sleep_backoff(attempt, retry_after=_retry_after(raw))
                continue

            log.error("%s permanent HTTP %d: %s", model, raw.status_code, raw.text[:250])
            return None

        return None

    def _parse(self, payload: dict, model: str, attempt: int) -> LLMResponse | None:
        candidates = payload.get("candidates") or []
        if not candidates:
            return None
        content = candidates[0].get("content") or {}
        text = "".join(p.get("text", "") for p in content.get("parts") or [])
        if not text.strip():
            return None

        usage = payload.get("usageMetadata") or {}
        with self._lock:
            self.stats.calls += 1
            self.stats.retries += attempt - 1
            self.stats.prompt_tokens += usage.get("promptTokenCount", 0)
            self.stats.output_tokens += usage.get("candidatesTokenCount", 0)
            self.stats.thinking_tokens += usage.get("thoughtsTokenCount", 0)
            self.stats.by_model[model] = self.stats.by_model.get(model, 0) + 1

        return LLMResponse(
            text=text,
            model=model,
            provider=self.provider,
            prompt_tokens=usage.get("promptTokenCount", 0),
            output_tokens=usage.get("candidatesTokenCount", 0),
            thinking_tokens=usage.get("thoughtsTokenCount", 0),
            attempts=attempt,
        )

    @staticmethod
    def _sleep_backoff(attempt: int, retry_after: float | None = None) -> None:
        if retry_after is not None:
            delay = min(retry_after, MODELS.backoff_max_seconds)
        else:
            delay = min(
                MODELS.backoff_base_seconds ** attempt, MODELS.backoff_max_seconds
            )
        # Jitter prevents parallel workers from retrying in lockstep.
        time.sleep(delay * (0.7 + 0.6 * random.random()))


def _is_daily_quota_exhausted(response: requests.Response) -> bool:
    """True when the 429 is a per-day cap rather than a per-minute burst limit."""
    try:
        payload = response.json()
    except ValueError:
        return False
    body = json.dumps(payload)
    return "PerDay" in body or "generate_content_free_tier_requests" in body


def _retry_after(response: requests.Response) -> float | None:
    """Honour the server's own backoff hint when it supplies one."""
    header = response.headers.get("Retry-After")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    try:
        details = response.json().get("error", {}).get("details", [])
        for entry in details:
            delay = entry.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return float(delay[:-1])
    except (ValueError, AttributeError, TypeError):
        pass
    return None


# Magic-byte signatures, checked before the filename is trusted.
#
# The dataset's extensions are not reliable: one ".jpg" is actually a PNG and one
# ".mp3" is actually M4A. Declaring a mimeType that contradicts the bytes makes the
# API hang until the request times out rather than failing fast, which looked like
# a rate-limit problem for a long time. Sniffing the content fixes it outright.
_MAGIC: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"GIF8", 0, "image/gif"),
    (b"BM", 0, "image/bmp"),
    (b"WEBP", 8, "image/webp"),      # RIFF....WEBP
    (b"WAVE", 8, "audio/wav"),       # RIFF....WAVE
    (b"ID3", 0, "audio/mpeg"),
    (b"OggS", 0, "audio/ogg"),
    (b"fLaC", 0, "audio/flac"),
    (b"ftyp", 4, "audio/mp4"),       # M4A / MP4 container
)


def sniff_mime(data: bytes, filename: str = "") -> str:
    """Identify a media type from its bytes, falling back to the extension."""
    for signature, offset, mime in _MAGIC:
        if data[offset : offset + len(signature)] == signature:
            return mime
    # Frame-synced MP3 with no ID3 header.
    if len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0:
        return "audio/mpeg"
    guessed = mimetypes.guess_type(filename)[0] if filename else None
    return guessed or "application/octet-stream"


def _inline_media(path: Path) -> dict | None:
    """Base64-inline a media file. Dataset media is small enough to skip uploads."""
    try:
        data = path.read_bytes()
    except OSError as exc:
        log.error("cannot read media %s: %s", path, exc)
        return None
    mime = sniff_mime(data, path.name)
    declared = mimetypes.guess_type(path.name)[0]
    if declared and declared != mime:
        log.info("%s is really %s, not %s (extension is wrong)", path.name, mime, declared)
    return {"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode("ascii")}}


class FallbackClient:
    """Try each client in order; the first usable answer wins.

    Lets the judge treat "Anthropic, then Gemini, then nothing" as one object. A
    client is skipped when it is unavailable, when its circuit has opened, or when
    it cannot handle the request's media - Anthropic reads images but not audio,
    so voice notes fall through to Gemini automatically rather than being dropped.
    """

    provider = "fallback"

    def __init__(self, *clients) -> None:
        self.clients = [c for c in clients if c is not None]
        self.served: dict[str, int] = {}

    @property
    def available(self) -> bool:
        return any(c.available for c in self.clients)

    @property
    def circuit_open(self) -> bool:
        return all(getattr(c, "circuit_open", False) for c in self.clients) if self.clients else True

    @property
    def stats(self) -> UsageStats:
        """Merged usage across every client, so callers see one accounting."""
        total = UsageStats()
        for client in self.clients:
            s = client.stats
            total.calls += s.calls
            total.cache_hits += s.cache_hits
            total.retries += s.retries
            total.rate_limit_hits += s.rate_limit_hits
            total.failures += s.failures
            total.prompt_tokens += s.prompt_tokens
            total.output_tokens += s.output_tokens
            total.thinking_tokens += s.thinking_tokens
            for model, n in s.by_model.items():
                total.by_model[model] = total.by_model.get(model, 0) + n
        return total

    def generate(self, prompt: str, **kwargs) -> LLMResponse | None:
        for client in self.clients:
            if not client.available:
                continue
            supports = getattr(client, "supports", None)
            if supports is not None and not supports(kwargs.get("media")):
                continue
            response = client.generate(prompt, **kwargs)
            if response is not None:
                self.served[client.provider] = self.served.get(client.provider, 0) + 1
                return response
            log.warning("%s could not answer; trying the next provider", client.provider)
        return None
