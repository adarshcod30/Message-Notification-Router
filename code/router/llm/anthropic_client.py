"""Anthropic client: same contract as the Gemini one, plus caching and a spend guard.

Implements the identical surface (`available`, `generate`, `stats`, `circuit_open`)
so :class:`router.llm.provider.FallbackClient` can chain the two without either
knowing about the other.

Three things are specific to this provider:

* **Prompt caching.** The system prompt carries the whole rationale taxonomy and is
  byte-identical on every call — 2,412 tokens re-sent 110 times. Marking it
  ``cache_control: ephemeral`` costs one write and 109 cheap reads, which measured
  at roughly half the price of sending it each time.
* **Budget enforcement.** Every request reserves its worst-case cost from a
  :class:`~router.llm.budget.BudgetGuard` *before* going out, and settles on the
  real token counts afterwards. A run cannot overspend its ceiling even if it
  retries.
* **Structured output via a forced tool call.** Anthropic has no
  ``response_schema`` parameter, so the JSON schema is exposed as a single tool
  and ``tool_choice`` forces it. That yields schema-valid JSON rather than prose
  that merely looks like JSON.
"""

from __future__ import annotations

import base64
import json
import logging
import threading
from pathlib import Path

import requests

from ..config import MODELS, PATHS
from .budget import BudgetExceeded, BudgetGuard
from .provider import AdaptiveRateLimiter, LLMResponse, ResponseCache, UsageStats, sniff_mime

log = logging.getLogger(__name__)

_API_URL = "https://api.anthropic.com/v1/messages"
_VERSION = "2023-06-01"
_RETRY_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}

# Anthropic accepts images inline but not audio; audio-bearing calls are routed to
# Gemini by the fallback chain rather than silently dropping the attachment.
_SUPPORTED_MEDIA_PREFIX = "image/"


class AnthropicClient:
    """Claude REST client with caching, retry, budget enforcement, and a breaker."""

    provider = "anthropic"

    def __init__(
        self,
        models: tuple[str, ...] | None = None,
        *,
        budget: BudgetGuard,
        cache_enabled: bool = True,
        cache_namespace: str = "anthropic",
        max_attempts: int | None = None,
        timeout_seconds: float | None = None,
    ) -> None:
        self.models = tuple(m.strip() for m in (models or MODELS.anthropic_models) if m.strip())
        self.budget = budget
        self.cache = ResponseCache(PATHS.cache / cache_namespace, cache_enabled)
        self.limiter = AdaptiveRateLimiter(MODELS.anthropic_rpm)
        self.stats = UsageStats()
        self.max_attempts = max_attempts or MODELS.max_attempts
        self.timeout_seconds = timeout_seconds or MODELS.timeout_seconds
        self._session = requests.Session()
        self._lock = threading.Lock()
        self._circuit_open = False
        self._consecutive_failures = 0

    # ---------------- public API ----------------

    @property
    def available(self) -> bool:
        return bool(MODELS.anthropic_api_key) and bool(self.models) and not self._circuit_open

    @property
    def circuit_open(self) -> bool:
        return self._circuit_open

    def supports(self, media: list[Path] | None) -> bool:
        """False when the request carries media this provider cannot read."""
        for path in media or []:
            try:
                mime = sniff_mime(path.read_bytes()[:32], path.name)
            except OSError:
                return False
            if not mime.startswith(_SUPPORTED_MEDIA_PREFIX):
                return False
        return True

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        media: list[Path] | None = None,
        response_schema: dict | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        thinking_budget: int | None = None,  # noqa: ARG002 - parity with GeminiClient
        cache_salt: str = "",
    ) -> LLMResponse | None:
        if not self.supports(media):
            log.debug("anthropic cannot read this media; deferring to the next provider")
            return None

        content: list[dict] = [{"type": "text", "text": prompt}]
        for path in media or []:
            block = _inline_image(path)
            if block:
                content.append(block)

        max_tokens = max_output_tokens or MODELS.max_output_tokens
        body: dict = {
            "max_tokens": max_tokens,
            "temperature": MODELS.temperature if temperature is None else temperature,
            "messages": [{"role": "user", "content": content}],
        }
        if system:
            # Cache the system block: identical across every message in a run.
            body["system"] = [
                {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
            ]
        if response_schema is not None:
            body["tools"] = [{
                "name": "emit_routing_decision",
                "description": "Return the routing decision for this message.",
                "input_schema": response_schema,
            }]
            body["tool_choice"] = {"type": "tool", "name": "emit_routing_decision"}

        cache_key = ResponseCache.key({"body": body, "salt": cache_salt, "models": self.models})
        if (hit := self.cache.get(cache_key)) is not None:
            with self._lock:
                self.stats.calls += 1
                self.stats.cache_hits += 1
            return LLMResponse(
                text=hit["text"], model=hit.get("model", "cached"),
                provider=self.provider, cached=True,
            )

        if self._circuit_open:
            return None

        for model in self.models:
            if self._circuit_open:
                break
            response = self._call_with_retry(model, body, max_tokens)
            if response is not None:
                self.cache.put(cache_key, {"text": response.text, "model": model})
                with self._lock:
                    self._consecutive_failures = 0
                return response
            log.warning("anthropic model %s exhausted its retries; falling back", model)

        with self._lock:
            self.stats.failures += 1
            self._consecutive_failures += 1
            if self._consecutive_failures >= MODELS.circuit_breaker_threshold:
                self._circuit_open = True
                log.error("anthropic circuit opened after %d consecutive failures",
                          self._consecutive_failures)
        return None

    # ---------------- internals ----------------

    def _call_with_retry(self, model: str, body: dict, max_tokens: int) -> LLMResponse | None:
        headers = {
            "x-api-key": MODELS.anthropic_api_key,
            "anthropic-version": _VERSION,
            "content-type": "application/json",
        }
        payload = dict(body, model=model)
        # Worst-case input estimate: 4 chars per token is deliberately pessimistic.
        estimated_input = len(json.dumps(payload)) // 3

        for attempt in range(1, self.max_attempts + 1):
            if self._circuit_open:
                return None
            try:
                reservation = self.budget.reserve(model, estimated_input, max_tokens)
            except BudgetExceeded as exc:
                log.error("budget guard refused a call: %s", exc)
                with self._lock:
                    self._circuit_open = True
                return None

            self.limiter.acquire()
            try:
                raw = self._session.post(
                    _API_URL, headers=headers, json=payload, timeout=self.timeout_seconds
                )
            except requests.RequestException as exc:
                self.budget.release(reservation)
                log.warning("anthropic %s attempt %d transport error: %s", model, attempt, exc)
                _sleep_backoff(attempt)
                continue

            if raw.status_code == 200:
                parsed = self._parse(raw.json(), model, attempt, reservation)
                if parsed is not None:
                    return parsed
                self.budget.release(reservation)
                return None

            self.budget.release(reservation)
            if raw.status_code in _RETRY_STATUS:
                if raw.status_code == 429:
                    self.limiter.penalise()
                    with self._lock:
                        self.stats.rate_limit_hits += 1
                _sleep_backoff(attempt, _retry_after(raw))
                continue

            detail = _error_detail(raw)
            log.error("anthropic %s permanent HTTP %d: %s", model, raw.status_code, detail)
            # Out of credit is terminal for the whole run, not just this call.
            if "credit balance" in detail.lower() or raw.status_code in {401, 403}:
                with self._lock:
                    self._circuit_open = True
            return None

        return None

    def _parse(self, payload: dict, model: str, attempt: int, reservation: float) -> LLMResponse | None:
        usage = payload.get("usage") or {}
        cache_write = usage.get("cache_creation_input_tokens", 0) or 0
        cache_read = usage.get("cache_read_input_tokens", 0) or 0
        input_tokens = usage.get("input_tokens", 0) or 0
        output_tokens = usage.get("output_tokens", 0) or 0
        self.budget.settle(model, reservation, input_tokens, output_tokens, cache_write, cache_read)

        text = ""
        for block in payload.get("content") or []:
            if block.get("type") == "tool_use":
                text = json.dumps(block.get("input") or {})
                break
            if block.get("type") == "text":
                text += block.get("text", "")
        if not text.strip():
            return None

        with self._lock:
            self.stats.calls += 1
            self.stats.retries += attempt - 1
            self.stats.prompt_tokens += input_tokens + cache_read + cache_write
            self.stats.output_tokens += output_tokens
            self.stats.by_model[model] = self.stats.by_model.get(model, 0) + 1

        return LLMResponse(
            text=text, model=model, provider=self.provider,
            prompt_tokens=input_tokens, output_tokens=output_tokens, attempts=attempt,
        )


def _inline_image(path: Path) -> dict | None:
    try:
        data = path.read_bytes()
    except OSError as exc:
        log.error("cannot read media %s: %s", path, exc)
        return None
    mime = sniff_mime(data, path.name)
    if not mime.startswith(_SUPPORTED_MEDIA_PREFIX):
        return None
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": mime,
                   "data": base64.b64encode(data).decode("ascii")},
    }


def _error_detail(response: requests.Response) -> str:
    try:
        return str(response.json().get("error", {}).get("message", ""))[:250]
    except ValueError:
        return response.text[:250]


def _retry_after(response: requests.Response) -> float | None:
    header = response.headers.get("retry-after")
    if header:
        try:
            return float(header)
        except ValueError:
            pass
    return None


def _sleep_backoff(attempt: int, retry_after: float | None = None) -> None:
    import random
    import time

    delay = min(retry_after if retry_after is not None
                else MODELS.backoff_base_seconds ** attempt, MODELS.backoff_max_seconds)
    time.sleep(delay * (0.7 + 0.6 * random.random()))
