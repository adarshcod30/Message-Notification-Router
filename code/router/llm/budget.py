"""Hard spend ceiling for paid model calls.

This exists because the account funding this run has a small, fixed balance. A
retry storm or an accidental `--judge online` on the wrong dataset could burn it
in minutes, so the guard is a *precondition* on every call rather than a report
written afterwards.

Two properties matter:

* **It refuses before spending, not after.** ``reserve()`` is called with a
  worst-case estimate before the request goes out. If that estimate would breach
  the ceiling the call never happens.
* **It settles on actuals.** After the response arrives, ``settle()`` replaces the
  estimate with the real token counts, so an over-cautious estimate does not
  permanently consume headroom.

Prices are per million tokens and must be kept in step with the provider's
published rates; they are declared here rather than inline so there is exactly
one place to update.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelPricing:
    """USD per million tokens."""

    input: float
    output: float
    cache_write: float
    cache_read: float

    def cost(
        self, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0
    ) -> float:
        return (
            input_tokens * self.input
            + output_tokens * self.output
            + cache_write * self.cache_write
            + cache_read * self.cache_read
        ) / 1_000_000


# Anthropic list prices, USD per million tokens.
PRICING: dict[str, ModelPricing] = {
    "claude-sonnet-4-5-20250929": ModelPricing(3.00, 15.00, 3.75, 0.30),
    "claude-haiku-4-5-20251001": ModelPricing(1.00, 5.00, 1.25, 0.10),
    "claude-opus-4-1-20250805": ModelPricing(15.00, 75.00, 18.75, 1.50),
}
# Gemini free tier costs nothing; priced at zero so the same guard can wrap it.
_FREE = ModelPricing(0.0, 0.0, 0.0, 0.0)


def pricing_for(model: str) -> ModelPricing:
    if model in PRICING:
        return PRICING[model]
    if model.startswith("gemini"):
        return _FREE
    # Unknown paid model: price it as the most expensive one we know, so an
    # unrecognised name is conservative rather than free.
    return max(PRICING.values(), key=lambda p: p.output)


class BudgetExceeded(RuntimeError):
    """Raised when a call would breach the configured ceiling."""


@dataclass
class BudgetGuard:
    """Thread-safe spend tracker with a hard ceiling."""

    ceiling_usd: float
    spent_usd: float = 0.0
    reserved_usd: float = 0.0
    calls: int = 0
    refused: int = 0
    by_model: dict[str, float] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    @property
    def committed_usd(self) -> float:
        return self.spent_usd + self.reserved_usd

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.ceiling_usd - self.committed_usd)

    def can_afford(self, estimate_usd: float) -> bool:
        with self._lock:
            return self.committed_usd + estimate_usd <= self.ceiling_usd

    def reserve(self, model: str, input_tokens: int, max_output_tokens: int) -> float:
        """Hold worst-case cost for a pending call. Raises if unaffordable."""
        price = pricing_for(model)
        estimate = price.cost(input_tokens, max_output_tokens)
        with self._lock:
            if self.spent_usd + self.reserved_usd + estimate > self.ceiling_usd:
                self.refused += 1
                raise BudgetExceeded(
                    f"call to {model} needs ~${estimate:.4f}; "
                    f"${self.remaining_usd:.4f} of the ${self.ceiling_usd:.2f} ceiling remains"
                )
            self.reserved_usd += estimate
        return estimate

    def settle(
        self,
        model: str,
        estimate_usd: float,
        input_tokens: int,
        output_tokens: int,
        cache_write: int = 0,
        cache_read: int = 0,
    ) -> float:
        """Release the reservation and record what the call actually cost."""
        actual = pricing_for(model).cost(input_tokens, output_tokens, cache_write, cache_read)
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - estimate_usd)
            self.spent_usd += actual
            self.calls += 1
            self.by_model[model] = self.by_model.get(model, 0.0) + actual
        return actual

    def release(self, estimate_usd: float) -> None:
        """Drop a reservation for a call that never completed."""
        with self._lock:
            self.reserved_usd = max(0.0, self.reserved_usd - estimate_usd)

    def as_dict(self) -> dict:
        return {
            "ceiling_usd": round(self.ceiling_usd, 4),
            "spent_usd": round(self.spent_usd, 4),
            "remaining_usd": round(self.remaining_usd, 4),
            "paid_calls": self.calls,
            "refused_calls": self.refused,
            "by_model": {k: round(v, 4) for k, v in self.by_model.items()},
        }
