"""Runtime configuration. Every value is overridable from the environment.

Secrets are read from the environment only - never hardcoded, never committed
(see AGENTS.md 6.3). ``.env`` is loaded if present, but real environment
variables always win so CI and graders can override without editing files.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

# code/router/config.py -> code/ -> repo root
CODE_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_DIR.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader. Real environment variables take precedence."""
    if not path.is_file():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


_load_dotenv(REPO_ROOT / ".env")
_load_dotenv(CODE_DIR / ".env")


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


# Every environment-derived field below uses ``default_factory``. A plain default
# is evaluated once, when the class body executes at import time, so re-creating
# the dataclass would silently reuse the original value - which quietly breaks
# `main.py ablate` and `main.py compare`, the two commands whose whole job is to
# vary configuration between runs.


@dataclass(frozen=True)
class Paths:
    repo_root: Path = REPO_ROOT
    code_dir: Path = CODE_DIR

    @property
    def dataset(self) -> Path:
        return Path(_env_str("ORCHESTRATE_DATASET_DIR", str(self.repo_root / "dataset")))

    @property
    def media(self) -> Path:
        return self.dataset / "media"

    @property
    def cache(self) -> Path:
        return Path(_env_str("ORCHESTRATE_CACHE_DIR", str(self.repo_root / ".cache")))

    @property
    def runs(self) -> Path:
        return Path(_env_str("ORCHESTRATE_RUNS_DIR", str(self.repo_root / "runs")))

    @property
    def output_csv(self) -> Path:
        return Path(_env_str("ORCHESTRATE_OUTPUT", str(self.repo_root / "output.csv")))


@dataclass(frozen=True)
class ModelConfig:
    """Gemini model selection and free-tier throughput controls.

    The free tier enforces both requests-per-minute and requests-per-day limits,
    and the exact ceiling differs per model and changes over time. Rather than
    hardcoding a number that will be wrong, the provider treats these as an
    opening guess and adapts down whenever the API returns 429.
    """

    # Ordered fallback chain. The provider walks it on persistent failure so a
    # single model's quota exhaustion degrades quality instead of killing the run.
    judge_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            _env_str(
                "ORCHESTRATE_JUDGE_MODELS",
                "gemini-2.5-flash,gemini-2.0-flash,gemini-2.5-flash-lite",
            ).split(",")
        )
    )
    media_models: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            _env_str(
                "ORCHESTRATE_MEDIA_MODELS",
                "gemini-2.5-flash,gemini-2.0-flash",
            ).split(",")
        )
    )

    api_base: str = _env_str(
        "ORCHESTRATE_GEMINI_BASE", "https://generativelanguage.googleapis.com/v1beta"
    )
    # Opening rate guess; the token bucket self-tunes downward on 429.
    requests_per_minute: int = field(default_factory=lambda: _env_int("ORCHESTRATE_RPM", 10))
    max_attempts: int = field(default_factory=lambda: _env_int("ORCHESTRATE_MAX_ATTEMPTS", 5))
    # Consecutive whole-chain failures before the client stops calling the API for
    # the rest of the run. A per-day quota wall is not worth rediscovering per row.
    circuit_breaker_threshold: int = field(
        default_factory=lambda: _env_int("ORCHESTRATE_CIRCUIT_BREAKER", 3)
    )
    backoff_base_seconds: float = field(default_factory=lambda: _env_float("ORCHESTRATE_BACKOFF_BASE", 2.0))
    backoff_max_seconds: float = field(default_factory=lambda: _env_float("ORCHESTRATE_BACKOFF_MAX", 60.0))
    timeout_seconds: float = field(default_factory=lambda: _env_float("ORCHESTRATE_TIMEOUT", 120.0))

    # Gemini 2.5 models think before answering. Thinking tokens are billed
    # against max_output_tokens, so a budget too small truncates the JSON body
    # mid-object. These values leave clear headroom above the observed usage.
    # Media understanding is enrichment, not correctness: a file that cannot be
    # read degrades one message's context, so it gets a tighter budget than the
    # judge and gives up quickly instead of stalling the run behind retries.
    media_max_attempts: int = field(default_factory=lambda: _env_int("ORCHESTRATE_MEDIA_MAX_ATTEMPTS", 2))
    media_timeout_seconds: float = field(default_factory=lambda: _env_float("ORCHESTRATE_MEDIA_TIMEOUT", 60.0))

    thinking_budget: int = field(default_factory=lambda: _env_int("ORCHESTRATE_THINKING_BUDGET", 1024))
    max_output_tokens: int = field(default_factory=lambda: _env_int("ORCHESTRATE_MAX_OUTPUT_TOKENS", 4096))
    temperature: float = field(default_factory=lambda: _env_float("ORCHESTRATE_TEMPERATURE", 0.0))

    @property
    def api_key(self) -> str:
        """Read at call time, never stored, never logged."""
        for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GOOGLE_GENAI_API_KEY"):
            value = os.environ.get(name, "").strip()
            if value:
                return value
        return ""


@dataclass(frozen=True)
class RouterConfig:
    """Pipeline behaviour knobs."""

    # Number of independent judge samples per message. >1 enables self-consistency
    # voting; disagreement is folded into the confidence as a genuine calibration
    # signal rather than being discarded.
    judge_samples: int = field(default_factory=lambda: _env_int("ORCHESTRATE_JUDGE_SAMPLES", 1))
    # Temperature used for samples 2..N when self-consistency is on. Sample 1 is
    # always greedy so a single-sample run stays fully deterministic.
    ensemble_temperature: float = field(default_factory=lambda: _env_float("ORCHESTRATE_ENSEMBLE_TEMPERATURE", 0.4))

    # Evidence retrieval.
    evidence_candidates: int = field(default_factory=lambda: _env_int("ORCHESTRATE_EVIDENCE_CANDIDATES", 8))
    evidence_max_emitted: int = field(default_factory=lambda: _env_int("ORCHESTRATE_EVIDENCE_MAX", 2))
    evidence_min_score: float = field(default_factory=lambda: _env_float("ORCHESTRATE_EVIDENCE_MIN_SCORE", 0.12))

    # auto = replay the expert artifact where it covers a message, else call the
    # online judge; expert / online force one arm; none is deterministic-only.
    judge_source: str = field(default_factory=lambda: _env_str("ORCHESTRATE_JUDGE_SOURCE", "auto"))
    use_llm: bool = field(default_factory=lambda: _env_bool("ORCHESTRATE_USE_LLM", True))
    use_media: bool = field(default_factory=lambda: _env_bool("ORCHESTRATE_USE_MEDIA", True))
    # Cache is what makes reruns free and keeps the pipeline reproducible under
    # a rate-limited key. Disable only to force a genuine re-query.
    use_cache: bool = field(default_factory=lambda: _env_bool("ORCHESTRATE_USE_CACHE", True))
    workers: int = field(default_factory=lambda: _env_int("ORCHESTRATE_WORKERS", 4))


PATHS = Paths()
MODELS = ModelConfig()
ROUTER = RouterConfig()
