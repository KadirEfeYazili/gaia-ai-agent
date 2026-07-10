"""Central configuration for the GAIA agent.

Every tunable is read from an environment variable so the exact same code runs
locally (via a ``.env`` file) and on Hugging Face Spaces (via repository
secrets). Importing this module performs no network calls and does not require
any secret to be present -- a missing key only surfaces when the model is
actually invoked.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass

# Load a local .env if present. This is a no-op on Spaces (there is no .env
# there) and is guarded so the app still starts if python-dotenv is missing.
try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - optional dependency
    pass

logger = logging.getLogger("gaia.config")

# Default scoring API for the course's final assignment.
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
# Gemini free tier default; override with GAIA_MODEL_ID (e.g. gemini/gemini-2.5-flash).
DEFAULT_MODEL_ID = "gemini/gemini-2.0-flash"


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except (TypeError, ValueError):
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AgentConfig:
    """Immutable snapshot of the agent's runtime configuration."""

    api_url: str = os.getenv("GAIA_API_URL", DEFAULT_API_URL)
    model_id: str = os.getenv("GAIA_MODEL_ID", DEFAULT_MODEL_ID)
    # Which smolagents model wrapper to use: "litellm" (default; covers Gemini,
    # OpenAI, Anthropic, ...), "hf" (Inference Providers) or "openai_server".
    model_provider: str = os.getenv("GAIA_MODEL_PROVIDER", "litellm")
    api_base: str | None = os.getenv("GAIA_API_BASE") or None
    # Speech-to-text model (LiteLLM id) used by the transcribe_audio tool.
    transcribe_model: str = os.getenv("GAIA_TRANSCRIBE_MODEL", "groq/whisper-large-v3")
    temperature: float = _env_float("GAIA_TEMPERATURE", 0.1)
    max_steps: int = _env_int("GAIA_MAX_STEPS", 6)
    max_tokens: int = _env_int("GAIA_MAX_TOKENS", 2048)
    request_timeout: int = _env_int("GAIA_REQUEST_TIMEOUT", 60)
    verbose: bool = _env_bool("GAIA_VERBOSE", False)
    # Retry transient model/network errors (503/429/DNS blips) this many times.
    max_retries: int = _env_int("GAIA_MAX_RETRIES", 3)
    retry_base_delay: float = _env_float("GAIA_RETRY_DELAY", 20.0)
    # Model-call retries inside smolagents. On tight free tiers (e.g. Groq's
    # 12k tokens/min) a rate limit should be absorbed here as a wait, NOT bubble
    # up and restart the whole task -- so keep this generous.
    model_retry_attempts: int = _env_int("GAIA_MODEL_RETRIES", 8)
    # Proactively throttle requests/min so we stay under the provider's
    # per-minute budget instead of hammering it and getting rate-limited.
    # 0 disables throttling. ~1.4 keeps a 70B under Groq's 12k TPM.
    requests_per_minute: float = _env_float("GAIA_RPM", 0.0)


# Shared, import-time configuration instance used across the app.
CONFIG = AgentConfig()


def _resolve_api_key(model_id: str) -> str | None:
    """Return the provider API key matching the given model id, if it is set."""
    m = model_id.lower()
    if "gemini" in m or m.startswith("google/"):
        return os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if m.startswith(("gpt", "o1", "o3", "openai/", "chatgpt")):
        return os.getenv("OPENAI_API_KEY")
    if "claude" in m or m.startswith("anthropic/"):
        return os.getenv("ANTHROPIC_API_KEY")
    if m.startswith("groq/"):
        return os.getenv("GROQ_API_KEY")
    if m.startswith("openrouter/"):
        return os.getenv("OPENROUTER_API_KEY")
    return None


def build_model(config: AgentConfig | None = None):
    """Construct the smolagents model wrapper described by ``config``.

    The provider is selected by ``config.model_provider`` so the LLM backend can
    be swapped purely through environment variables, with no code changes.
    """
    config = config or CONFIG
    provider = config.model_provider.lower()

    # smolagents has its own in-place rate-limit retryer whose default base wait
    # is 60s with exponential backoff (i.e. ~60/120/180s). Shorten it so a rate
    # limit costs seconds, not minutes; GaiaAgent's own loop is the outer net.
    import smolagents.models as _sm_models

    _sm_models.RETRY_WAIT = max(1, int(config.retry_base_delay))
    # Absorb rate limits as in-place waits (see AgentConfig.model_retry_attempts).
    _sm_models.RETRY_MAX_ATTEMPTS = max(1, config.model_retry_attempts)

    if provider == "hf":
        from huggingface_hub import get_token
        from smolagents import InferenceClientModel

        # HF_TOKEN env var on Spaces; fall back to the locally cached login token.
        token = (
            os.getenv("HF_TOKEN")
            or os.getenv("HUGGINGFACEHUB_API_TOKEN")
            or get_token()
        )
        return InferenceClientModel(
            model_id=config.model_id, token=token, max_tokens=config.max_tokens
        )

    if provider == "openai_server":
        from smolagents import OpenAIServerModel

        return OpenAIServerModel(
            model_id=config.model_id,
            api_base=config.api_base,
            api_key=_resolve_api_key(config.model_id),
            max_tokens=config.max_tokens,
        )

    # Default: LiteLLM -- one wrapper, many providers (Gemini, OpenAI, ...).
    from smolagents import LiteLLMModel

    kwargs: dict = {
        "model_id": config.model_id,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "num_retries": 0,  # GaiaAgent owns retry timing so it can honor Retry-After
    }
    if config.requests_per_minute > 0:
        kwargs["requests_per_minute"] = config.requests_per_minute
    api_key = _resolve_api_key(config.model_id)
    if api_key:
        kwargs["api_key"] = api_key
    if config.api_base:
        kwargs["api_base"] = config.api_base
    return LiteLLMModel(**kwargs)
