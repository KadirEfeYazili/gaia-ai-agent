"""The GAIA agent.

A thin, well-documented wrapper around a smolagents ``CodeAgent`` (which reasons
by writing and executing Python) that:
  * builds the (configurable) LLM backend from :mod:`config`,
  * registers a focused set of tools from :mod:`tools`,
  * applies GAIA-specific prompting from :mod:`prompts`,
  * passes attached images to the multimodal model, and
  * extracts and normalises the final answer for exact-match grading.

The public contract matches the template's ``BasicAgent`` -- an instance is
called with the question string -- with two optional extras (``task_id`` and
``file_name``) so that file-based tasks work.
"""
from __future__ import annotations

import logging
import os
import re
import time
from typing import List, Optional

from PIL import Image
from smolagents import CodeAgent, VisitWebpageTool

from config import CONFIG, AgentConfig, build_model
from prompts import GAIA_ANSWER_RULES, build_task_prompt
from tools import (
    download_gaia_file,
    fetch_gaia_file,
    transcribe_audio,
    web_search,
    wikipedia_search,
)

logger = logging.getLogger("gaia.agent")

# Attachment extensions handed to the model as images (Gemini is multimodal).
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"}

# Imports the CodeAgent may use in the Python it writes. These cover the parsing
# and computation GAIA needs without opening the sandbox up wholesale.
_AUTHORIZED_IMPORTS = [
    "pandas", "numpy", "requests", "bs4", "openpyxl",
    "math", "statistics", "datetime", "time", "json", "re", "csv",
    "io", "os", "collections", "itertools", "string", "pathlib",
    "urllib", "zipfile", "PIL",
]

# Substrings that mark a *transient* failure worth retrying (network blips,
# rate limits, provider overload) rather than a genuine error.
_TRANSIENT_MARKERS = (
    "getaddrinfo", "apiconnectionerror", "serviceunavailable", "503",
    "ratelimiterror", "429", "high demand", "unavailable", "overloaded",
    "connection", "timeout", "temporarily",
)


def _is_transient(exc: Exception) -> bool:
    """True if the exception looks like a transient network/provider error."""
    text = f"{type(exc).__name__} {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_MARKERS)


# Patterns for the provider-suggested wait (e.g. Groq: "Please try again in 7.5s").
_RETRY_AFTER_PATTERNS = (
    r"try again in ([0-9]+(?:\.[0-9]+)?)\s*s",
    r'retrydelay"?\s*[:=]\s*"?([0-9]+(?:\.[0-9]+)?)',
    r"retry[- ]?after[\"\s:=]+([0-9]+(?:\.[0-9]+)?)",
)


def _retry_delay_seconds(exc: Exception, fallback: float, cap: float = 60.0) -> float:
    """Honor the provider's suggested wait (Retry-After / 'try again in Ns'), capped.

    Falls back to the given exponential-backoff value when no hint is present, so
    we never sit on a fixed long sleep.
    """
    resp = getattr(exc, "response", None)
    try:
        header = resp.headers.get("retry-after") if resp is not None else None
        if header:
            return min(float(header), cap)
    except Exception:  # noqa: BLE001
        pass
    text = str(exc).lower()
    for pattern in _RETRY_AFTER_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return min(float(match.group(1)), cap)
    return fallback


class GaiaAgent:
    """A GAIA-benchmark agent with reasoning, tool use and file handling."""

    def __init__(self, config: AgentConfig = CONFIG) -> None:
        self.config = config
        self.model = build_model(config)
        self.agent = CodeAgent(
            model=self.model,
            tools=[web_search, wikipedia_search, VisitWebpageTool(max_output_length=3000),
                   fetch_gaia_file, transcribe_audio],
            additional_authorized_imports=_AUTHORIZED_IMPORTS,
            max_steps=config.max_steps,
            verbosity_level=2 if config.verbose else 0,
        )
        logger.info(
            "GaiaAgent ready (provider=%s, model=%s, max_steps=%d)",
            config.model_provider, config.model_id, config.max_steps,
        )

    def __call__(
        self,
        question: str,
        task_id: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> str:
        """Answer a single GAIA question and return an exact-match-ready string.

        Transient failures (DNS blips, rate limits, provider 503s) are retried
        with exponential backoff; only after ``max_retries`` attempts do we give
        up and return an ``AGENT ERROR`` string so the overall run can continue.
        """
        prompt = build_task_prompt(question, task_id, file_name)
        images = self._maybe_load_image(task_id, file_name)
        logger.info("Running task %s (file=%s, image=%s)", task_id, file_name, bool(images))

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                raw = self.agent.run(prompt, images=images) if images else self.agent.run(prompt)
                answer = self._extract_final_answer(str(raw))
                # If the run ended without a concrete answer, derive one from the
                # reasoning it produced along the way.
                if self._looks_unfinished(answer):
                    derived = self._answer_from_reasoning(question)
                    if derived:
                        answer = derived
                logger.info("Task %s answer: %r", task_id, answer)
                return answer
            except Exception as exc:  # noqa: BLE001 - isolate failures per task
                last_exc = exc
                if _is_transient(exc) and attempt < self.config.max_retries:
                    fallback = min(self.config.retry_base_delay * 2 ** (attempt - 1), 30.0)
                    delay = _retry_delay_seconds(exc, fallback)
                    logger.warning(
                        "Task %s transient error (attempt %d/%d): %s -- retrying in %.1fs",
                        task_id, attempt, self.config.max_retries, str(exc)[:150], delay,
                    )
                    time.sleep(delay)
                    continue
                logger.error("Task %s failed: %s", task_id, exc)
                return f"AGENT ERROR: {exc}"
        return f"AGENT ERROR: {last_exc}"

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _looks_unfinished(answer: str) -> bool:
        """True when the text is empty or reads as reasoning rather than an answer."""
        text = (answer or "").strip()
        if not text:
            return True
        return text.lower().startswith(
            ("thought", "action", "observation", "```", "let's", "let me", "i will", "i'll")
        )

    def _answer_from_reasoning(self, question: str) -> str:
        """Derive a final answer from the reasoning captured during the run.

        A run occasionally ends without producing a concrete answer; the steps it
        recorded usually still contain the needed information. This condenses those
        steps and requests a single answer in GAIA's expected format.
        """
        try:
            parts: List[str] = []
            for step in getattr(self.agent.memory, "steps", []):
                output = getattr(step, "model_output", None)
                observation = getattr(step, "observations", None)
                if output:
                    parts.append(str(output))
                if observation:
                    parts.append(f"Observation: {observation}")
            context = "\n".join(parts)[-6000:]
            if not context:
                return ""
            instruction = (
                f"Question:\n{question}\n\nResearch notes:\n{context}\n\n"
                "Reply with only the answer: no explanation, no prefix, no trailing period. "
                "Numbers as digits without separators or units unless asked; strings without "
                "articles or abbreviations; lists comma-separated. If the notes are inconclusive, "
                "give your single best answer."
            )
            messages = [
                {"role": "system", "content": [{"type": "text", "text": GAIA_ANSWER_RULES}]},
                {"role": "user", "content": [{"type": "text", "text": instruction}]},
            ]
            response = self.model.generate(messages)
            text = getattr(response, "content", None) or str(response)
            return self._extract_final_answer(str(text))
        except Exception as exc:  # noqa: BLE001 - best-effort fallback
            logger.warning("Could not derive answer from reasoning: %s", exc)
            return ""

    def _maybe_load_image(
        self, task_id: Optional[str], file_name: Optional[str]
    ) -> Optional[List[Image.Image]]:
        """Return ``[PIL.Image]`` for image attachments, otherwise ``None``."""
        if not task_id or not file_name:
            return None
        if os.path.splitext(file_name)[1].lower() not in _IMAGE_EXTS:
            return None
        try:
            path = download_gaia_file(task_id, file_name)
            return [Image.open(path)]
        except Exception as exc:  # noqa: BLE001 - missing files are expected; degrade quietly
            logger.info("No image preloaded for task %s (%s)", task_id, exc)
            return None

    @staticmethod
    def _extract_final_answer(text: str) -> str:
        """Pull the answer out of the model output and normalise it.

        Handles both the CodeAgent's ``final_answer`` return value and any stray
        ``FINAL ANSWER:`` prefix, then strips markdown/quotes and a trailing dot.
        """
        text = (text or "").strip()
        matches = re.findall(r"FINAL ANSWER:\s*(.+)", text, flags=re.IGNORECASE)
        answer = matches[-1] if matches else text
        # Keep a single line (a GAIA answer should be one line).
        lines = [ln.strip() for ln in answer.splitlines() if ln.strip()]
        answer = lines[0] if lines else ""
        # Strip surrounding markdown emphasis / code ticks / quotes.
        answer = answer.strip(" *`\"'")
        # Drop a single trailing period ("Paris." -> "Paris").
        if answer.endswith("."):
            answer = answer[:-1].strip()
        return answer
