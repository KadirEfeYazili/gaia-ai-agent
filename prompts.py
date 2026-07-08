"""Prompt fragments for the GAIA agent.

GAIA answers are graded by *normalised exact match*, so the model must return
only the answer, in a strict format. These helpers append the GAIA answer-format
rules to each question; the agent then returns the answer via ``final_answer``
and :meth:`agent.GaiaAgent._extract_final_answer` normalises it before it is
submitted.
"""
from __future__ import annotations

# High-level guidance on how to approach a task and which tool to reach for.
GAIA_SYSTEM_GUIDANCE = """Answer the GAIA question below. Verify facts with tools instead of guessing: \
web_search then visit_webpage; wikipedia_search; fetch_gaia_file for attachments. Do all math/data in code."""

# Answer-format rules adapted from the GAIA benchmark's system prompt. Keeping
# them precise matters: the grader is exact-match.
GAIA_ANSWER_RULES = """Call final_answer with ONLY the answer (no explanation, no trailing period):
- number: digits only, no thousands separators or units unless asked.
- string: no articles, no abbreviations.
- list: comma-separated with ", ", applying these rules to each item."""


def build_task_prompt(
    question: str,
    task_id: str | None = None,
    file_name: str | None = None,
) -> str:
    """Compose the full prompt handed to the agent for a single GAIA task."""
    parts = [GAIA_SYSTEM_GUIDANCE]
    if file_name:
        parts.append(
            f'This task has an attached file named "{file_name}". '
            f'Download it first with fetch_gaia_file(task_id="{task_id}", file_name="{file_name}") '
            "and read its contents before answering."
        )
    parts.append(f"QUESTION:\n{question}")
    parts.append(GAIA_ANSWER_RULES)
    return "\n\n".join(parts)
