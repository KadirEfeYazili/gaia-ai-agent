"""Quick local smoke test for the GAIA agent.

Fetches question(s) from the scoring API and runs the agent on them locally so
you can validate your setup (and ``GEMINI_API_KEY``) without submitting anything.

Usage:
    python smoke_test.py            # one random question
    python smoke_test.py --all      # the full public question set (no submission)
"""
from __future__ import annotations

import argparse
import logging

import requests

from agent import GaiaAgent
from config import CONFIG


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GAIA agent locally without submitting.")
    parser.add_argument("--all", action="store_true", help="Run every public question (no submission).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )

    agent = GaiaAgent()
    if args.all:
        questions = requests.get(f"{CONFIG.api_url}/questions", timeout=30).json()
    else:
        questions = [requests.get(f"{CONFIG.api_url}/random-question", timeout=30).json()]

    for q in questions:
        print("\n" + "=" * 80)
        print("Q:", q.get("question"))
        answer = agent(
            q.get("question"),
            task_id=q.get("task_id"),
            file_name=q.get("file_name") or None,
        )
        print("A:", answer)


if __name__ == "__main__":
    main()
