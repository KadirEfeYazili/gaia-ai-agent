"""Gradio app + submission runner for the HF Agents Course final assignment.

This file keeps the *official* evaluation workflow intact -- fetch questions
from the course scoring API, run the agent over them, submit the answers and
show the score -- and simply swaps the placeholder ``BasicAgent`` for the real
:class:`agent.GaiaAgent`. The Gradio UI (HF login + run button + results table)
is preserved as required by the assignment.
"""
import logging
import os

import gradio as gr
import pandas as pd
import requests

from agent import GaiaAgent

# --- Logging ---------------------------------------------------------------
# Configured once at import so messages appear both in local runs and in the
# Hugging Face Spaces container logs.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("gaia.app")

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetches all questions, runs the agent on them, submits all answers,
    and displays the results.
    """
    # --- Determine HF Space Runtime URL and Repo URL ---
    space_id = os.getenv("SPACE_ID")  # Used to build a link to this Space's code.

    if profile:
        username = f"{profile.username}"
        logger.info("User logged in: %s", username)
    else:
        logger.info("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    # 1. Instantiate Agent (swapped in for the template's BasicAgent).
    try:
        agent = GaiaAgent()
    except Exception as e:
        logger.exception("Error instantiating agent")
        return (
            f"Error initializing agent: {e}\n\n"
            "Make sure GEMINI_API_KEY is configured (Space secret or local .env).",
            None,
        )
    # This link points to your (public) code, and is required by the submission.
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    logger.info("Agent code URL: %s", agent_code)

    # 2. Fetch Questions
    logger.info("Fetching questions from: %s", questions_url)
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            logger.warning("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        logger.info("Fetched %d questions.", len(questions_data))
    except requests.exceptions.RequestException as e:
        logger.error("Error fetching questions: %s", e)
        return f"Error fetching questions: {e}", None
    except requests.exceptions.JSONDecodeError as e:
        logger.error("Error decoding JSON response from questions endpoint: %s", e)
        return f"Error decoding server response for questions: {e}", None
    except Exception as e:
        logger.exception("Unexpected error fetching questions")
        return f"An unexpected error occurred fetching questions: {e}", None

    # 3. Run the Agent on every question.
    results_log = []
    answers_payload = []
    logger.info("Running agent on %d questions...", len(questions_data))
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        file_name = item.get("file_name") or None  # non-empty when a file is attached
        if not task_id or question_text is None:
            logger.warning("Skipping item with missing task_id or question: %s", item)
            continue
        try:
            submitted_answer = agent(question_text, task_id=task_id, file_name=file_name)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append(
                {"Task ID": task_id, "Question": question_text, "Submitted Answer": submitted_answer}
            )
        except Exception as e:
            logger.exception("Error running agent on task %s", task_id)
            results_log.append(
                {"Task ID": task_id, "Question": question_text, "Submitted Answer": f"AGENT ERROR: {e}"}
            )

    if not answers_payload:
        logger.warning("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    # 4. Prepare Submission
    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    logger.info(status_update)

    # 5. Submit
    logger.info("Submitting %d answers to: %s", len(answers_payload), submit_url)
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        logger.info("Submission successful.")
        results_df = pd.DataFrame(results_log)
        return final_status, results_df
    except requests.exceptions.HTTPError as e:
        error_detail = f"Server responded with status {e.response.status_code}."
        try:
            error_json = e.response.json()
            error_detail += f" Detail: {error_json.get('detail', e.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {e.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except requests.exceptions.RequestException as e:
        status_message = f"Submission Failed: Network error - {e}"
        logger.error(status_message)
        results_df = pd.DataFrame(results_log)
        return status_message, results_df
    except Exception as e:
        status_message = f"An unexpected error occurred during submission: {e}"
        logger.exception("Unexpected error during submission")
        results_df = pd.DataFrame(results_log)
        return status_message, results_df


# --- Build Gradio Interface using Blocks ---
with gr.Blocks() as demo:
    gr.Markdown("# GAIA Agent - Course Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**

        1. Log in to your Hugging Face account with the button below (this uses your HF username for submission).
        2. Click 'Run Evaluation & Submit All Answers' to fetch the questions, run the agent, submit the answers, and see the score.

        ---
        **Notes:**
        - A full run can take several minutes: the agent reasons and calls tools for each question.
        - This project was built for the Hugging Face Agents Course final assignment.
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers")

    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])


if __name__ == "__main__":
    logger.info("%s App Starting %s", "-" * 20, "-" * 20)
    # Log SPACE_HOST / SPACE_ID at startup for information (present on Spaces).
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        logger.info("SPACE_HOST found: %s", space_host_startup)
        logger.info("   Runtime URL: https://%s.hf.space", space_host_startup)
    else:
        logger.info("SPACE_HOST not found (running locally?).")

    if space_id_startup:
        logger.info("SPACE_ID found: %s", space_id_startup)
        logger.info("   Repo URL: https://huggingface.co/spaces/%s", space_id_startup)
    else:
        logger.info("SPACE_ID not found (running locally?). Repo URL cannot be determined.")

    logger.info("Launching Gradio Interface for GAIA Agent Evaluation...")
    demo.launch(debug=True, share=False)
