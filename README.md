# GAIA Agent — Hugging Face Agents Course

A modular, tool-using LLM agent designed to solve the complex [GAIA (General AI Assistants)](https://huggingface.co/gaia-benchmark) benchmark questions. Built using the `smolagents` framework, this project demonstrates an advanced implementation of a `CodeAgent` that reasons by autonomously writing and executing Python code.

## 🚀 How It Works

The agent acts as a dynamic problem solver. Instead of relying solely on hardcoded tools, it leverages a Python interpreter to handle arithmetic, dates, and file parsing (`.xlsx`, `.csv`, `.json`, `.py`) naturally within its execution environment.

| File | Responsibility |
|------|----------------|
| `app.py` | Gradio UI + official evaluation and submission workflow. |
| `agent.py` | `GaiaAgent`: builds the model, registers tools, and normalizes final answers. |
| `tools.py` | Agent tools: `web_search`, `wikipedia_search`, `visit_webpage`, `fetch_gaia_file`, `transcribe_audio`. |
| `prompts.py` | Formatting rules and task-specific prompt engineering. |
| `config.py` | Environment-driven configurations and swappable LLM backends. |

## 🧰 Agent Capabilities

| Capability | How it's used |
|------------|----------------|
| Web search | `web_search` (DuckDuckGo) finds open-web facts; `visit_webpage` reads the chosen page. |
| Encyclopedic lookup | `wikipedia_search` for entities, dates, and reference data. |
| File handling | `fetch_gaia_file` downloads an attachment; the Python interpreter parses `.xlsx`, `.csv`, `.json`, `.py`. |
| Audio | `transcribe_audio` (Groq Whisper) turns voice-memo attachments into text. |
| Vision | Image attachments are passed to the multimodal model for picture-based questions. |
| Computation | The CodeAgent writes and runs Python for math, dates, and data wrangling. |

## 🛠️ Installation & Local Setup

1. **Clone the repository and set up the environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Configure your API keys:**
   Copy the environment template and set your model provider credentials:
   ```bash
   cp .env.example .env
   ```

3. **Run the agent:**
   * Run the agent locally on a random question:
     ```bash
     python smoke_test.py
     ```
   * Run the web interface locally via Gradio:
     ```bash
     python app.py
     ```

## 🎯 Key Features & Future Roadmap
- **Code-As-Action:** Uses LLM code generation to manipulate data sheets and run calculations dynamically.
- **Extensible Architecture:** Easily swap backend models (OpenAI, Anthropic, Hugging Face) via `config.py`.
- **Multimodal Input:** Transcribes audio attachments (Groq Whisper) and passes images to the multimodal model.
- **Planned Enhancements:** Integrating premium search APIs (Tavily/Serper) for more reliable retrieval.

## 🏅 Certification
This project has been officially verified by the Hugging Face team. You can view the formal completion credential [here](https://cdn-uploads.huggingface.co/production/uploads/noauth/hBhZKCiCSFT7BaZe61zfe.webp).

## 💳 Credits
Developed as part of the Hugging Face Agents Course. Powered by `smolagents`.

