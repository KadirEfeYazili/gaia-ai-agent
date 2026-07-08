"""Custom tools exposed to the GAIA CodeAgent.

Only capabilities the agent cannot already perform in plain Python are provided
here: web/Wikipedia search and downloading a task's attached file. Arithmetic,
dates, string manipulation and file *parsing* (xlsx/csv/json/py) are handled by
the CodeAgent directly in the Python it writes, so wrapping them as tools would
be redundant.
"""
from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Optional

import requests
from smolagents import tool

from config import CONFIG

logger = logging.getLogger("gaia.tools")

# A descriptive User-Agent; some services (e.g. the Wikipedia API) expect one.
_HEADERS = {"User-Agent": "gaia-agent/1.0 (HF Agents Course final assignment)"}


@tool
def web_search(query: str, max_results: int = 5) -> str:
    """Search the web (DuckDuckGo) and return the top results as text.

    Retries with exponential backoff when the search endpoint rate-limits us,
    which DuckDuckGo does frequently during longer runs.

    Args:
        query: The search query.
        max_results: How many results to return (clamped to the range 1-10).

    Returns:
        Newline-separated "title / url / snippet" blocks, or an error message.
    """
    from ddgs import DDGS

    n = max(1, min(int(max_results), 10))
    last_error: Optional[Exception] = None
    for attempt in range(4):
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=n))
            if not results:
                return f"No web results found for: {query!r}"
            blocks = []
            for r in results:
                title = r.get("title", "")
                href = r.get("href") or r.get("url", "")
                body = (r.get("body") or r.get("snippet", ""))[:250]
                blocks.append(f"- {title}\n  {href}\n  {body}")
            return "\n".join(blocks)
        except Exception as exc:  # noqa: BLE001 - report a readable message to the agent
            last_error = exc
            delay = 2 ** attempt
            logger.warning("web_search attempt %d failed: %s (retry in %ds)", attempt + 1, exc, delay)
            time.sleep(delay)
    return f"Web search failed after retries: {last_error}"


@tool
def wikipedia_search(query: str) -> str:
    """Look up a topic on English Wikipedia via the MediaWiki API.

    Returns the best-matching page's title, canonical URL and introductory
    extract. Use `visit_webpage` on that URL when you need the full article
    (for example a discography or results table).

    Args:
        query: The topic or page title to search for.

    Returns:
        "title / url / intro extract", or an error message.
    """
    api = "https://en.wikipedia.org/w/api.php"
    try:
        search = requests.get(
            api,
            params={"action": "query", "list": "search", "srsearch": query,
                    "srlimit": 1, "format": "json"},
            headers=_HEADERS, timeout=CONFIG.request_timeout,
        )
        search.raise_for_status()
        hits = search.json().get("query", {}).get("search", [])
        if not hits:
            return f"No Wikipedia page found for: {query!r}"
        title = hits[0]["title"]

        extract = requests.get(
            api,
            params={"action": "query", "prop": "extracts|info", "inprop": "url",
                    "explaintext": 1, "exintro": 1, "titles": title, "format": "json"},
            headers=_HEADERS, timeout=CONFIG.request_timeout,
        )
        extract.raise_for_status()
        pages = extract.json().get("query", {}).get("pages", {})
        page = next(iter(pages.values()), {})
        url = page.get("fullurl", "https://en.wikipedia.org/wiki/" + title.replace(" ", "_"))
        summary = page.get("extract", "").strip()[:1200]
        return f"{title}\n{url}\n\n{summary}"
    except Exception as exc:  # noqa: BLE001
        logger.warning("wikipedia_search failed: %s", exc)
        return f"Wikipedia lookup failed: {exc}"


def download_gaia_file(task_id: str, file_name: str = "") -> str:
    """Download a task's attachment to a temp file and return its local path.

    Plain helper (not a tool) so it can be reused for image preloading in the
    agent as well as by the ``fetch_gaia_file`` tool.
    """
    url = f"{CONFIG.api_url}/files/{task_id}"
    response = requests.get(url, headers=_HEADERS, timeout=CONFIG.request_timeout)
    response.raise_for_status()
    ext = os.path.splitext(file_name)[1] if file_name else ""
    path = os.path.join(tempfile.gettempdir(), f"gaia_{task_id}{ext}")
    with open(path, "wb") as fh:
        fh.write(response.content)
    logger.info("Downloaded GAIA file for task %s -> %s (%d bytes)", task_id, path, len(response.content))
    return path


@tool
def fetch_gaia_file(task_id: str, file_name: str = "") -> str:
    """Download the file attached to a GAIA task and return its local path.

    After calling this, read the file with ordinary Python: pandas.read_excel /
    read_csv for spreadsheets, open() for text/code/JSON, PIL.Image.open for
    images, and so on.

    Args:
        task_id: The GAIA task id whose attachment should be downloaded.
        file_name: Original file name; used only to choose the saved extension.

    Returns:
        The absolute local path to the downloaded file, or an error message.
    """
    try:
        return download_gaia_file(task_id, file_name)
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_gaia_file failed for %s: %s", task_id, exc)
        return f"Failed to download file for task {task_id}: {exc}"
