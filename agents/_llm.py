from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, List, Optional

import requests

DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b-instruct")
DEFAULT_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class OllamaError(RuntimeError):
    """Raised when the Ollama API returns an error."""


def ollama_chat(
    messages: Iterable[Dict[str, str]],
    *,
    model: Optional[str] = None,
    temperature: float = 0.2,
    max_retries: int = 3,
    request_timeout: int = 120,
) -> str:
    """Send a chat completion request to a local Ollama server.

    Parameters
    ----------
    messages:
        Sequence of messages in the OpenAI-compatible format.
    model:
        Name of the Ollama model to use. Defaults to :data:`DEFAULT_MODEL`.
    temperature:
        Sampling temperature. Use lower values for more deterministic output.
    max_retries:
        Number of times to retry failed requests with exponential backoff.
    request_timeout:
        Timeout for the HTTP request in seconds.
    """

    base_url = DEFAULT_BASE_URL.rstrip("/")
    model_name = model or DEFAULT_MODEL
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": list(messages),
        "stream": False,
        "options": {"temperature": temperature},
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(
                f"{base_url}/api/chat",
                json=payload,
                timeout=request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            message = data.get("message") or {}
            content = message.get("content")
            if not content:
                raise OllamaError("Ollama response did not include content")
            return content.strip()
        except (requests.RequestException, OllamaError) as exc:
            if attempt == max_retries:
                raise OllamaError(f"Ollama request failed: {exc}") from exc
            sleep_time = 2 ** (attempt - 1)
            time.sleep(sleep_time)

    raise OllamaError("Ollama request failed after retries")


__all__ = ["ollama_chat", "OllamaError", "DEFAULT_MODEL"]
