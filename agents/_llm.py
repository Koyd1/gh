from __future__ import annotations

import os
from typing import Any, Dict, Iterable, Optional

import requests

try:
    import streamlit as st
except ImportError:  # pragma: no cover - Streamlit not always installed
    st = None  # type: ignore

try:
    import google.generativeai as genai
except ImportError:  # pragma: no cover - optional dependency
    genai = None


class LLMError(RuntimeError):
    """Raised when the LLM API returns an error."""


def _get_setting(key: str, default: Optional[str] = None) -> Optional[str]:
    """Pull setting from env or Streamlit secrets."""

    value = os.getenv(key)
    if value:
        return value
    if st is not None:
        try:
            secret_value = st.secrets[key]
        except Exception:  # pragma: no cover - secrets missing/locked
            secret_value = None
        if secret_value:
            return str(secret_value)
    return default


def _resolve_backend() -> str:
    backend = _get_setting("LLM_BACKEND")
    if backend:
        return backend.lower()

    # Автоматически переключаемся на облачный бэкенд, если есть ключи.
    if _get_setting("GEMINI_API_KEY"):
        return "gemini"
    if _get_setting("OPENAI_API_KEY"):
        return "openai"
    return "ollama"


def _resolve_default_model() -> str:
    return _get_setting("OLLAMA_MODEL", "llama3.1:8b-instruct") or "llama3.1:8b-instruct"


def _resolve_base_url() -> str:
    return _get_setting("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"


def _resolve_timeout() -> float:
    timeout_setting = _get_setting("OLLAMA_TIMEOUT")
    if timeout_setting:
        try:
            return float(timeout_setting)
        except ValueError:
            pass
    return 180.0


DEFAULT_MODEL = _resolve_default_model()
DEFAULT_BASE_URL = _resolve_base_url()
DEFAULT_TIMEOUT = _resolve_timeout()


# 🔹 Ollama (локально)
def ollama_chat(
    messages: Iterable[Dict[str, str]], *, model: Optional[str] = None, temperature: float = 0.2
) -> str:
    model_name = model or _get_setting("OLLAMA_MODEL", DEFAULT_MODEL) or DEFAULT_MODEL
    base_url = (_get_setting("OLLAMA_BASE_URL", DEFAULT_BASE_URL) or DEFAULT_BASE_URL).rstrip("/")
    payload: Dict[str, Any] = {
        "model": model_name,
        "messages": list(messages),
        "stream": False,
        "options": {"temperature": temperature},
    }
    try:
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=DEFAULT_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
        content = (data.get("message") or {}).get("content", "")
        if not content:
            raise ValueError("empty content")
        return content.strip()
    except Exception as exc:  # pragma: no cover - network failure
        raise LLMError(f"Ollama request failed: {exc}") from exc


# 🔹 OpenAI (в облаке)
def openai_chat(
    messages: Iterable[Dict[str, str]], *, model: Optional[str] = None, temperature: float = 0.2
) -> str:
    try:
        from openai import OpenAI  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise LLMError("openai package is not installed") from exc

    api_key = _get_setting("OPENAI_API_KEY")
    if not api_key:
        raise LLMError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model or _get_setting("OPENAI_MODEL", "gpt-4o-mini") or "gpt-4o-mini",
        messages=list(messages),
        temperature=temperature,
    )
    message = resp.choices[0].message.content
    if not message:
        raise LLMError("OpenAI returned empty content")
    return message.strip()


# 🔹 Gemini (Google API)
def gemini_chat(
    messages: Iterable[Dict[str, str]], *, model: Optional[str] = None, temperature: float = 0.2
) -> str:
    if genai is None:
        raise LLMError("google-generativeai is not installed")

    api_key = _get_setting("GEMINI_API_KEY")
    if not api_key:
        raise LLMError("GEMINI_API_KEY is not set")

    genai.configure(api_key=api_key)
    model_name = model or _get_setting("GEMINI_MODEL", "gemini-1.5-flash") or "gemini-1.5-flash"

    # склеим user/system в один prompt (Gemini не понимает роли по OpenAI-стилю)
    prompt_parts = []
    for m in messages:
        role = m.get("role")
        content = m.get("content", "")
        if role == "system":
            prompt_parts.append(f"[SYSTEM]\n{content}\n")
        elif role == "user":
            prompt_parts.append(f"[USER]\n{content}\n")
        elif role == "assistant":
            prompt_parts.append(f"[ASSISTANT]\n{content}\n")

    response = genai.GenerativeModel(model_name).generate_content(
        "".join(prompt_parts),
        generation_config={"temperature": temperature},
    )
    text = getattr(response, "text", None)
    if not text:
        raise LLMError("Gemini returned empty content")
    return text.strip()


# 🔹 Универсальная точка входа
def chat(messages: Iterable[Dict[str, str]], *, model: Optional[str] = None, temperature: float = 0.2) -> str:
    backend = _resolve_backend()
    if backend == "ollama":
        return ollama_chat(messages, model=model, temperature=temperature)
    if backend == "openai":
        return openai_chat(messages, model=model, temperature=temperature)
    if backend == "gemini":
        return gemini_chat(messages, model=model, temperature=temperature)
    raise LLMError(f"Unknown LLM_BACKEND: {backend}")


__all__ = [
    "chat",
    "LLMError",
    "ollama_chat",
    "openai_chat",
    "gemini_chat",
    "DEFAULT_MODEL",
]
