from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv, find_dotenv
from langchain_core.language_models.chat_models import BaseChatModel

# Gemini (Google AI Studio)
from langchain_google_genai import ChatGoogleGenerativeAI

# Ollama (local LLM)
from langchain_community.chat_models import ChatOllama


# ---------------------------------------------------------
# Load .env reliably (repo-root aware)
# ---------------------------------------------------------
def _load_env() -> None:
    """
    Make .env loading robust even when running:
      - python -c ...
      - notebooks
      - from a subfolder
    Priority:
      1) .env found by python-dotenv find_dotenv
      2) repo_root/.env (repo_root = 3 levels up from this file)
    """
    # 1) try find_dotenv (walks up from CWD)
    env_path = find_dotenv(usecwd=True)

    if env_path:
        load_dotenv(env_path, override=True)
        return

    # 2) fallback: assume repo structure app/agents/llm_factory.py
    # repo_root = .../garv-amadeus-text2sql
    repo_root = Path(__file__).resolve().parents[2]  # agents -> app -> repo_root
    fallback_env = repo_root / ".env"
    if fallback_env.exists():
        load_dotenv(fallback_env, override=True)


_load_env()


# ---------------------------------------------------------
# Public factory
# ---------------------------------------------------------
def get_llm(
    temperature: float = 0.0,
    model_override: Optional[str] = None,
) -> BaseChatModel:
    """
    Central LLM factory.

    Controlled via env:
      LLM_PROVIDER=gemini | ollama

    Returns:
      LangChain ChatModel
    """
    provider = os.getenv("LLM_PROVIDER", "ollama").strip().lower()

    if provider == "gemini":
        return _get_gemini_llm(temperature=temperature, model_override=model_override)

    if provider == "ollama":
        return _get_ollama_llm(temperature=temperature, model_override=model_override)

    raise ValueError(f"Unsupported LLM_PROVIDER='{provider}'. Supported: gemini, ollama")


# ---------------------------------------------------------
# Gemini (Google AI Studio)
# ---------------------------------------------------------
def _get_gemini_llm(
    temperature: float,
    model_override: Optional[str] = None,
) -> ChatGoogleGenerativeAI:
    """
    Gemini via Google AI Studio (Google AI API key).

    Required env:
      GEMINI_API_KEY or GOOGLE_API_KEY
      GEMINI_MODEL (optional)
    """
    api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()
    if not api_key:
        # Helpful debug info
        cwd = os.getcwd()
        raise ValueError(
            "GEMINI_API_KEY/GOOGLE_API_KEY not found in environment.\n"
            f"- Current working directory: {cwd}\n"
            "- Ensure .env is in repo root OR export GEMINI_API_KEY/GOOGLE_API_KEY.\n"
            "- In .env use: GEMINI_API_KEY=... (quotes optional)\n"
        )

    model = (model_override or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True,
    )


# ---------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------
def _get_ollama_llm(
    temperature: float,
    model_override: Optional[str] = None,
) -> ChatOllama:
    base_url = (os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434").strip()
    model = (model_override or os.getenv("OLLAMA_MODEL") or "qwen2.5:7b").strip()

    return ChatOllama(
        base_url=base_url,
        model=model,
        temperature=temperature,
    )
