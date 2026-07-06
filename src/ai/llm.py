"""LLM interface — abstract provider pattern for real model integration.

Designed to support any OpenAI-compatible API (OpenAI, Groq, Ollama,
LM Studio, Together, etc.) Swap providers by changing the config.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Literal

import httpx


LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1024"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))


@dataclass
class LLMConfig:
    provider: str = LLM_PROVIDER
    api_key: str = LLM_API_KEY
    base_url: str = LLM_BASE_URL
    model: str = LLM_MODEL
    max_tokens: int = LLM_MAX_TOKENS
    temperature: float = LLM_TEMPERATURE
    extra_headers: dict[str, str] = field(default_factory=dict)


default_config = LLMConfig()


FREE_PROVIDERS: dict[str, LLMConfig] = {
    "groq": LLMConfig(
        provider="groq",
        base_url="https://api.groq.com/openai/v1",
        model="llama-3.3-70b-versatile",
    ),
    "ollama": LLMConfig(
        provider="ollama",
        base_url="http://localhost:11434/v1",
        model="llama3.2",
        api_key="ollama",
    ),
    "openrouter": LLMConfig(
        provider="openrouter",
        base_url="https://openrouter.ai/api/v1",
        model="meta-llama/llama-3.3-70b-instruct",
    ),
    "together": LLMConfig(
        provider="together",
        base_url="https://api.together.xyz/v1",
        model="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
    ),
    "github": LLMConfig(
        provider="github",
        base_url="https://models.inference.ai.azure.com",
        model="gpt-4o-mini",
    ),
}


def _build_headers(config: LLMConfig | None = None) -> dict[str, str]:
    cfg = config or default_config
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {cfg.api_key}",
    }
    headers.update(cfg.extra_headers)
    return headers


def _build_body(
    messages: list[dict],
    config: LLMConfig | None = None,
) -> dict:
    cfg = config or default_config
    return {
        "model": cfg.model,
        "messages": messages,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "stream": False,
    }


async def complete(
    messages: list[dict],
    config: LLMConfig | None = None,
) -> str:
    """Send a chat completion request to the configured LLM provider.

    Parameters
    ----------
    messages : list[dict]
        Standard OpenAI chat messages format, e.g.
        ``[{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]``
    config : LLMConfig, optional
        Provider configuration. Falls back to env vars / defaults.

    Returns
    -------
    str
        The model's response text.
    """
    cfg = config or default_config
    headers = _build_headers(cfg)
    body = _build_body(messages, cfg)

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{cfg.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def complete_stream(
    messages: list[dict],
    config: LLMConfig | None = None,
):
    """Stream a chat completion from the configured LLM provider."""
    cfg = config or default_config
    headers = _build_headers(cfg)
    body = _build_body(messages, cfg)
    body["stream"] = True

    async with httpx.AsyncClient(timeout=120.0) as client:
        async with client.stream(
            "POST",
            f"{cfg.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=body,
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str == "[DONE]":
                        break
                    if data_str:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0].get("delta", {})
                        yield delta.get("content", "")


