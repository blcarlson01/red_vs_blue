"""
llm_client.py

Compatibility layer for analysis scripts:
- Uses inspect_ai.model.get_model when available.
- Falls back to OpenAI-compatible chat completions (e.g., Ollama /v1)
  when inspect_ai is not installed.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any


def _resolve_model_name(model_name: str) -> str:
    """Convert provider/model style names to plain model ID for OpenAI clients."""
    if "/" in model_name:
        return model_name.split("/", 1)[1]
    return model_name


def _resolve_base_url() -> str | None:
    return (
        os.environ.get("INSPECT_EVAL_MODEL_BASE_URL")
        or os.environ.get("MODEL_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
    )


@dataclass
class _FallbackResponse:
    completion: str


class _FallbackModel:
    def __init__(self, model_name: str, base_url: str | None = None):
        from openai import OpenAI

        self.model_name = _resolve_model_name(model_name)
        self.base_url = base_url or _resolve_base_url()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            api_key = "ollama"
        self.client = OpenAI(base_url=self.base_url, api_key=api_key)

    async def generate(self, input: str, config: dict[str, Any] | None = None):
        config = config or {}
        max_tokens = int(config.get("max_tokens", 1000))
        temperature = float(config.get("temperature", 0.3))

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": input}],
            max_tokens=max_tokens,
            temperature=temperature,
        )

        content = ""
        if response.choices:
            message = response.choices[0].message
            content = message.content or ""

        return _FallbackResponse(completion=content)


def get_model(model_name: str):
    """Get model instance with inspect_ai primary and OpenAI-compatible fallback."""
    try:
        from inspect_ai.model import get_model as inspect_get_model

        return inspect_get_model(model_name)
    except Exception:
        return _FallbackModel(model_name)


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Extract the first valid JSON object from arbitrary model text."""
    if not text:
        return None

    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False

        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break

        start = text.find("{", start + 1)

    return None


async def generate_json_with_retries(
    model: Any,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float = 0.3,
    retries: int = 3,
    error_context: str = "analysis",
) -> dict[str, Any] | None:
    """Generate structured JSON with retry + parsing fallback."""
    for attempt in range(retries):
        try:
            response = await model.generate(
                input=prompt,
                config={
                    "max_tokens": max_tokens,
                    "temperature": max(0.0, temperature - (0.1 * attempt)),
                },
            )
            text = response.completion if hasattr(response, "completion") else (
                response.choices[0].message.content if response.choices else ""
            )
            parsed = extract_json_object(text)
            if parsed is not None:
                return parsed
            print(f"Warning: Could not parse JSON for {error_context} (attempt {attempt + 1}/{retries})")
        except Exception as e:
            print(f"Error during {error_context} (attempt {attempt + 1}/{retries}): {e}")

    return None
