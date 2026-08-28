"""Backend abstraction for LLM calls (OpenAI-compatible)."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

import openai
from openai import OpenAI


class LLMBackend:
    """Abstraction over OpenAI-compatible APIs."""

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "gpt-4o",
    ):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, system: str, prompt: str, temperature: float = 0.1) -> str:
        """Generate text from system + user prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=4096,
        )
        return response.choices[0].message.content or ""

    def generate_json(self, system: str, prompt: str) -> dict:
        """Generate JSON response. Raises ValueError if not valid JSON."""
        content = self.generate(
            system=system,
            prompt=prompt,
            temperature=0.1,
        )
        # Try to parse as JSON (might have markdown codeblocks)
        content = _extract_json(content)
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from LLM: {e}\nResponse: {content}")


def _extract_json(text: str) -> str:
    """Extract JSON from markdown code blocks or return as-is."""
    # Find JSON inside ```json ... ``` or ``` ... ``` anywhere in text
    match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def create_backend(provider: str = "openai", config=None) -> LLMBackend:
    """Factory for creating LLMBackend instances.

    Args:
        provider: "openai", "anthropic" (via OpenAI-compatible proxy), or
                  any OpenAI-compatible provider name.
        config: QualityConfig instance or dict with api_key, base_url, model.

    Returns:
        Configured LLMBackend instance.
    """
    if config is None:
        config = {}
    if isinstance(config, dict):
        api_key = config.get("openai_api_key", os.environ.get("OPENAI_API_KEY", ""))
        base_url = config.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")
        model = config.get("primary_model", "gpt-4o")
        if "anthropic_api_key" in config:
            os.environ["ANTHROPIC_API_KEY"] = config["anthropic_api_key"]
    else:
        api_key = config.openai_api_key or os.environ.get("OPENAI_API_KEY", "")
        base_url = config.openai_base_url or os.environ.get("OPENAI_BASE_URL")
        model = config.primary_model

    if not api_key:
        raise ValueError("API key not provided. Set OPENAI_API_KEY or pass api_key.")

    return LLMBackend(api_key=api_key, base_url=base_url, model=model)
