"""Configuration for quality gate thresholds."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class QualityConfig:
    """Configuration for quality gate thresholds."""

    # Self-score thresholds
    self_score_threshold: float = 6.0  # min self-score on 0-10
    hallucination_rate_threshold: float = 0.15  # max 15% hallucination rate
    warning_hallucination_threshold: float = 0.08  # warning at 8%

    # Citation requirements
    min_citations: int = 2
    require_citations_for_factual: bool = True

    # Fallback behavior
    max_retries: int = 2  # max regeneration attempts
    fallback_responses: bool = True  # return "I don't know" on final failure

    # Model selection
    primary_model: str = "gpt-4o"
    fallback_model: str = "gpt-4o-mini"
    judge_model: str = "claude-sonnet-4-20250514"  # strong at evaluating

    # API configuration
    openai_api_key: str = os.environ.get("OPENAI_API_KEY", "")
    anthropic_api_key: str = os.environ.get("ANTHROPIC_API_KEY", "")
    openai_base_url: Optional[str] = None  # for OpenAI-compatible providers

    @classmethod
    def from_yaml(cls, path: str | Path) -> "QualityConfig":
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            return cls()
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_env(cls) -> "QualityConfig":
        """Load from environment variables."""
        return cls(
            openai_api_key=os.environ.get("OPENAI_API_KEY", ""),
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
            openai_base_url=os.environ.get("OPENAI_BASE_URL", None) or None,
        )

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            k: v
            for k, v in self.__dict__.items()
            if k not in ("openai_api_key", "anthropic_api_key", "openai_base_url")
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


DEFAULT_CONFIG = QualityConfig()
