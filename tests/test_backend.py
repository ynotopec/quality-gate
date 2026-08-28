"""Tests for quality_gate.backend module."""

from quality_gate.backend import LLMBackend, create_backend, _extract_json


class TestExtractJSON:
    def test_plain_json(self):
        text = '{"key": "value"}'
        assert _extract_json(text) == text

    def test_markdown_json(self):
        text = '```json\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_markdown_json_no_lang(self):
        text = '```\n{"key": "value"}\n```'
        assert _extract_json(text) == '{"key": "value"}'

    def test_json_with_extra_text(self):
        text = 'Here is the result:\n```json\n{"key": "value"}\n```'
        # _extract_json removes the code block wrapper
        result = _extract_json(text)
        assert result == '{"key": "value"}'


class TestCreateBackend:
    def test_create_with_api_key(self):
        """Factory creates backend with explicit config."""
        from quality_gate.backend import create_backend
        backend = create_backend(config={
            "openai_api_key": "test-key",
            "primary_model": "gpt-4o-mini",
        })
        assert backend.model == "gpt-4o-mini"
        assert backend.client.api_key == "test-key"

    def test_create_without_api_key_raises(self):
        """Factory raises ValueError without API key."""
        import os
        old_key = os.environ.get("OPENAI_API_KEY")
        try:
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]
            try:
                create_backend()
                assert False, "Should have raised ValueError"
            except ValueError:
                pass
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
