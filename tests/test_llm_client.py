"""Unit tests for the LLM client module."""
import pytest

from core.llm_client import (
    PROVIDERS,
    _strip_thinking_tags,
    configure,
    parse_json_response,
)


class TestStripThinkingTags:
    def test_strips_think_block(self):
        text = "<think>some reasoning here</think>Final answer."
        assert _strip_thinking_tags(text) == "Final answer."

    def test_strips_multiline_think_block(self):
        text = "<think>\nline1\nline2\n</think>Result"
        assert _strip_thinking_tags(text) == "Result"

    def test_strips_case_insensitive(self):
        text = "<THINK>reasoning</THINK>Output"
        assert _strip_thinking_tags(text) == "Output"

    def test_strips_incomplete_think_at_end(self):
        text = "Some output\n<think>cut off reasoning"
        result = _strip_thinking_tags(text)
        assert "<think>" not in result.lower()

    def test_no_think_tags_unchanged(self):
        text = "Normal response without think tags."
        assert _strip_thinking_tags(text) == text

    def test_empty_string(self):
        assert _strip_thinking_tags("") == ""

    def test_none_input(self):
        assert _strip_thinking_tags(None) is None

    def test_preserves_content_after_think(self):
        text = "<think>I need to think about this carefully.</think>The answer is 42."
        assert _strip_thinking_tags(text) == "The answer is 42."


class TestParseJsonResponse:
    def test_parse_plain_json_object(self):
        result = parse_json_response('{"score": 8, "verdict": "pass"}')
        assert result["score"] == 8
        assert result["verdict"] == "pass"

    def test_parse_plain_json_array(self):
        result = parse_json_response('[{"id": "h1"}, {"id": "h2"}]')
        assert isinstance(result, list)
        assert len(result) == 2

    def test_parse_markdown_fenced_json(self):
        text = '```json\n{"score": 7}\n```'
        result = parse_json_response(text)
        assert result["score"] == 7

    def test_parse_json_in_prose(self):
        text = 'Here is the result: {"verdict": "weak", "score": 5} based on analysis.'
        result = parse_json_response(text)
        assert result["verdict"] == "weak"

    def test_parse_array_in_prose(self):
        text = 'Result: ["item1", "item2", "item3"]'
        result = parse_json_response(text)
        assert isinstance(result, list)
        assert "item1" in result

    def test_empty_string_returns_raw_content(self):
        result = parse_json_response("")
        assert "raw_content" in result

    def test_unparseable_returns_raw_content(self):
        result = parse_json_response("This is not JSON at all.")
        assert "raw_content" in result

    def test_strips_think_tags_before_parsing(self):
        text = '<think>reasoning</think>{"score": 9}'
        result = parse_json_response(text)
        assert result["score"] == 9

    def test_nested_json_parsed_correctly(self):
        text = '{"stage": "A", "verdicts": {"h1": "pass", "h2": "fail"}}'
        result = parse_json_response(text)
        assert result["stage"] == "A"
        assert result["verdicts"]["h1"] == "pass"

    def test_fenced_without_language_tag(self):
        text = '```\n[{"id": "x"}]\n```'
        result = parse_json_response(text)
        assert isinstance(result, list)


class TestConfigure:
    def test_configure_known_provider(self):
        configure("deepseek", "test-key", "deepseek-chat")

    def test_configure_minimax(self):
        configure("minimax", "test-key")

    def test_configure_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            configure("nonexistent", "test-key")

    def test_all_providers_registered(self):
        for name in ["deepseek", "openai", "glm", "minimax", "anthropic"]:
            assert name in PROVIDERS
            assert "base_url" in PROVIDERS[name]
            assert "default_model" in PROVIDERS[name]
            assert "api_key_env" in PROVIDERS[name]
