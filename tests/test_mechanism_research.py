"""Unit tests for core/base_mechanism_research.py — BaseMechanismResearcher."""
import pytest

from core.base_mechanism_research import BaseMechanismResearcher

# ---------------------------------------------------------------------------
# Concrete stub — satisfies all abstract methods so we can exercise base logic
# ---------------------------------------------------------------------------

class _ConcreteResearcher(BaseMechanismResearcher):
    def _get_explore_prompt(self, **kwargs):
        return ("explore prompt", "system")

    def _get_specify_prompt(self, selected_hypothesis, critique, **kwargs):
        return ("specify prompt", "system")

    def _get_codegen_prompt(self, spec, reference_code, **kwargs):
        return "codegen prompt"

    def _get_reference_code(self, **kwargs):
        return "# reference code"

    def _parse_spec_metadata(self, spec, session_id):
        return ("mechanism", "strategy", "target")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBaseMechanismResearcherAbstract:
    def test_cannot_instantiate_directly(self):
        """BaseMechanismResearcher is abstract and must not be instantiatable."""
        with pytest.raises(TypeError):
            BaseMechanismResearcher()  # type: ignore[abstract]

    def test_concrete_subclass_can_be_instantiated(self):
        """A fully-implemented subclass must instantiate without error."""
        researcher = _ConcreteResearcher(api_key="dummy")
        assert researcher is not None

    def test_missing_abstract_method_raises(self):
        """A subclass missing one abstract method must also fail to instantiate."""
        class _Incomplete(BaseMechanismResearcher):
            def _get_explore_prompt(self, **kwargs):
                return ("", "")
            # _get_specify_prompt, _get_codegen_prompt, etc. intentionally omitted

        with pytest.raises(TypeError):
            _Incomplete()  # type: ignore[abstract]


class TestSyntaxCheck:
    def setup_method(self):
        self.researcher = _ConcreteResearcher(api_key="dummy")

    def test_valid_code_returns_none(self):
        code = "x = 1\nprint(x)\n"
        assert self.researcher._syntax_check(code) is None

    def test_valid_function_returns_none(self):
        code = "def foo(a, b):\n    return a + b\n"
        assert self.researcher._syntax_check(code) is None

    def test_invalid_syntax_returns_error_string(self):
        code = "def broken(\n"
        result = self.researcher._syntax_check(code)
        assert result is not None
        assert isinstance(result, str)
        assert "SyntaxError" in result

    def test_invalid_expression_returns_error_string(self):
        code = "x = (1 +"
        result = self.researcher._syntax_check(code)
        assert result is not None
        assert "SyntaxError" in result

    def test_empty_string_is_valid(self):
        assert self.researcher._syntax_check("") is None

    def test_multiline_valid_class(self):
        code = "class Foo:\n    def bar(self):\n        pass\n"
        assert self.researcher._syntax_check(code) is None


class TestStripFences:
    def setup_method(self):
        self.researcher = _ConcreteResearcher(api_key="dummy")

    def test_removes_python_fence(self):
        code = "```python\nx = 1\n```"
        result = self.researcher._strip_fences(code)
        assert result == "x = 1"

    def test_removes_plain_fence(self):
        code = "```\nx = 1\n```"
        result = self.researcher._strip_fences(code)
        assert result == "x = 1"

    def test_no_fences_unchanged(self):
        code = "x = 1\ny = 2"
        result = self.researcher._strip_fences(code)
        assert result == code

    def test_only_opening_fence_removed(self):
        # No closing fence — only leading fence stripped
        code = "```python\nx = 1\ny = 2"
        result = self.researcher._strip_fences(code)
        assert result == "x = 1\ny = 2"

    def test_multiline_body_preserved(self):
        code = "```python\ndef foo():\n    return 42\n```"
        result = self.researcher._strip_fences(code)
        assert result == "def foo():\n    return 42"

    def test_empty_string_unchanged(self):
        assert self.researcher._strip_fences("") == ""

    def test_fence_with_extra_tag_stripped(self):
        code = "```py\npass\n```"
        result = self.researcher._strip_fences(code)
        assert result == "pass"


class TestExtractSelected:
    def setup_method(self):
        self.researcher = _ConcreteResearcher(api_key="dummy")

    def test_extracts_selected_line_from_critique(self):
        exploration = "some exploration text"
        critique = (
            "## Critique\n"
            "1. Hypothesis A — could fail because X.\n"
            "2. Hypothesis B — feasible but risky.\n"
            "**Selected**: 2 — highest impact-to-complexity ratio.\n"
        )
        result = self.researcher._extract_selected(exploration, critique)
        assert "**Selected**" in result
        assert "2" in result

    def test_case_insensitive_selected_keyword(self):
        exploration = "exploration"
        critique = "**SELECTED**: 1 — best option.\n"
        result = self.researcher._extract_selected(exploration, critique)
        assert "**SELECTED**" in result

    def test_falls_back_to_exploration_tail_when_no_selected(self):
        # Critique has no **Selected** line
        exploration = "A" * 900  # longer than 800 chars
        critique = "No selection made here."
        result = self.researcher._extract_selected(exploration, critique)
        # Should return last 800 chars of exploration
        assert result == exploration[-800:]

    def test_falls_back_to_full_exploration_when_short(self):
        exploration = "short exploration"
        critique = "no selection"
        result = self.researcher._extract_selected(exploration, critique)
        assert result == exploration

    def test_selected_line_stripped_of_leading_whitespace(self):
        exploration = "exp"
        critique = "   **Selected**: 3 — most feasible.\n"
        result = self.researcher._extract_selected(exploration, critique)
        # _extract_selected does line.strip() before returning
        assert result.startswith("**Selected**")
