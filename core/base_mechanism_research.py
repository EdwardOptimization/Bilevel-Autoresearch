"""Base class for Level 2 mechanism research.

Subclasses implement domain-specific methods:
- _get_explore_prompt(**kwargs): return (prompt_str, system_str) for Round 1
- _get_specify_prompt(selected_hypothesis, critique, **kwargs): return (prompt_str, system_str)
- _get_codegen_prompt(spec, reference_code, **kwargs): return prompt_str for Round 4
- _get_reference_code(**kwargs): return reference code string for codegen
- _parse_spec_metadata(spec, session_id): return domain-specific metadata tuple

The shared protocol (Explore -> Critique -> Specify -> Generate with retries) lives here.
Domain-specific prompts, apply, and validate logic stay in subclasses.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

from core.llm_client import LLMClient

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Shared prompt constants
# ---------------------------------------------------------------------------

CRITIQUE_SYSTEM = """You are a rigorous critic. Your job is to find failure modes and
implementation traps in proposed mechanism changes. Be specific and honest."""

CRITIQUE_PROMPT = """\
## Hypotheses to Critique

{exploration}

---

For each hypothesis, provide:
1. **Most likely failure mode** — how could this make things worse?
2. **Implementation trap** — what is the hardest part to get right in code?
3. **Evidence from trace** — does the failure trace actually support this hypothesis?
4. **Score**: impact (1–5) × feasibility (1–5) ÷ complexity (1–5)  →  final float

End with: **Selected**: [hypothesis number] — one sentence why.
"""

CODEGEN_SYSTEM = """You are an expert Python developer. Write clean, correct Python code.
Return ONLY the Python source — no markdown fences, no explanation outside comments."""

FIX_PROMPT = """\
The code you generated failed with this error:

```
{error}
```

Here is the code that failed:

```python
{code}
```

Fix the error. Return ONLY the corrected Python code (no fences, no explanation).
"""


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class BaseMechanismResearcher(ABC):
    """
    Shared protocol for Level-2 mechanism research sessions.

    Subclasses supply domain-specific prompts and spec-parsing via abstract
    methods.  The 4-round session loop (Explore, Critique, Specify,
    Generate-with-retries) is handled here.

    Args:
        model:            LLM model identifier (e.g. "deepseek-chat")
        provider:         Provider name (e.g. "deepseek")
        api_key:          API key for the provider
        max_code_retries: How many times to retry code generation on error
        artifacts_base:   Root directory for session artifact files
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        provider: str = "deepseek",
        api_key: str = "",
        max_code_retries: int = 3,
        artifacts_base: Path | None = None,
    ):
        self.client = LLMClient(provider, api_key, model)
        self.max_code_retries = max_code_retries
        self.artifacts_base = (
            Path(artifacts_base) if artifacts_base else Path("artifacts/mechanism_research")
        )

    # ------------------------------------------------------------------
    # Abstract interface — subclasses must implement
    # ------------------------------------------------------------------

    @abstractmethod
    def _get_explore_prompt(self, **kwargs) -> tuple[str, str]:
        """Return (prompt, system) for Round 1: Exploration.

        kwargs contains whatever domain-specific context the subclass needs
        (e.g. trace_summary, runner_code, article_id, ...).
        """
        ...

    @abstractmethod
    def _get_specify_prompt(
        self,
        selected_hypothesis: str,
        critique: str,
        **kwargs,
    ) -> tuple[str, str]:
        """Return (prompt, system) for Round 3: Specification."""
        ...

    @abstractmethod
    def _get_codegen_prompt(self, spec: str, reference_code: str, **kwargs) -> str:
        """Return the prompt string for Round 4: Code generation.

        The system prompt is always CODEGEN_SYSTEM.
        """
        ...

    @abstractmethod
    def _get_reference_code(self, **kwargs) -> str:
        """Return reference code to pass to _get_codegen_prompt."""
        ...

    @abstractmethod
    def _parse_spec_metadata(self, spec: str, session_id: str) -> tuple:
        """Parse implementation metadata from the spec text.

        Returns a domain-specific tuple, e.g.:
          - article domain: (inject_after, stage_name)
          - train domain:   (mechanism_name, impl_strategy, target)
        """
        ...

    # ------------------------------------------------------------------
    # Shared session protocol
    # ------------------------------------------------------------------

    def _run_session(
        self,
        session_dir: Path,
        session_id: str,
        log_prefix: str,
        explore_kwargs: dict,
        specify_kwargs: dict,
        codegen_kwargs: dict,
    ) -> dict:
        """
        Execute the 4-round research session.

        Returns a dict with keys:
          exploration, critique, spec, code, retries,
          spec_metadata, selected_hypothesis

        where spec_metadata is whatever _parse_spec_metadata() returns.
        Callers (subclass research() methods) use this dict to build their
        domain-specific result objects.
        """
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)

        # --- Round 1: Explore ---
        logger.info(f"[{log_prefix} {session_id}] Round 1: Exploration")
        explore_prompt, explore_system = self._get_explore_prompt(**explore_kwargs)
        exploration = self.client.call(
            explore_prompt,
            system=explore_system,
            max_tokens=4000,
        )
        (session_dir / "01_exploration.md").write_text(exploration, encoding="utf-8")
        logger.info(
            f"[{log_prefix} {session_id}] Exploration complete ({len(exploration)} chars)"
        )

        # --- Round 2: Critique ---
        logger.info(f"[{log_prefix} {session_id}] Round 2: Critique")
        critique = self.client.call(
            CRITIQUE_PROMPT.format(exploration=exploration),
            system=CRITIQUE_SYSTEM,
            max_tokens=3000,
        )
        (session_dir / "02_critique.md").write_text(critique, encoding="utf-8")
        logger.info(f"[{log_prefix} {session_id}] Critique complete")

        # --- Round 3: Specify ---
        logger.info(f"[{log_prefix} {session_id}] Round 3: Specification")
        selected_hypothesis = self._extract_selected(exploration, critique)
        specify_prompt, specify_system = self._get_specify_prompt(
            selected_hypothesis=selected_hypothesis,
            critique=critique,
            **specify_kwargs,
        )
        spec = self.client.call(
            specify_prompt,
            system=specify_system,
            max_tokens=2500,
        )
        (session_dir / "03_spec.md").write_text(spec, encoding="utf-8")
        logger.info(f"[{log_prefix} {session_id}] Specification complete")

        # Parse domain-specific metadata from spec
        spec_metadata = self._parse_spec_metadata(spec, session_id)

        # --- Round 4+: Generate code with retries ---
        logger.info(f"[{log_prefix} {session_id}] Round 4: Code generation")
        reference_code = self._get_reference_code(**codegen_kwargs)
        code, retries = self._generate_with_retries(
            spec=spec,
            reference_code=reference_code,
            session_dir=session_dir,
            codegen_kwargs=codegen_kwargs,
        )

        logger.info(
            f"[{log_prefix} {session_id}] Session complete. retries={retries}"
        )
        return {
            "exploration": exploration,
            "critique": critique,
            "spec": spec,
            "code": code,
            "retries": retries,
            "spec_metadata": spec_metadata,
            "selected_hypothesis": selected_hypothesis,
        }

    # ------------------------------------------------------------------
    # Shared code generation with retries
    # ------------------------------------------------------------------

    def _generate_with_retries(
        self,
        spec: str,
        reference_code: str,
        session_dir: Path,
        codegen_kwargs: dict,
    ) -> tuple[str, int]:
        """Generate code, retrying with error feedback on failure."""
        codegen_prompt = self._get_codegen_prompt(
            spec=spec,
            reference_code=reference_code,
            **codegen_kwargs,
        )
        code = self.client.call(
            codegen_prompt,
            system=CODEGEN_SYSTEM,
            max_tokens=6000,
        )
        code = self._strip_fences(code)

        for attempt in range(self.max_code_retries):
            (session_dir / f"04_code_attempt_{attempt + 1}.py").write_text(
                code, encoding="utf-8"
            )
            error = self._syntax_check(code)
            if error is None:
                logger.info(f"[CodeGen] Code OK on attempt {attempt + 1}")
                return code, attempt

            logger.warning(f"[CodeGen] Syntax error attempt {attempt + 1}: {error[:200]}")
            (session_dir / f"04_error_{attempt + 1}.txt").write_text(error, encoding="utf-8")

            code = self.client.call(
                FIX_PROMPT.format(error=error[:2000], code=code[:4000]),
                system=CODEGEN_SYSTEM,
                max_tokens=6000,
            )
            code = self._strip_fences(code)

        # Final attempt
        (session_dir / "04_code_final.py").write_text(code, encoding="utf-8")
        final_error = self._syntax_check(code)
        if final_error is not None:
            (session_dir / "04_code_final_error.txt").write_text(
                final_error, encoding="utf-8"
            )
            raise RuntimeError(
                f"Code generation failed after {self.max_code_retries} retries. "
                f"Last error: {final_error}"
            )
        return code, self.max_code_retries

    # ------------------------------------------------------------------
    # Shared utilities
    # ------------------------------------------------------------------

    def _extract_selected(self, exploration: str, critique: str) -> str:
        """Pull the selected hypothesis from critique output, fall back to exploration tail."""
        for line in critique.splitlines():
            if line.strip().lower().startswith("**selected**"):
                return line.strip()
        return exploration[-800:]

    def _syntax_check(self, code: str) -> str | None:
        """Return error string if code has syntax errors, else None."""
        try:
            compile(code, "<generated>", "exec")
            return None
        except SyntaxError as e:
            return f"SyntaxError: {e}"

    def _strip_fences(self, code: str) -> str:
        """Remove markdown code fences if the LLM added them anyway."""
        lines = code.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    def _extract_domain(self, exploration: str) -> str:
        """Best-effort extraction of the domain the LLM mentioned first."""
        for line in exploration.splitlines():
            low = line.lower()
            if "domain" in low or "field" in low or "inspired" in low:
                return line.strip()[:120]
        return exploration[:120]
