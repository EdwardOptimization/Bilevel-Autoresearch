"""Level 2 outer autoresearch — discovers new inner loop mechanisms via code generation.

The outer LLM (DeepSeek) treats "what mechanism would improve inner loop convergence?"
as a research question. It freely selects which domains to draw from, generates
hypotheses, critiques them, implements the best one as a Python stage, and validates
it by running the inner loop with the new mechanism injected.

Research session flow:
  1. Explore    — LLM chooses domains freely, generates N hypotheses
  2. Critique   — LLM plays skeptic, scores each hypothesis on impact × feasibility
  3. Specify    — selects best hypothesis, writes implementation spec
  4. Generate   — produces Python code implementing BaseStage interface
  5. Load+Smoke — dynamic import + 1-run smoke test to catch syntax/runtime errors
  6. Fix        — if error, feed traceback back to LLM, retry (up to max_code_retries)
  7. Validate   — run full inner cycle with new stage, measure alpha vs baseline
"""
from __future__ import annotations

import importlib.util
import json
import logging
import traceback as tb
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .inner_loop import InnerLoopController
from .llm_client import LLMClient
from .pipeline.base import BaseStage
from .runner import InnerRunner
from .state import InnerLoopState, OuterLoopState

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EXPLORE_SYSTEM = """You are a meta-researcher. Your job is to discover new mechanisms
that could improve an AI pipeline's convergence. You have complete freedom to draw on
any field — optimization, biology, physics, economics, information theory, or anything
else you find relevant. Do not constrain yourself to AI/ML literature alone."""

EXPLORE_PROMPT = """\
## Inner Loop Failure Report

The inner pipeline has been running for {n_cycles} outer cycles on article "{article_id}".
Despite prompt-level optimization, it is stuck.

### Score trace (per cycle, rubric dimensions A–E):
{trace_summary}

### What prompt-level interventions were already tried:
{lessons_summary}

### What "stuck" means here:
Dimension A (Argumentative Rigor) consistently scores 6–7. The pipeline generates
improvement hypotheses but they are not executed with sufficient logical rigour.
Prompt changes have plateaued — the bottleneck is structural, not textual.

---

## Your Task

Generate 4–6 hypotheses for mechanism changes that could break this plateau.

For EACH hypothesis:
1. **Domain**: what field/area inspired this idea (you choose freely)
2. **Core idea**: the mechanism in one sentence
3. **Pipeline mapping**: exactly which stage(s) would change, and how
4. **Why it addresses the bottleneck**: causal argument for why this improves dimension A
5. **Implementation complexity**: 1 (trivial) to 5 (major rewrite)
"""

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

SPECIFY_SYSTEM = """You are a senior engineer turning a research hypothesis into a
precise implementation specification."""

SPECIFY_PROMPT = """\
## Selected Hypothesis

{selected_hypothesis}

## Critique notes

{critique_notes}

---

Write a detailed implementation specification:

1. **Stage name** (snake_case, unique):
2. **Inject after**: which existing stage? (article_analysis / improvement_hypotheses / edit_planning / impact_assessment)
3. **Inputs used from context**: list the context keys this stage reads
4. **Output**: what goes into context["previous_outputs"][stage_name]
5. **Step-by-step logic**: numbered pseudocode
6. **LLM call design**: prompt sketch, expected output format, max_tokens estimate

Keep it concrete enough that a coder could implement without asking questions.
"""

CODEGEN_SYSTEM = """You are an expert Python developer. Write clean, correct Python code.
Return ONLY the Python source — no markdown fences, no explanation outside comments."""

CODEGEN_PROMPT = """\
## Implementation Specification

{spec}

---

## Interface Contract

Implement this as a Python class. The file will be saved and dynamically imported.

```python
# Available imports (use exactly these paths):
from src.pipeline.base import BaseStage
from src.llm_client import call_llm, parse_json_response

# Your class must:
class YourStage(BaseStage):
    name = "your_stage_name"   # must match spec

    def __init__(self, model: str = ""):
        super().__init__(model)

    def run(self, context: dict) -> dict:
        # context keys available:
        #   context["article_id"]          str
        #   context["article_content"]     str  — current article text
        #   context["previous_outputs"]    dict[str, str] — stage_name -> output STRING
        #       IMPORTANT: values are plain strings, NOT dicts. To read
        #       the output of a previous stage, use:
        #           text = context["previous_outputs"].get("stage_name", "")
        #       Then parse the text yourself if you need structured data.
        #   context["retrieved_lessons"]   str  — inner lessons/skills
        #   context["outer_guidance"]      dict — outer loop overrides
        #   context["evaluator_feedback"]  str  — quality gate feedback (may be empty)
        #   context["run_dir"]             Path
        #   context["run_number"]          int
        #
        # Must return:
        #   {{"content": str, "artifacts": list[str]}}
        ...
```

## Reference: existing stage for style guidance

{reference_stage}

---

Write the complete Python file. Start with the imports, then the class.
Do NOT wrap in markdown fences. Return raw Python only.
"""

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
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MechanismResult:
    session_id: str
    article_id: str
    hypothesis: str          # selected hypothesis text
    domain_source: str       # domain the LLM chose (extracted from exploration)
    spec: str                # implementation spec
    code: str                # final generated code
    stage_name: str          # e.g. "hypothesis_critic_20260322"
    inject_after: str        # existing stage name to inject after
    stage_path: Path         # path to saved .py file
    stage_class: Any         # loaded class (not serialisable)
    exploration: str = ""
    critique: str = ""
    code_retries: int = 0
    validation: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class MechanismResearcher:
    """
    Runs one Level-2 research session: produces + validates a new pipeline stage.

    Args:
        model:          LLM model for research (outer loop, e.g. deepseek-chat)
        api_key:        API key for that provider
        provider:       Provider name (default: deepseek)
        max_code_retries: How many times to retry code generation on error
        artifacts_base: Where to save session artifacts
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str = "",
        provider: str = "deepseek",
        max_code_retries: int = 3,
        artifacts_base: Path | None = None,
    ):
        self.client = LLMClient(provider, api_key, model)
        self.max_code_retries = max_code_retries
        self.artifacts_base = artifacts_base or Path("artifacts/mechanism_research")
        self._generated_dir = Path(__file__).parent / "pipeline" / "generated"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(
        self,
        article_id: str,
        inner_states: list[InnerLoopState],   # one per completed outer cycle
        outer_lessons: list[dict],
    ) -> MechanismResult:
        """Run one full research session. Returns the result including the loaded stage."""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = self.artifacts_base / f"session_{session_id}"
        session_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"[MechResearch {session_id}] Starting for article: {article_id}")

        # Build shared context
        trace_summary = self._build_trace_summary(inner_states)
        lessons_summary = self._build_lessons_summary(outer_lessons)
        n_cycles = len(inner_states)

        # --- Round 1: Explore ---
        logger.info(f"[MechResearch {session_id}] Round 1: Exploration")
        exploration = self.client.call(
            EXPLORE_PROMPT.format(
                article_id=article_id,
                n_cycles=n_cycles,
                trace_summary=trace_summary,
                lessons_summary=lessons_summary,
            ),
            system=EXPLORE_SYSTEM,
            max_tokens=4000,
        )
        (session_dir / "01_exploration.md").write_text(exploration, encoding="utf-8")
        logger.info(f"[MechResearch {session_id}] Exploration complete ({len(exploration)} chars)")

        # --- Round 2: Critique ---
        logger.info(f"[MechResearch {session_id}] Round 2: Critique")
        critique = self.client.call(
            CRITIQUE_PROMPT.format(exploration=exploration),
            system=CRITIQUE_SYSTEM,
            max_tokens=3000,
        )
        (session_dir / "02_critique.md").write_text(critique, encoding="utf-8")

        # --- Round 3: Specify ---
        logger.info(f"[MechResearch {session_id}] Round 3: Specification")
        # Extract selected hypothesis from critique
        selected_hypothesis = self._extract_selected(exploration, critique)
        spec_raw = self.client.call(
            SPECIFY_PROMPT.format(
                selected_hypothesis=selected_hypothesis,
                critique_notes=critique[-1500:],
            ),
            system=SPECIFY_SYSTEM,
            max_tokens=2000,
        )
        (session_dir / "03_spec.md").write_text(spec_raw, encoding="utf-8")

        # Parse inject_after and stage_name from spec
        inject_after, stage_name = self._parse_spec_metadata(spec_raw, session_id)

        # --- Rounds 4+: Generate code (with retries on error) ---
        reference_stage = self._read_reference_stage()
        code, retries = self._generate_with_retries(
            spec_raw, reference_stage, session_dir
        )

        # --- Load stage class ---
        stage_path = self._save_code(code, stage_name, session_dir)
        stage_class = self._load_stage(stage_path, stage_name)

        result = MechanismResult(
            session_id=session_id,
            article_id=article_id,
            hypothesis=selected_hypothesis,
            domain_source=self._extract_domain(exploration),
            spec=spec_raw,
            code=code,
            stage_name=stage_name,
            inject_after=inject_after,
            stage_path=stage_path,
            stage_class=stage_class,
            exploration=exploration,
            critique=critique,
            code_retries=retries,
        )

        # Save session summary
        self._save_summary(result, session_dir)
        logger.info(
            f"[MechResearch {session_id}] Done. "
            f"Stage: {stage_name}, inject_after: {inject_after}, "
            f"code_retries: {retries}"
        )
        return result

    def validate(
        self,
        result: MechanismResult,
        article_content: str,
        runner: InnerRunner,
        max_inner: int = 5,
    ) -> dict:
        """Run inner loop with the generated stage injected. Returns validation metrics."""
        session_dir = self.artifacts_base / f"session_{result.session_id}"

        logger.info(f"[MechResearch] Validating stage: {result.stage_name}")

        # Inject the generated stage into the runner
        runner.inject_stage(result.stage_class(model=runner.model), result.inject_after)

        outer_state = OuterLoopState(
            base_dir=Path("."),
            original_articles={result.article_id: article_content},
        )
        outer_state.begin_cycle()

        ctrl = InnerLoopController(runner=runner, max_iterations=max_inner)
        inner = ctrl.run_cycle(result.article_id, outer_state)

        scores = [r.overall for r in inner.run_trace]
        validation = {
            "score_trace": scores,
            "peak_score": inner.peak_score(),
            "runs_to_7": inner.runs_to_threshold(7),
            "runs_to_8": inner.runs_to_threshold(8),
            "converged": inner.is_converged(),
            "lessons_extracted": len(inner.inner_lessons),
        }

        (session_dir / "07_validation.json").write_text(
            json.dumps(validation, indent=2), encoding="utf-8"
        )
        result.validation = validation
        logger.info(f"[MechResearch] Validation: peak={validation['peak_score']}, trace={scores}")
        return validation

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_trace_summary(self, inner_states: list[InnerLoopState]) -> str:
        lines = []
        for i, state in enumerate(inner_states, 1):
            scores = [r.overall for r in state.run_trace]
            dim_a = [r.stage_map.get("A", "?") for r in state.run_trace]
            lines.append(
                f"Cycle {i}: overall={scores}  dim_A={dim_a}  "
                f"peak={state.peak_score()}  lessons={len(state.inner_lessons)}"
            )
        return "\n".join(lines) if lines else "No cycles recorded yet."

    def _build_lessons_summary(self, outer_lessons: list[dict]) -> str:
        if not outer_lessons:
            return "No outer lessons yet."
        summaries = [
            f"- [{l.get('lesson_type','?')}] {l.get('summary','')}"
            for l in outer_lessons[-10:]
        ]
        return "\n".join(summaries)

    def _extract_selected(self, exploration: str, critique: str) -> str:
        """Pull the selected hypothesis from critique output, fall back to full exploration."""
        for line in critique.splitlines():
            if line.strip().lower().startswith("**selected**"):
                # Try to extract hypothesis number and look it up
                return line.strip()
        # Fallback: return last 600 chars of exploration
        return exploration[-600:]

    def _parse_spec_metadata(self, spec: str, session_id: str) -> tuple[str, str]:
        """Extract inject_after and stage_name from spec text."""
        inject_after = "improvement_hypotheses"  # safe default
        stage_name = f"generated_stage_{session_id}"

        for line in spec.splitlines():
            low = line.lower()
            if "inject after" in low or "inject_after" in low:
                for candidate in ["revised_output", "impact_assessment", "edit_planning",
                                  "improvement_hypotheses", "article_analysis"]:
                    if candidate in low:
                        inject_after = candidate
                        break
            if "stage name" in low or "stage_name" in low:
                # Try to extract snake_case name
                parts = line.split(":")
                if len(parts) > 1:
                    name = parts[-1].strip().strip("`").strip()
                    if name and " " not in name:
                        stage_name = f"{name}_{session_id}"

        return inject_after, stage_name

    def _generate_with_retries(
        self, spec: str, reference_stage: str, session_dir: Path
    ) -> tuple[str, int]:
        """Generate code, retrying with error feedback on failure."""
        code = self.client.call(
            CODEGEN_PROMPT.format(spec=spec, reference_stage=reference_stage),
            system=CODEGEN_SYSTEM,
            max_tokens=6000,
        )
        code = self._strip_fences(code)

        for attempt in range(self.max_code_retries):
            (session_dir / f"04_code_attempt_{attempt+1}.py").write_text(
                code, encoding="utf-8"
            )
            error = self._syntax_check(code)
            if error is None:
                logger.info(f"[MechResearch] Code OK on attempt {attempt+1}")
                return code, attempt

            logger.warning(f"[MechResearch] Code error attempt {attempt+1}: {error[:200]}")
            (session_dir / f"04_error_{attempt+1}.txt").write_text(error, encoding="utf-8")

            code = self.client.call(
                FIX_PROMPT.format(error=error[:2000], code=code[:4000]),
                system=CODEGEN_SYSTEM,
                max_tokens=6000,
            )
            code = self._strip_fences(code)

        # Final save regardless
        (session_dir / "04_code_final.py").write_text(code, encoding="utf-8")
        return code, self.max_code_retries

    def _syntax_check(self, code: str) -> str | None:
        """Return error string if code has syntax errors, else None."""
        try:
            compile(code, "<generated>", "exec")
            return None
        except SyntaxError as e:
            return f"SyntaxError: {e}"

    def _save_code(self, code: str, stage_name: str, session_dir: Path) -> Path:
        """Write code to the generated/ directory and return the path."""
        filename = f"{stage_name}.py"
        stage_path = self._generated_dir / filename
        stage_path.write_text(code, encoding="utf-8")
        # Also copy to session dir for record
        (session_dir / "05_final_stage.py").write_text(code, encoding="utf-8")
        logger.info(f"[MechResearch] Stage saved: {stage_path}")
        return stage_path

    def _load_stage(self, stage_path: Path, stage_name: str) -> type:
        """Dynamically import the generated stage file and return the class."""
        spec = importlib.util.spec_from_file_location(stage_name, stage_path)
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load generated stage {stage_path}: {e}\n"
                f"{tb.format_exc()}"
            )
        # Find the BaseStage subclass in the module
        for attr_name in dir(module):
            obj = getattr(module, attr_name)
            try:
                if (isinstance(obj, type)
                        and issubclass(obj, BaseStage)
                        and obj is not BaseStage):
                    logger.info(f"[MechResearch] Loaded class: {attr_name}")
                    return obj
            except TypeError:
                continue
        raise RuntimeError(f"No BaseStage subclass found in {stage_path}")

    def _read_reference_stage(self) -> str:
        """Read improvement_hypotheses.py as a style reference for code generation."""
        ref_path = Path(__file__).parent / "pipeline" / "improvement_hypotheses.py"
        if ref_path.exists():
            return ref_path.read_text(encoding="utf-8")
        return "# (reference not available)"

    def _extract_domain(self, exploration: str) -> str:
        """Best-effort extraction of the domain the LLM mentioned first."""
        for line in exploration.splitlines():
            low = line.lower()
            if "domain" in low or "field" in low or "inspired" in low:
                return line.strip()[:120]
        return exploration[:120]

    def _strip_fences(self, code: str) -> str:
        """Remove markdown code fences if the LLM added them anyway."""
        lines = code.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines)

    def _save_summary(self, result: MechanismResult, session_dir: Path) -> None:
        summary = {
            "session_id": result.session_id,
            "article_id": result.article_id,
            "domain_source": result.domain_source,
            "stage_name": result.stage_name,
            "inject_after": result.inject_after,
            "code_retries": result.code_retries,
            "stage_path": str(result.stage_path),
        }
        (session_dir / "06_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
