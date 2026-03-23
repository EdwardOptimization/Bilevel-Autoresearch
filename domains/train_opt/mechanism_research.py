"""Level 2 outer autoresearch for the training domain — discovers new inner loop mechanisms
via code generation that modifies the TrainRunner class directly.

Unlike the article domain's mechanism_research.py (which injects BaseStage pipeline stages),
the training domain has no stage abstraction. Improvements are implemented as:
  - New helper classes added to runner.py (e.g. a new tracker or strategy)
  - Modified methods on TrainRunner (e.g. a new _propose or keep/discard logic)
  - New attributes on TrainRunner.__init__

Research session flow:
  1. Explore   — analyze inner loop trace, propose 3-4 mechanism improvements
  2. Critique  — find failure modes, score each hypothesis on impact × feasibility
  3. Specify   — write implementation spec for the selected hypothesis
  4. Generate  — produce a DROP-IN replacement for a specific method or a new helper class
  5. Apply     — string-replace the method or append the new class into runner.py
  6. Fix       — if syntax check fails, feed error back to LLM and retry
  7. Validate  — import-check the modified runner.py

"""
from __future__ import annotations

import ast
import json
import logging
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from core.base_mechanism_research import (
    CODEGEN_SYSTEM,
    FIX_PROMPT,
    BaseMechanismResearcher,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EXPLORE_SYSTEM = """You are a meta-researcher specializing in hyperparameter optimization
search algorithms. Your job is to propose concrete mechanism changes that could improve an
inner loop optimizer's ability to find good hyperparameter configurations efficiently.
Draw on any field — optimization theory, evolutionary algorithms, Bayesian optimization,
statistical physics, reinforcement learning, or anything else you find useful."""

EXPLORE_PROMPT = """\
## Inner Loop Trace Analysis

The training hyperparameter optimizer has been running with the following results:

### Iteration trace (most recent {n_iters} iterations):
{trace_summary}

### Current runner.py mechanisms already in place:
{runner_summary}

### Identified bottleneck / failure pattern:
{bottleneck}

---

## Your Task

Propose 3-4 mechanism improvements to the TrainRunner inner loop optimizer.
These improvements will be implemented as code changes to runner.py.

For EACH hypothesis:
1. **Domain**: what field/area inspired this idea
2. **Core idea**: the mechanism in one sentence
3. **Implementation target**: which method or class to add/modify in runner.py
   (e.g. "_propose", "run_iteration keep/discard logic", "new helper class ElitePool")
4. **Why it addresses the bottleneck**: causal argument
5. **Implementation complexity**: 1 (trivial) to 5 (major rewrite)
6. **Risk of regressions**: low / medium / high
"""

CRITIQUE_SYSTEM = """You are a rigorous critic reviewing proposed improvements to a
hyperparameter search algorithm. Find failure modes and implementation traps.
Be specific and honest — bad mechanisms waste GPU compute."""

CRITIQUE_PROMPT = """\
## Proposed Mechanism Improvements

{exploration}

---

For each hypothesis, provide:
1. **Most likely failure mode** — how could this make search WORSE?
2. **Implementation trap** — what is the hardest part to code correctly?
3. **Evidence from trace** — does the trace actually support needing this?
4. **Score**: impact (1–5) × feasibility (1–5) ÷ complexity (1–5)  →  final float

End with: **Selected**: [hypothesis number] — one sentence why.
"""

SPECIFY_SYSTEM = """You are a senior ML engineer turning a research hypothesis into a
precise implementation specification for modifying a Python class."""

SPECIFY_PROMPT = """\
## Selected Hypothesis

{selected_hypothesis}

## Critique notes

{critique_notes}

## Current runner.py code (relevant section)

```python
{runner_section}
```

---

Write a detailed implementation specification:

1. **Mechanism name** (snake_case, descriptive):
2. **Implementation strategy**: choose ONE of:
   - "new_method": add a new method to TrainRunner
   - "replace_method": replace an existing method of TrainRunner
   - "new_helper_class": add a new standalone helper class (like ElitePool, CrashMemory)
   - "modify_init": add new attributes to TrainRunner.__init__
3. **Target**: method name to replace, or class/attribute name to add
4. **Interface**: exact signature (inputs, outputs, types)
5. **Step-by-step logic**: numbered pseudocode
6. **Integration points**: how TrainRunner.__init__ and run_iteration/other methods call this

Keep it concrete enough that a coder could implement without asking questions.
"""

CODEGEN_PROMPT = """\
## Implementation Specification

{spec}

---

## Interface Contract

You are writing code that will be inserted into or used by runner.py in the training domain.

Available imports (already present in runner.py):
```python
import json, logging, math, random, re, shutil, subprocess, time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from core.llm_client import LLMClient, parse_json_response
```

The TrainRunner class has these key attributes:
- self.trace: TrainTrace — has .results (list[TrainResult]), .best_bpb, .best_iteration
- self.crash_memory: CrashMemory — call .record(changes, iter, error_hint) and .get_warning_text()
- self.momentum: MomentumTracker — call .record(...) and .get_momentum_text()
- self.elite_pool: ElitePool — call .add(...), .get_elite_text(), .generate_crossover(...)
- self.step_calibrator: StepSizeCalibrator — call .record(...) and .get_step_size_text()
- self.plateau_detector: PlateauDetector — call .record(...) and .check_plateau()
- self.client: LLMClient — call .call(prompt, system=..., max_tokens=...)
- self.search_config: SearchConfig — has .active_params, .frozen_params, .strategy, .guidance
- self.current_code: str — current working train.py content
- self._best_code: str|None — train.py of best config so far
- self.train_py, self.work_dir, self.artifacts_dir: Path

TrainResult dataclass fields: iteration, val_bpb, peak_vram_mb, training_seconds,
  num_params_m, status ("keep"|"discard"|"crash"), changes (dict), description (str)

## Reference: existing helper class for style guidance

```python
{reference_code}
```

---

## Your Task

{codegen_task}

Write ONLY the code that should be inserted (the new class or the replacement method).
Do NOT rewrite the entire runner.py. Do NOT wrap in markdown fences.
Return raw Python only.
"""

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TrainMechanismResult:
    session_id: str
    hypothesis: str
    mechanism_name: str
    implementation_strategy: str   # new_method | replace_method | new_helper_class | modify_init
    target: str                    # method/class/attribute name
    spec: str
    code: str                      # the generated code fragment
    exploration: str = ""
    critique: str = ""
    code_retries: int = 0
    applied: bool = False
    validated: bool = False
    validation_error: str = ""
    session_dir: Path = field(default_factory=lambda: Path("."))


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class TrainMechanismResearcher(BaseMechanismResearcher):
    """
    Runs one Level-2 research session for the training domain.

    Produces a code patch that modifies runner.py, applying it in-place and
    validating that the modified file still imports correctly.

    Args:
        model:            LLM model for research (e.g. deepseek-chat)
        api_key:          API key for the provider
        provider:         Provider name (default: deepseek)
        max_code_retries: How many times to retry code generation on error
    """

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: str = "",
        provider: str = "deepseek",
        max_code_retries: int = 3,
    ):
        super().__init__(
            model=model,
            provider=provider,
            api_key=api_key,
            max_code_retries=max_code_retries,
        )
        self._runner_path = Path(__file__).parent / "runner.py"

    # ------------------------------------------------------------------
    # Abstract method implementations (BaseMechanismResearcher interface)
    # ------------------------------------------------------------------

    def _get_explore_prompt(self, **kwargs) -> tuple[str, str]:
        return (
            EXPLORE_PROMPT.format(
                n_iters=kwargs["n_iters"],
                trace_summary=kwargs["trace_summary"],
                runner_summary=kwargs["runner_summary"],
                bottleneck=kwargs["bottleneck"],
            ),
            EXPLORE_SYSTEM,
        )

    def _get_specify_prompt(
        self,
        selected_hypothesis: str,
        critique: str,
        **kwargs,
    ) -> tuple[str, str]:
        return (
            SPECIFY_PROMPT.format(
                selected_hypothesis=selected_hypothesis,
                critique_notes=critique[-2000:],
                runner_section=kwargs.get("runner_section", "")[:3000],
            ),
            SPECIFY_SYSTEM,
        )

    def _get_codegen_prompt(self, spec: str, reference_code: str, **kwargs) -> str:
        return CODEGEN_PROMPT.format(
            spec=spec,
            reference_code=reference_code,
            codegen_task=kwargs.get("codegen_task", ""),
        )

    def _get_reference_code(self, **kwargs) -> str:
        runner_code = kwargs.get("runner_code", "")
        impl_strategy = kwargs.get("impl_strategy", "new_helper_class")
        return self._read_reference_code(runner_code, impl_strategy)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def research(
        self,
        trace_summary: str,
        runner_code: str,
        session_dir: Path,
        bottleneck: str = "",
    ) -> TrainMechanismResult:
        """
        Run one full research session.

        Args:
            trace_summary: Human-readable trace of inner loop results
                           (scores, keeps, discards, crashes, best_bpb progression)
            runner_code:   Full text of runner.py (or the relevant section)
            session_dir:   Where to save session artifacts
            bottleneck:    Optional description of the specific bottleneck to address

        Returns:
            TrainMechanismResult with generated code and metadata.
            The result is NOT yet applied — call apply() to modify runner.py.
        """
        session_dir = Path(session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)

        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        logger.info(f"[TrainMechResearch {session_id}] Starting research session")

        # Summarize what mechanisms are already in place
        runner_summary = self._summarize_existing_mechanisms(runner_code)
        n_iters = trace_summary.count("iter ") or trace_summary.count("\n")

        if not bottleneck:
            bottleneck = self._infer_bottleneck(trace_summary)

        # --- Round 1: Explore ---
        logger.info(f"[TrainMechResearch {session_id}] Round 1: Exploration")
        exploration = self.client.call(
            EXPLORE_PROMPT.format(
                n_iters=n_iters,
                trace_summary=trace_summary,
                runner_summary=runner_summary,
                bottleneck=bottleneck,
            ),
            system=EXPLORE_SYSTEM,
            max_tokens=4000,
        )
        (session_dir / "01_exploration.md").write_text(exploration, encoding="utf-8")
        logger.info(f"[TrainMechResearch {session_id}] Exploration: {len(exploration)} chars")

        # --- Round 2: Critique ---
        logger.info(f"[TrainMechResearch {session_id}] Round 2: Critique")
        critique = self.client.call(
            CRITIQUE_PROMPT.format(exploration=exploration),
            system=CRITIQUE_SYSTEM,
            max_tokens=3000,
        )
        (session_dir / "02_critique.md").write_text(critique, encoding="utf-8")
        logger.info(f"[TrainMechResearch {session_id}] Critique complete")

        # --- Round 3: Specify ---
        logger.info(f"[TrainMechResearch {session_id}] Round 3: Specification")
        selected_hypothesis = self._extract_selected(exploration, critique)
        runner_section = self._extract_runner_section(runner_code)

        spec = self.client.call(
            SPECIFY_PROMPT.format(
                selected_hypothesis=selected_hypothesis,
                critique_notes=critique[-2000:],
                runner_section=runner_section[:3000],
            ),
            system=SPECIFY_SYSTEM,
            max_tokens=2500,
        )
        (session_dir / "03_spec.md").write_text(spec, encoding="utf-8")
        logger.info(f"[TrainMechResearch {session_id}] Specification complete")

        # Parse mechanism metadata from spec
        mechanism_name, impl_strategy, target = self._parse_spec_metadata(spec, session_id)

        # --- Rounds 4+: Generate code with retries ---
        reference_code = self._read_reference_code(runner_code, impl_strategy)
        codegen_task = self._build_codegen_task(mechanism_name, impl_strategy, target, spec)

        code, retries = self._generate_with_retries(
            spec=spec,
            reference_code=reference_code,
            codegen_task=codegen_task,
            session_dir=session_dir,
        )

        result = TrainMechanismResult(
            session_id=session_id,
            hypothesis=selected_hypothesis,
            mechanism_name=mechanism_name,
            implementation_strategy=impl_strategy,
            target=target,
            spec=spec,
            code=code,
            exploration=exploration,
            critique=critique,
            code_retries=retries,
            session_dir=session_dir,
        )

        self._save_summary(result, session_dir)
        logger.info(
            f"[TrainMechResearch {session_id}] Done. "
            f"mechanism={mechanism_name}, strategy={impl_strategy}, "
            f"target={target}, retries={retries}"
        )
        return result

    def apply(self, runner_path: Path, result: TrainMechanismResult) -> bool:
        """
        Apply the generated code patch to runner.py.

        Strategy determines how the patch is applied:
          - new_helper_class: append before the TrainRunner class definition
          - new_method / replace_method: insert or replace inside TrainRunner
          - modify_init: append to TrainRunner.__init__

        Returns True if the modified file passes a syntax check.
        """
        runner_path = Path(runner_path)
        original = runner_path.read_text(encoding="utf-8")

        # Always back up first
        backup_path = runner_path.with_suffix(f".py.bak_{result.session_id}")
        backup_path.write_text(original, encoding="utf-8")
        logger.info(f"[Apply] Backed up runner.py to {backup_path}")

        strategy = result.implementation_strategy
        code = result.code.strip()

        try:
            if strategy == "new_helper_class":
                patched = self._insert_helper_class(original, code)
            elif strategy == "replace_method":
                patched = self._replace_method(original, result.target, code)
            elif strategy == "new_method":
                patched = self._insert_method(original, code)
            elif strategy == "modify_init":
                patched = self._append_to_init(original, code)
            else:
                # Fallback: append before TrainRunner class
                logger.warning(
                    f"[Apply] Unknown strategy '{strategy}', "
                    f"falling back to new_helper_class insertion"
                )
                patched = self._insert_helper_class(original, code)

        except Exception as e:
            logger.error(f"[Apply] Patch application failed: {e}")
            result.validation_error = f"patch_apply_error: {e}"
            return False

        # Syntax check before writing
        error = self._syntax_check(patched)
        if error:
            logger.error(f"[Apply] Patched code has syntax error: {error}")
            result.validation_error = f"syntax_error: {error}"
            # Restore backup
            runner_path.write_text(original, encoding="utf-8")
            return False

        runner_path.write_text(patched, encoding="utf-8")
        (result.session_dir / "05_patched_runner.py").write_text(patched, encoding="utf-8")
        result.applied = True
        logger.info(f"[Apply] Successfully patched {runner_path}")
        return True

    def validate(self, runner_path: Path) -> bool:
        """
        Verify the modified runner.py can be imported without errors.

        Uses a subprocess to avoid polluting the current process's module cache.
        Returns True if import succeeds.
        """
        import subprocess
        runner_path = Path(runner_path)
        project_root = runner_path.parent.parent.parent  # research_evo_mvp/

        module_path = (
            runner_path.relative_to(project_root)
            .with_suffix("")
            .as_posix()
            .replace("/", ".")
        )

        cmd = [
            "python", "-c",
            f"import sys; sys.path.insert(0, '{project_root}'); "
            f"import {module_path}; print('IMPORT_OK')"
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=30
            )
            if "IMPORT_OK" in proc.stdout:
                logger.info(f"[Validate] Import check passed for {module_path}")
                return True
            else:
                err = proc.stderr.strip() or proc.stdout.strip()
                logger.error(f"[Validate] Import check failed: {err[:500]}")
                return False
        except subprocess.TimeoutExpired:
            logger.error("[Validate] Import check timed out")
            return False
        except Exception as e:
            logger.error(f"[Validate] Import check error: {e}")
            return False

    # ------------------------------------------------------------------
    # Code patch helpers
    # ------------------------------------------------------------------

    def _insert_helper_class(self, original: str, new_class_code: str) -> str:
        """Insert a new helper class before the TrainRunner class definition."""
        marker = "\nclass TrainRunner:"
        idx = original.find(marker)
        if idx == -1:
            # Try without leading newline
            marker = "class TrainRunner:"
            idx = original.find(marker)
        if idx == -1:
            raise ValueError("Could not find 'class TrainRunner:' in runner.py")

        separator = "\n\n\n# ---------------------------------------------------------------------------\n"
        insert_block = f"{separator}{new_class_code.strip()}\n\n\n"
        return original[:idx] + insert_block + original[idx:]

    def _replace_method(self, original: str, method_name: str, new_method_code: str) -> str:
        """Replace an existing method in TrainRunner with new code.

        Finds the method by its def line and replaces it with the new code.
        Uses AST to determine the method boundaries precisely.
        """
        # First try AST-based replacement (precise)
        try:
            return self._ast_replace_method(original, method_name, new_method_code)
        except Exception as e:
            logger.warning(f"[Replace] AST replacement failed ({e}), trying regex fallback")

        # Fallback: regex-based replacement (less precise but works for simple cases)
        pattern = rf'(    def {re.escape(method_name)}\(.*?\n)((?:(?!    def |\nclass ).)*)'
        new_block = new_method_code.rstrip() + "\n"
        # Indent the replacement if needed
        lines = new_method_code.splitlines()
        if lines and not lines[0].startswith("    "):
            new_block = textwrap.indent(new_method_code.strip(), "    ") + "\n"
        patched, count = re.subn(pattern, new_block, original, count=1, flags=re.DOTALL)
        if count == 0:
            raise ValueError(
                f"Could not find method '{method_name}' in runner.py for replacement"
            )
        return patched

    def _ast_replace_method(self, original: str, method_name: str, new_method_code: str) -> str:
        """Precisely replace a method using AST node positions."""
        tree = ast.parse(original)
        lines = original.splitlines(keepends=True)

        # Find the TrainRunner class
        runner_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TrainRunner":
                runner_class = node
                break
        if runner_class is None:
            raise ValueError("TrainRunner class not found via AST")

        # Find the target method
        target_node = None
        for node in runner_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == method_name:
                target_node = node
                break
        if target_node is None:
            raise ValueError(f"Method '{method_name}' not found in TrainRunner")

        start_line = target_node.lineno - 1   # 0-indexed
        end_line = target_node.end_lineno     # exclusive (already 1-indexed end)

        # Prepare replacement with proper indentation
        new_lines = new_method_code.rstrip().splitlines()
        if new_lines and not new_lines[0].startswith("    "):
            new_method_indented = textwrap.indent(new_method_code.strip(), "    ") + "\n"
        else:
            new_method_indented = new_method_code.rstrip() + "\n"

        patched_lines = lines[:start_line] + [new_method_indented] + lines[end_line:]
        return "".join(patched_lines)

    def _insert_method(self, original: str, new_method_code: str) -> str:
        """Append a new method to the TrainRunner class (before last class-level code).

        Inserts just before the final private helper block, or at the end of the class.
        """
        # Find the end of the file or the last method in TrainRunner
        # Simple strategy: append before the last line of the file that's inside the class
        # More robust: insert before the _apply_changes method (always present)
        anchor = "    def _apply_changes("
        idx = original.find(anchor)
        if idx == -1:
            # Fallback: append at end of file
            new_block = "\n" + textwrap.indent(new_method_code.strip(), "    ") + "\n"
            return original.rstrip() + new_block + "\n"

        # Insert before anchor
        lines = new_method_code.rstrip().splitlines()
        if lines and not lines[0].startswith("    "):
            insert_block = textwrap.indent(new_method_code.strip(), "    ")
        else:
            insert_block = new_method_code.strip()

        insert_text = "\n" + insert_block + "\n\n"
        return original[:idx] + insert_text + original[idx:]

    def _append_to_init(self, original: str, new_init_code: str) -> str:
        """Append attribute initializations to TrainRunner.__init__.

        Finds the end of __init__ and inserts before the next method definition.
        """
        try:
            return self._ast_append_to_init(original, new_init_code)
        except Exception as e:
            logger.warning(f"[AppendInit] AST approach failed ({e}), using regex")

        # Fallback: find last line of __init__ before next method
        init_pattern = r'(    def __init__\(.*?\n(?:(?!    def ).)*)'
        match = re.search(init_pattern, original, re.DOTALL)
        if not match:
            raise ValueError("Could not find __init__ in TrainRunner")

        insert_at = match.end()
        snippet = textwrap.indent(new_init_code.strip(), "        ") + "\n"
        return original[:insert_at] + snippet + original[insert_at:]

    def _ast_append_to_init(self, original: str, new_code: str) -> str:
        """Append code to __init__ using AST positions."""
        tree = ast.parse(original)
        lines = original.splitlines(keepends=True)

        runner_class = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TrainRunner":
                runner_class = node
                break
        if runner_class is None:
            raise ValueError("TrainRunner not found")

        init_node = None
        for node in runner_class.body:
            if isinstance(node, ast.FunctionDef) and node.name == "__init__":
                init_node = node
                break
        if init_node is None:
            raise ValueError("__init__ not found in TrainRunner")

        # Insert at the end of __init__
        end_line = init_node.end_lineno  # 1-indexed, inclusive
        snippet = textwrap.indent(new_code.strip(), "        ") + "\n"
        patched_lines = lines[:end_line] + [snippet] + lines[end_line:]
        return "".join(patched_lines)

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _generate_with_retries(
        self,
        spec: str,
        reference_code: str,
        codegen_task: str,
        session_dir: Path,
    ) -> tuple[str, int]:
        """Generate code, retrying with error feedback on failures."""
        code = self.client.call(
            CODEGEN_PROMPT.format(
                spec=spec,
                reference_code=reference_code,
                codegen_task=codegen_task,
            ),
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
            (session_dir / "04_code_final_error.txt").write_text(final_error, encoding="utf-8")
            raise RuntimeError(
                f"Code generation failed after {self.max_code_retries} retries. "
                f"Last error: {final_error}"
            )
        return code, self.max_code_retries

    # ------------------------------------------------------------------
    # Parsing / extraction helpers
    # ------------------------------------------------------------------

    def _extract_selected(self, exploration: str, critique: str) -> str:
        """Pull the selected hypothesis from critique output."""
        for line in critique.splitlines():
            if line.strip().lower().startswith("**selected**"):
                return line.strip()
        # Fallback: last 800 chars of exploration
        return exploration[-800:]

    def _parse_spec_metadata(
        self, spec: str, session_id: str
    ) -> tuple[str, str, str]:
        """Extract mechanism_name, implementation_strategy, and target from spec text.

        Returns (mechanism_name, impl_strategy, target).
        """
        mechanism_name = f"generated_mechanism_{session_id}"
        impl_strategy = "new_helper_class"  # safe default
        target = f"GeneratedMechanism_{session_id}"

        valid_strategies = {
            "new_method", "replace_method", "new_helper_class", "modify_init"
        }

        for line in spec.splitlines():
            low = line.lower()

            if "mechanism name" in low or "mechanism_name" in low:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    candidate = parts[1].strip().strip("`").strip("*").strip()
                    if candidate and " " not in candidate:
                        mechanism_name = re.sub(r"[^a-zA-Z0-9_]", "", candidate)

            if "implementation strategy" in low or "implementation_strategy" in low:
                for strat in valid_strategies:
                    if strat in low.replace(" ", "_") or strat in low:
                        impl_strategy = strat
                        break

            if "target" in low and ":" in line:
                parts = line.split(":", 1)
                if len(parts) > 1:
                    candidate = parts[1].strip().strip("`").strip("*").strip()
                    if candidate and len(candidate) < 80:
                        target = candidate

        return mechanism_name, impl_strategy, target

    def _summarize_existing_mechanisms(self, runner_code: str) -> str:
        """Extract the docstring header from runner.py listing existing mechanisms."""
        # The docstring at the top of runner.py lists all implemented improvements
        lines = runner_code.splitlines()
        in_docstring = False
        docstring_lines: list[str] = []
        for line in lines[:120]:  # Only look at the first 120 lines
            if line.strip().startswith('"""') and not in_docstring:
                in_docstring = True
                docstring_lines.append(line)
                if line.count('"""') >= 2:
                    break  # Single-line docstring
                continue
            if in_docstring:
                docstring_lines.append(line)
                if '"""' in line and len(docstring_lines) > 1:
                    break
        summary = "\n".join(docstring_lines[:60])
        if not summary:
            summary = "(runner.py docstring not found — mechanisms unknown)"
        return summary

    def _extract_runner_section(self, runner_code: str) -> str:
        """Extract the TrainRunner class definition (first 3000 chars) for context."""
        idx = runner_code.find("class TrainRunner:")
        if idx == -1:
            return runner_code[:3000]
        return runner_code[idx: idx + 3000]

    def _infer_bottleneck(self, trace_summary: str) -> str:
        """Try to infer the bottleneck from the trace text."""
        low = trace_summary.lower()
        if "crash" in low and trace_summary.lower().count("crash") > 2:
            return (
                "High crash rate — many proposals result in OOM or training divergence. "
                "Need better crash avoidance or faster recovery."
            )
        if "discard" in low and trace_summary.lower().count("discard") > 5:
            return (
                "Too many discards with no improvement — proposals are not helpful. "
                "Need better proposal generation or exploration strategy."
            )
        return (
            "Search appears to be converging slowly. "
            "Need improved exploration or better use of search history."
        )

    def _build_codegen_task(
        self,
        mechanism_name: str,
        impl_strategy: str,
        target: str,
        spec: str,
    ) -> str:
        """Build a clear task description for the code generation prompt."""
        if impl_strategy == "new_helper_class":
            return (
                f"Implement a new helper class named '{target}' (or a descriptive variant). "
                f"This class will be added to runner.py and used by TrainRunner. "
                f"Include docstring, __init__, and all methods described in the spec."
            )
        elif impl_strategy == "replace_method":
            return (
                f"Write a REPLACEMENT for the method '{target}' in TrainRunner. "
                f"Include the full method with proper 4-space indentation (it will be placed "
                f"inside the class body). Match the existing method's signature exactly unless "
                f"the spec explicitly changes it."
            )
        elif impl_strategy == "new_method":
            return (
                "Write a new method for the TrainRunner class. "
                "The method should be named something descriptive (4-space indent, "
                "it goes inside the class body). Implement the mechanism described in the spec."
            )
        elif impl_strategy == "modify_init":
            return (
                "Write Python statements that initialize new attributes for TrainRunner. "
                "These will be appended to the end of __init__. "
                "Use 8-space indentation (inside __init__). "
                "Include inline comments explaining each attribute."
            )
        else:
            return (
                f"Implement the mechanism '{mechanism_name}' as described in the spec. "
                f"Write clean, well-commented Python code."
            )

    def _read_reference_code(self, runner_code: str, impl_strategy: str) -> str:
        """Extract a relevant reference snippet from runner.py for style guidance."""
        if impl_strategy == "new_helper_class":
            # Use ElitePool as a reference (it's a well-structured helper class)
            idx = runner_code.find("class ElitePool:")
            if idx != -1:
                return runner_code[idx: idx + 2000]
            # Fallback: CrashMemory
            idx = runner_code.find("class CrashMemory:")
            if idx != -1:
                return runner_code[idx: idx + 1500]
        elif impl_strategy in ("replace_method", "new_method"):
            # Use _propose as reference (it's the most complex method)
            idx = runner_code.find("    def _propose(")
            if idx != -1:
                return runner_code[idx: idx + 1500]
        elif impl_strategy == "modify_init":
            idx = runner_code.find("    def __init__(")
            if idx != -1:
                return runner_code[idx: idx + 1500]

        # Generic fallback: first 1000 chars of TrainRunner
        idx = runner_code.find("class TrainRunner:")
        if idx != -1:
            return runner_code[idx: idx + 1000]
        return "(reference not available)"

    def _save_summary(self, result: TrainMechanismResult, session_dir: Path) -> None:
        summary = {
            "session_id": result.session_id,
            "mechanism_name": result.mechanism_name,
            "implementation_strategy": result.implementation_strategy,
            "target": result.target,
            "code_retries": result.code_retries,
            "hypothesis": result.hypothesis[:300],
        }
        (session_dir / "06_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
