"""Stage C: Experiment plan or code — lesson injection point.

Two-phase generation:
1. Generate the experiment plan (text) — covers methodology, controls, metrics
2. Generate the Python mock implementation separately (dedicated token budget)
Combining both prevents the code from being truncated by the plan text.
"""
from ..llm_client import call_llm
from .base import BaseStage

PLAN_SYSTEM = """You are a research engineer designing experiments.
Design concrete, executable experiments that directly test the stated hypotheses.
Be specific about methodology, metrics, baselines, and expected outputs.
Do NOT include Python code — that will be generated separately.

CRITICAL COMPLETENESS RULE: Cover ALL hypotheses with specific experiments.
If budget is tight, write shorter sections — but address every hypothesis."""

CODE_SYSTEM = """You are a Python developer implementing a research simulation.
Generate complete, executable Python code that simulates the experiment.
The code must be runnable end-to-end without modification.

CRITICAL COMPLETENESS RULES:
- Use EXACTLY 6 tasks (TC-001 through TC-006) — not more, not fewer
- All 4 hypotheses (H1 through H4) must have simulation functions
- No stubs, no "# TODO", no truncated definitions
- If you run low on space, use shorter variable names / docstrings — but COMPLETE all functions
- The code must be self-contained and executable"""


class ExperimentPlanStage(BaseStage):
    name = "experiment_plan_or_code"
    max_retries = 2  # Extra retry for complex two-phase generation

    def run(self, context: dict) -> dict:
        topic = context["topic"]
        hypotheses = context["previous_outputs"].get("hypothesis_generation", "")
        lessons_text = context.get("retrieved_lessons", "")  # INJECTION POINT

        lessons_section = f"\n{lessons_text}\n" if lessons_text else ""
        hyp_text = hypotheses[:3500] if len(hypotheses) > 3500 else hypotheses

        # ── Phase 1: Generate experiment plan (text only) ─────────────────────
        plan_prompt = f"""## Research Topic
{topic}

## Hypotheses to Test
{hyp_text}
{lessons_section}
## Your Task
Design a concrete experiment plan to test ALL of the hypotheses above.
Do NOT include Python code — that will be written separately.

⚠️ COVERAGE REQUIREMENT: You MUST address ALL hypotheses listed (H1, H2, H3, H4 — not just H1).

Provide:
1. **Experiment Overview**: What will be tested and why
2. **Methodology per Hypothesis**: Write one dedicated subsection FOR EACH hypothesis (H1 through H4)
3. **Task Corpus**: Use exactly 6 tasks (TC-001 through TC-006) — document this size explicitly
4. **Baseline / Controls**: What are we comparing against?
5. **Metrics**: Exactly what will be measured
   ⚠️ Do NOT use BLEU-4, BERTScore, or ROUGE — these measure surface similarity, not research quality.
   Use: expert judgment rubrics, factuality scores, or construct-validated evaluation.
6. **Expected Results per Hypothesis**: What outcomes support/refute each hypothesis?
7. **Limitations**: What would this experiment NOT tell us?"""

        plan_content = call_llm(plan_prompt, system=PLAN_SYSTEM, model=self.model, max_tokens=7000)

        # ── Phase 2: Generate mock implementation (dedicated budget) ──────────
        code_prompt = f"""## Research Topic
{topic}

## Hypotheses to Test
{hyp_text}

## Experiment Plan Summary
{plan_content[:2500]}

## Task
Write a complete Python mock implementation to simulate this experiment.

Requirements:
- EXACTLY 6 tasks in the task corpus (TC-001 through TC-006) — match plan's 3 domains (2 tasks each)
- Each task: dict with keys: id, description, domain, complexity, expected_feedback_type
- Simulation functions for ALL 4 hypotheses (simulate_h1, simulate_h2, simulate_h3, simulate_h4)
- Use IDENTICAL parameter names and domain categories as the plan above
- A main() function that runs all simulations and prints results
- No BLEU-4, BERTScore, or ROUGE metrics
- Keep all numerical parameters (latencies, thresholds) consistent with plan values
- Code must be complete and executable end-to-end

Return ONLY the Python code (no explanation, no markdown fences)."""

        code_content = call_llm(code_prompt, system=CODE_SYSTEM, model=self.model, max_tokens=7000)

        # Clean code if wrapped in markdown fences
        import re
        code_match = re.search(r"```python\s*\n(.*?)\n```", code_content, re.DOTALL)
        if code_match:
            code_content = code_match.group(1)

        # Combine for stage content (evaluation sees both)
        combined = f"{plan_content}\n\n## Mock Implementation\n\n```python\n{code_content}\n```"

        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "experiment_plan.md", combined))
        artifacts.append(self._save_artifact(context["run_dir"], "experiment_code.py", code_content))

        return {"content": combined, "artifacts": artifacts}
