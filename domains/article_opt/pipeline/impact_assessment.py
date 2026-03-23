"""Stage D: Impact Assessment — predict post-edit rubric scores and flag risks.

Simulates what a reviewer would score AFTER the edit plan is applied.
Also flags regressions: edits that might hurt other dimensions.
"""
import json

from core.llm_client import call_llm, parse_json_response

from .base import BaseStage

SYSTEM = """You are a rigorous article reviewer predicting the impact of proposed edits.
Your job: given the current article and a concrete edit plan, predict the rubric scores
AFTER the edits are applied, and identify any risks of regression.

Be honest. If an edit plan is vague or partial, predict accordingly — do not assume
perfect execution. Flag any edit that could hurt dimensions other than its target."""

RUBRIC_BRIEF = """Rubric dimensions (0–10):
A: Argumentative Rigor (target ≥8) — claims have clear support chains
B: Conceptual Clarity (target ≥8) — key terms defined and consistent
C: Cross-Article Consistency (target ≥8) — no contradictions with companion articles
D: Insight Novelty (target ≥7) — non-obvious, explicitly stated insights
E: Actionability (target ≥7) — clear implication for practice or future work"""


class ImpactAssessmentStage(BaseStage):
    name = "impact_assessment"

    def run(self, context: dict) -> dict:
        article = context["article_content"]
        analysis = context["previous_outputs"].get("article_analysis", "")
        edit_plan = context["previous_outputs"].get("edit_planning", "")
        retry_feedback = context.get("evaluator_feedback", "")
        retry_section = f"\n## Previous Attempt Feedback\n{retry_feedback}\n" if retry_feedback else ""

        plan_excerpt = edit_plan[:3500] if len(edit_plan) > 3500 else edit_plan
        analysis_excerpt = analysis[:2000] if len(analysis) > 2000 else analysis

        prompt = f"""## Current Article (first 2000 chars)
{article[:2000]}

## Current Rubric Assessment
{analysis_excerpt}

## Proposed Edit Plan
{plan_excerpt}

{RUBRIC_BRIEF}
{retry_section}
## Your Task

### Part 1: Predicted Post-Edit Scores
For each dimension (A through E), predict the score AFTER the edit plan is applied:
- Current score → Predicted score
- Reasoning: which specific edit caused the change (or why it didn't change)

### Part 2: Regression Risks
List any edits that might HURT dimensions other than their target:
- Edit: [which hypothesis's edit]
- Risk: [which dimension could regress]
- Mechanism: [why this could happen]
- Mitigation: [what to watch for in Stage E]

### Part 3: Go / No-Go Decision
- **Proceed**: The plan is sound, proceed to Stage E (revised output)
- **Revise plan**: One or more edits need rethinking (describe what)
- **Abort pass**: The plan is fundamentally flawed (describe why)

If Go: state the expected overall score improvement (e.g., "5.6 → 7.2")."""

        content = call_llm(prompt, system=SYSTEM, model=self.model, max_tokens=5000)

        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "impact_assessment.md", content))

        # Extract predicted scores as JSON
        score_prompt = f"""Extract predicted post-edit scores from this impact assessment.
Return JSON with:
- "predicted_scores": dict with keys A, B, C, D, E (integers 0-10)
- "overall_predicted": float (average, 1 decimal)
- "decision": "proceed" | "revise_plan" | "abort"
- "regression_risks": list of dimension letters at risk

Impact assessment:
{content[:2000]}

Return ONLY the JSON object."""
        score_raw = call_llm(score_prompt, model=self.model, max_tokens=2000)
        predicted = parse_json_response(score_raw)
        if isinstance(predicted, dict) and "raw_content" not in predicted:
            artifacts.append(self._save_artifact(
                context["run_dir"], "predicted_scores.json",
                json.dumps(predicted, indent=2, ensure_ascii=False)
            ))

        return {"content": content, "artifacts": artifacts, "predicted": predicted}
