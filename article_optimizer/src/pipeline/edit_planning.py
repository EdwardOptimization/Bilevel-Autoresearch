"""Stage C: Edit Planning — two-phase: select hypotheses, then plan concrete edits.

Two-phase like V1's experiment_plan_or_code:
Phase 1: Select which hypotheses to implement this run and why (triage)
Phase 2: Concrete edit plan with exact section references and new content drafts

Lesson injection point.
max_retries = 2 (complex stage, like V1's Stage C).
"""
from ..llm_client import call_llm
from .base import BaseStage

TRIAGE_SYSTEM = """You are a senior editor triaging article improvements.
Select the highest-impact hypotheses to implement in this revision pass.
Be ruthless: it is better to implement 2 hypotheses well than 4 poorly.
Do NOT include Python code or pseudocode."""

PLAN_SYSTEM = """You are a technical editor writing a precise revision plan.
Each edit plan item must be specific enough that a different editor could
execute it exactly without asking follow-up questions.

Specify:
- The exact section (by header name or first words of the paragraph)
- The exact type of change (add sentence / rewrite paragraph / restructure section / add example)
- The exact new content to add (write the actual sentences, not a description of them)
- What to remove or replace (quote the original text being replaced)"""


class EditPlanningStage(BaseStage):
    name = "edit_planning"
    max_retries = 2

    def run(self, context: dict) -> dict:
        article = context["article_content"]
        analysis = context["previous_outputs"].get("article_analysis", "")
        hypotheses = context["previous_outputs"].get("improvement_hypotheses", "")
        lessons_text = context.get("retrieved_lessons", "")  # INJECTION POINT
        retry_feedback = context.get("evaluator_feedback", "")

        lessons_section = f"\n{lessons_text}\n" if lessons_text else ""
        retry_section = f"\n## Previous Attempt Feedback\n{retry_feedback}\n" if retry_feedback else ""
        hyp_excerpt = hypotheses[:3000] if len(hypotheses) > 3000 else hypotheses

        # ── Phase 1: Triage ───────────────────────────────────────────────────
        triage_prompt = f"""## Improvement Hypotheses
{hyp_excerpt}

## Article Analysis Summary
{analysis[:1500] if len(analysis) > 1500 else analysis}
{lessons_section}{retry_section}
## Your Task: Triage
Select 2-3 hypotheses to implement in this revision pass.

For each selected hypothesis:
1. **Why selected**: How much will this improve the overall rubric score?
2. **Implementation risk**: Could this change introduce new weaknesses?
3. **Dependencies**: Does this hypothesis need to be implemented before/after another?

For rejected hypotheses: one-sentence reason for deferring.

End with: **Selected for this pass: H? H? [H?]** (list the IDs)"""

        triage = call_llm(triage_prompt, system=TRIAGE_SYSTEM, model=self.model, max_tokens=2000)

        # ── Phase 2: Concrete edit plan ───────────────────────────────────────
        plan_prompt = f"""## Article
---
{article}
---

## Selected Hypotheses to Implement
{triage}

## Full Hypothesis Details
{hyp_excerpt}

## Your Task: Write the Concrete Edit Plan
For each selected hypothesis, produce an edit plan with:

1. **Hypothesis**: [H? — one-line summary]
2. **Section**: [exact section header or "Introduction" / "Conclusion" etc.]
3. **Change Type**: add_sentence | rewrite_paragraph | restructure_section | add_example | add_definition
4. **Original Text** (if replacing): quote the exact text being replaced
5. **New Text**: write the EXACT new sentences/paragraph to add or use as replacement
   Do not write "[add example here]" — write the actual example.
6. **Placement**: before/after which sentence or at what position in the section

CONSTRAINT: Every "New Text" field must be final, ready-to-insert prose.
No placeholders. No "[write X here]". The text must be ready to paste into the article."""

        plan = call_llm(plan_prompt, system=PLAN_SYSTEM, model=self.model, max_tokens=7000)

        combined = f"## Triage\n\n{triage}\n\n---\n\n## Edit Plan\n\n{plan}"

        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "triage.md", triage))
        artifacts.append(self._save_artifact(context["run_dir"], "edit_plan.md", plan))
        artifacts.append(self._save_artifact(context["run_dir"], "combined.md", combined))

        return {"content": combined, "artifacts": artifacts}
