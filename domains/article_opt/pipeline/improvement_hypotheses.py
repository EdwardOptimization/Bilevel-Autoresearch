"""Stage B: Improvement Hypotheses — lesson injection point.

Generate H1-H4 improvement hypotheses grounded in Stage A's analysis.
Each hypothesis targets a specific rubric weakness and proposes a concrete fix.
"""
import json

from core.llm_client import call_llm, parse_json_response

from .base import BaseStage

SYSTEM = """You are a senior editor generating improvement hypotheses for an article.
Each hypothesis must:
- Target a SPECIFIC weakness identified in the article analysis
- Propose a CONCRETE, verifiable fix (not "make it clearer" — name the exact change)
- Be falsifiable: after applying the fix, a reviewer can objectively check if it worked
- Be independent from other hypotheses where possible

CRITICAL COMPLETENESS RULE: Generate ALL 4 hypotheses fully.
H1 through H4 — never truncate. If space is tight, write shorter rationales
but complete all 4 hypotheses."""


class ImprovementHypothesesStage(BaseStage):
    name = "improvement_hypotheses"

    def run(self, context: dict) -> dict:
        article = context["article_content"]
        analysis = context["previous_outputs"].get("article_analysis", "")
        lessons_text = context.get("retrieved_lessons", "")  # INJECTION POINT
        retry_feedback = context.get("evaluator_feedback", "")

        lessons_section = f"\n{lessons_text}\n" if lessons_text else ""
        retry_section = f"\n## Previous Attempt Feedback\n{retry_feedback}\n" if retry_feedback else ""
        outer_section = self._outer_guidance(context)
        analysis_excerpt = analysis[:3000] if len(analysis) > 3000 else analysis

        prompt = f"""## Article (first 1500 chars for reference)
{article[:1500]}

## Article Analysis
{analysis_excerpt}
{lessons_section}{outer_section}{retry_section}
## Your Task
Generate exactly 4 improvement hypotheses based on the analysis above.

For each hypothesis (H1 through H4) provide:
1. **Target Dimension**: Which rubric dimension (A/B/C/D/E) does this address?
2. **Weakness Being Fixed**: Quote the specific sentence or section from the analysis
   that identifies this weakness
3. **Hypothesis Statement**: "If we [specific change], then [dimension X] will improve
   from [current score] to [target score] because [mechanism]"
4. **Concrete Fix**: Exactly what text to add, modify, or restructure — be specific enough
   that a different editor could apply it without asking follow-up questions
5. **Verification**: How will we know the fix worked? (what would a reviewer see?)
6. **Priority**: High / Medium / Low with one-sentence justification

ORDERING: List hypotheses from highest to lowest impact on overall score.
Prioritize dimensions furthest below their target threshold."""

        content = call_llm(prompt, system=SYSTEM, model=self.model, max_tokens=6000)

        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "hypotheses.md", content))

        # Structured JSON
        struct_prompt = f"""Convert these improvement hypotheses to a JSON array.
Each item: {{
  "id": "H1",
  "target_dimension": "A",
  "hypothesis": "If we... then...",
  "concrete_fix": "...",
  "priority": "high|medium|low"
}}

Hypotheses:
{content[:3000]}

Return ONLY the JSON array."""
        struct_raw = call_llm(struct_prompt, model=self.model, max_tokens=3000)
        struct = parse_json_response(struct_raw)
        if isinstance(struct, list):
            artifacts.append(self._save_artifact(
                context["run_dir"], "hypotheses.json",
                json.dumps(struct, indent=2, ensure_ascii=False)
            ))

        return {"content": content, "artifacts": artifacts}
