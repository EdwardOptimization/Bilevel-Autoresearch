import json
from src.pipeline.base import BaseStage
from src.llm_client import call_llm, parse_json_response

SYSTEM = """You are a precision editing coach. Analyze this edit plan for logical sub-skills."""

class SubskillFeedbackLoopStage(BaseStage):
    name = "subskill_feedback_loop"

    def __init__(self, model: str = ""):
        super().__init__(model)

    def run(self, context: dict) -> dict:
        plan_text = context["previous_outputs"].get("edit_planning", "")
        critique_text = context["previous_outputs"].get("article_analysis", "")
        article_text = context["article_content"]
        lessons_text = context.get("retrieved_lessons", "")
        retry_feedback = context.get("evaluator_feedback", "")

        lessons_section = f"\n{lessons_text}\n" if lessons_text else ""
        retry_section = f"\n## Previous Attempt Feedback\n{retry_feedback}\n" if retry_feedback else ""
        outer_section = self._outer_guidance(context)

        prompt = f"""ORIGINAL CRITIQUE: {critique_text[:2000]}

EDIT PLAN TO ANALYZE:
{plan_text[:3000]}

ANALYSIS TASK:
1. Extract these components:
   - Premises (claims with intended support)
   - Transitions between ideas
   - Technical/jargon terms
   - Conclusion statements

2. Score each sub-skill (0-1, 1=excellent):
   - Premise Clarity: Each premise clearly states claim AND evidence?
   - Transition Logic: Explicit links between ideas?
   - Jargon Usage: Technical terms explained or necessary?
   - Conclusion Support: Conclusions reference specific premises?

3. For scores < 0.7, give ONE specific directive:
   Format: "In '[exact text]', [problem]. Try: '[example rewrite]'"

4. Revise plan applying top 2 directives.

OUTPUT FORMAT:
{{
  "subskill_scores": {{"premise_clarity": X, "transition_logic": X, "jargon_usage": X, "conclusion_support": X}},
  "specific_feedback": ["directive1", "directive2", ...],
  "revised_plan": "Revised text here..."
}}
{lessons_section}{outer_section}{retry_section}"""

        content = call_llm(prompt, system=SYSTEM, model=self.model, max_tokens=800)
        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "subskill_feedback.md", content))

        struct_prompt = f"""Convert the analysis to a JSON object exactly as specified.
The output must be a JSON object with keys: subskill_scores, specific_feedback, revised_plan.
subskill_scores must be an object with keys: premise_clarity, transition_logic, jargon_usage, conclusion_support.
specific_feedback must be a list of strings.
revised_plan must be a string.

Analysis:
{content[:2000]}

Return ONLY the JSON object."""
        struct_raw = call_llm(struct_prompt, model=self.model, max_tokens=1500)
        struct = parse_json_response(struct_raw)
        if isinstance(struct, dict):
            artifacts.append(self._save_artifact(
                context["run_dir"], "subskill_feedback.json",
                json.dumps(struct, indent=2, ensure_ascii=False)
            ))

        return {"content": content, "artifacts": artifacts}