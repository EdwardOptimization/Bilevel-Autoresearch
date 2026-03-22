"""Stage B: Hypothesis generation — lesson injection point."""
import json

from ..llm_client import call_llm, parse_json_response
from .base import BaseStage

SYSTEM = """You are a research scientist generating testable hypotheses.
Each hypothesis must be:
- Specific and falsifiable
- Grounded in the literature
- Accompanied by explicit, measurable evaluation criteria with named statistical tests
Do not generate vague or untestable claims.

CRITICAL COMPLETENESS RULE: You MUST generate ALL hypotheses completely. Never truncate.
If you are approaching a length limit, write SHORTER hypotheses — but NEVER stop before
completing the last one. A truncated hypothesis is a critical pipeline failure."""


class HypothesisGenerationStage(BaseStage):
    name = "hypothesis_generation"

    def run(self, context: dict) -> dict:
        topic = context["topic"]
        lit_scan = context["previous_outputs"].get("literature_scan", "")
        lessons_text = context.get("retrieved_lessons", "")  # INJECTION POINT

        lessons_section = f"\n{lessons_text}\n" if lessons_text else ""

        prompt = f"""## Research Topic
{topic}

## Literature Scan
{lit_scan[:2000] if len(lit_scan) > 2000 else lit_scan}
{lessons_section}
## Your Task
Generate exactly 4 research hypotheses about this topic.
(4 hypotheses — not 3, not 5. This ensures completeness within the output budget.)

For each hypothesis provide:
1. **Hypothesis Statement**: Clear, falsifiable claim
2. **Rationale**: Why this hypothesis follows from the literature (2-3 sentences)
3. **Operationalized Variables**: IV and DV with specific measurement methods
4. **Evaluation Criteria**: Named statistical test + expected effect size (e.g., "paired t-test, Cohen's d ≥ 0.5, p < 0.05")
5. **Expected Outcome**: What result confirms vs. refutes the hypothesis
6. **Priority**: High / Medium / Low (with one-sentence justification)

MANDATORY: All 4 hypotheses must be fully complete. Write H1 through H4 in order.
Before writing H4, verify you have enough space to complete it. If not, write a shorter H4 — but NEVER truncate it."""

        content = call_llm(prompt, system=SYSTEM, model=self.model, max_tokens=5500)

        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "hypotheses.md", content))

        # Structured JSON version
        struct_prompt = f"""Convert these hypotheses to a JSON array.
Each item: {{"id": "H1", "statement": "...", "evaluation_criteria": "...", "priority": "high|medium|low"}}

Hypotheses:
{content}

Return ONLY the JSON array."""
        struct_raw = call_llm(struct_prompt, model=self.model, max_tokens=1024)
        struct = parse_json_response(struct_raw)
        if isinstance(struct, list):
            struct_json = json.dumps(struct, indent=2, ensure_ascii=False)
        else:
            struct_json = json.dumps([], indent=2)
        artifacts.append(self._save_artifact(context["run_dir"], "hypotheses.json", struct_json))

        return {"content": content, "artifacts": artifacts}
