"""Stage D: Experiment result summary."""
import json

from ..llm_client import call_llm, parse_json_response
from .base import BaseStage

SYSTEM = """You are a research analyst summarizing experimental results.
Be honest about what the experiment found — including null and negative results.
Do not exaggerate positive findings. Link results explicitly to each hypothesis."""


class ExperimentResultSummaryStage(BaseStage):
    name = "experiment_result_summary"

    def run(self, context: dict) -> dict:
        topic = context["topic"]
        experiment_plan = context["previous_outputs"].get("experiment_plan_or_code", "")
        hypotheses = context["previous_outputs"].get("hypothesis_generation", "")
        # Skills (structural guidance) are injected; raw lessons are NOT (to preserve honest reporting)
        skill_text = context.get("retrieved_lessons", "")
        skill_section = f"\n{skill_text}\n" if skill_text else ""

        prompt = f"""## Research Topic
{topic}

## Hypotheses
{hypotheses[:3000] if len(hypotheses) > 3000 else hypotheses}

## Experiment Plan
{experiment_plan[:2000] if len(experiment_plan) > 2000 else experiment_plan}
{skill_section}
## Your Task
Simulate running the experiment from the plan and summarize the results.

⚠️ MANDATORY DISCLOSURE: Your FIRST sentence must be:
"NOTE: The following results are based on a simulated experimental run conducted
for methodological demonstration purposes. All quantitative findings are theoretical
projections, not empirical measurements."

Then provide:
1. **Execution Summary**: What was simulated and how
2. **Results per Hypothesis**: For EACH hypothesis — confirmed / refuted / inconclusive + evidence
   Include simulated quantitative findings with uncertainty ranges (e.g., "34.2% ± 4.1%")
3. **Key Findings**: Top 3-5 most important findings with specific numbers
4. **Unexpected Observations**: ≥3 anomalies, negative results, or surprising findings
   For each: (a) why it was surprising and (b) what it implies about the theory
5. **Negative Results**: What didn't work or wasn't supported? Be specific.
6. **Data Quality**: How reliable are these results? What are confidence levels?

Be realistic — not every hypothesis will be confirmed. Mixed results are expected and valuable."""

        content = call_llm(prompt, system=SYSTEM, model=self.model, max_tokens=5000)

        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "results.md", content))

        # Structured results JSON
        struct_prompt = f"""Extract results per hypothesis as JSON.
Array of: {{"hypothesis_id": "H1", "verdict": "confirmed|refuted|inconclusive", "evidence": "...", "confidence": 0.0-1.0}}

Results:
{content}

Return ONLY the JSON array."""
        struct_raw = call_llm(struct_prompt, model=self.model, max_tokens=1024)
        struct = parse_json_response(struct_raw)
        if isinstance(struct, list):
            struct_json = json.dumps(struct, indent=2, ensure_ascii=False)
        else:
            struct_json = json.dumps([], indent=2)
        artifacts.append(self._save_artifact(context["run_dir"], "results.json", struct_json))

        return {"content": content, "artifacts": artifacts}
