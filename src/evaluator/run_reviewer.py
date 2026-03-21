"""Whole-run reviewer.

DESIGN PRINCIPLE: Like StageEvaluator, this reviewer does NOT receive lesson memory.
It reviews the entire run based solely on current run artifacts and stage verdicts.
"""
import json

from ..llm_client import call_llm, parse_json_response

REVIEWER_SYSTEM = """You are a senior research advisor reviewing a complete research pipeline run.
Assess the run holistically: did it make genuine progress on the topic?
Identify failures, successes, and weak points clearly.
Respond with valid JSON only."""


REVIEWER_PROMPT = """## Research Topic
{topic}

## Stage Verdicts
{stage_verdicts}

## Stage Outputs
{stage_outputs}

---

Review this complete research pipeline run. Return a JSON object with:
{{
  "overall_verdict": "<pass|weak|fail>",
  "score": <0-10>,
  "summary": "<3-4 sentence overall assessment of the run>",
  "failure_points": ["<specific failure 1>", ...],
  "success_points": ["<specific success 1>", ...],
  "weak_points": ["<area that needs improvement>", ...],
  "recommendations": ["<actionable recommendation for next run>", ...]
}}

Be specific. Reference actual content from the stage outputs.
Return ONLY the JSON object."""


class RunReviewer:
    def __init__(self, model: str = ""):
        self.model = model

    def review(
        self,
        topic: str,
        stage_verdicts: dict[str, dict],
        stage_outputs: dict[str, str],
    ) -> dict:
        """
        Review the complete run.

        NOTE: This method intentionally receives NO lesson memory.
        The reviewer evaluates only the current run artifacts.
        """
        # Truncate stage outputs to avoid exceeding context
        truncated_outputs = {}
        for stage, content in stage_outputs.items():
            if len(content) > 2500:
                truncated_outputs[stage] = content[:2500] + "\n... [truncated for review]"
            else:
                truncated_outputs[stage] = content

        prompt = REVIEWER_PROMPT.format(
            topic=topic,
            stage_verdicts=json.dumps(stage_verdicts, indent=2, ensure_ascii=False),
            stage_outputs=json.dumps(truncated_outputs, indent=2, ensure_ascii=False),
        )

        raw = call_llm(prompt, system=REVIEWER_SYSTEM, model=self.model)
        result = parse_json_response(raw)

        if isinstance(result, dict) and "raw_content" not in result:
            result.setdefault("overall_verdict", "weak")
            result.setdefault("score", 5)
            result.setdefault("summary", "")
            result.setdefault("failure_points", [])
            result.setdefault("success_points", [])
            result.setdefault("weak_points", [])
            result.setdefault("recommendations", [])
            return result

        return {
            "overall_verdict": "weak",
            "score": 5,
            "summary": "Run reviewer returned unparseable response.",
            "failure_points": [],
            "success_points": [],
            "weak_points": ["Review parsing failed"],
            "recommendations": [],
        }
