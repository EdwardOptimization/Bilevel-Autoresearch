"""Stage-level evaluator.

DESIGN PRINCIPLE: This evaluator does NOT receive lesson memory.
It evaluates only the current stage output against the rubric.
Memory isolation is maintained by never passing lesson context here.
"""
from ..llm_client import call_llm, parse_json_response
from .rubric import GLOBAL_RUBRIC, STAGE_RUBRICS

EVALUATOR_SYSTEM = """You are a rigorous research pipeline evaluator.
Evaluate the given stage output objectively based on the provided rubric.
Do NOT be lenient. Identify real weaknesses.
Respond with valid JSON only."""


class StageEvaluator:
    def __init__(self, model: str = ""):
        self.model = model

    def evaluate(self, stage_name: str, stage_output: str, topic: str) -> dict:
        """
        Evaluate a single pipeline stage output.

        NOTE: This method intentionally receives NO lesson memory.
        The evaluator must remain isolated from the evolution layer.
        """
        rubric = STAGE_RUBRICS.get(stage_name, "Evaluate the quality and usefulness of this output.")

        prompt = f"""## Research Topic
{topic}

## Stage Being Evaluated
{stage_name}

## Stage Output
{stage_output}

## Evaluation Rubric
{rubric}

{GLOBAL_RUBRIC}

Return ONLY the JSON object."""

        raw = call_llm(prompt, system=EVALUATOR_SYSTEM, model=self.model)
        result = parse_json_response(raw)

        if isinstance(result, dict) and "raw_content" not in result:
            # Ensure required fields exist; coerce score to int (LLM may return "7" as str)
            result.setdefault("verdict", "weak")
            result["score"] = int(result.get("score", 5))
            result.setdefault("feedback", "Evaluation could not be parsed cleanly.")
            result.setdefault("strengths", [])
            result.setdefault("weaknesses", [])
            return result

        return {
            "verdict": "weak",
            "score": 5,
            "feedback": "Evaluator returned unparseable response.",
            "strengths": [],
            "weaknesses": ["Evaluation parsing failed"],
        }
