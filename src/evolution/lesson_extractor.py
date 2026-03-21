"""Extract structured lessons from a completed run's review."""
import json

from ..llm_client import call_llm, parse_json_response
from .lesson_schema import Lesson, make_lesson_id

EXTRACTION_SYSTEM = """You are a research experience analyst. Your job is to extract structured,
actionable lessons from a completed research pipeline run.

Each lesson must be grounded in specific evidence from the run. Do NOT generate vague platitudes.
Every lesson must have a concrete reuse_rule — an actionable instruction for the next run.

You MUST respond with valid JSON only (no markdown, no explanation outside the JSON)."""


EXTRACTION_PROMPT = """Analyze the following research run and extract 3-7 structured lessons.

## Run ID
{run_id}

## Topic
{topic}

## Stage Verdicts
{stage_verdicts}

## Run Review
{run_review}

## Stage Outputs Summary
{stage_outputs_summary}

---

Extract lessons as a JSON array. Each lesson must follow this exact schema:

{{
  "lesson_type": "<one of: failure_pattern, successful_pattern, decision, warning, guardrail>",
  "scope": "<one of: topic, stage, global>",
  "stage": "<pipeline stage this lesson applies to, e.g. hypothesis_generation>",
  "topic_tags": ["<tag1>", "<tag2>"],
  "summary": "<one sentence: what happened and why it matters>",
  "evidence": ["<specific artifact or observation from this run>"],
  "confidence": <0.0 to 1.0>,
  "reuse_rule": "<concrete actionable instruction for the next run>",
  "anti_pattern": "<what to avoid — leave empty string if not a failure_pattern>"
}}

Return ONLY the JSON array, nothing else."""


class LessonExtractor:
    def __init__(self, model: str = ""):
        self.model = model

    def extract(
        self,
        run_id: str,
        topic: str,
        run_review: dict,
        stage_verdicts: dict,
        stage_outputs: dict,
    ) -> list[Lesson]:
        """Extract structured lessons from run artifacts."""
        stage_outputs_summary = self._summarize_outputs(stage_outputs)

        prompt = EXTRACTION_PROMPT.format(
            run_id=run_id,
            topic=topic,
            stage_verdicts=json.dumps(stage_verdicts, indent=2, ensure_ascii=False),
            run_review=json.dumps(run_review, indent=2, ensure_ascii=False),
            stage_outputs_summary=stage_outputs_summary,
        )

        raw = call_llm(prompt, system=EXTRACTION_SYSTEM, model=self.model)
        parsed = parse_json_response(raw)

        if isinstance(parsed, dict) and "raw_content" in parsed:
            # LLM didn't return valid JSON — skip extraction gracefully
            return []

        lessons = []
        items = parsed if isinstance(parsed, list) else [parsed]
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                lesson = Lesson(
                    id=make_lesson_id(),
                    created_from_run=run_id,
                    lesson_type=item.get("lesson_type", "warning"),
                    scope=item.get("scope", "stage"),
                    stage=item.get("stage", "unknown"),
                    topic_tags=item.get("topic_tags", []),
                    summary=item.get("summary", ""),
                    evidence=item.get("evidence", []),
                    confidence=float(item.get("confidence", 0.5)),
                    reuse_rule=item.get("reuse_rule", ""),
                    anti_pattern=item.get("anti_pattern", ""),
                )
                if lesson.summary and lesson.reuse_rule:
                    lessons.append(lesson)
            except (TypeError, ValueError):
                continue

        return lessons

    def _summarize_outputs(self, stage_outputs: dict) -> str:
        """Create a brief summary of stage outputs for the extraction prompt."""
        lines = []
        for stage, output in stage_outputs.items():
            content = output.get("content", "")
            # Truncate long content
            if len(content) > 800:
                content = content[:800] + "... [truncated]"
            lines.append(f"### {stage}\n{content}\n")
        return "\n".join(lines)
