"""Stage E: Revised Output — apply the edit plan and produce the full revised article.

Outputs the COMPLETE revised article text (not a diff).
The next run's article_content will be set to this output.

Section-by-section generation when article is long (>3000 chars), to avoid
truncation — same pattern as V1's draft_writeup.py section-by-section approach.
"""
import re

from ..llm_client import call_llm
from .base import BaseStage

SYSTEM = """You are a technical editor applying a revision plan to an article.
Apply ONLY the edits specified in the plan. Do NOT:
- Add new ideas not in the plan
- Remove content not mentioned in the plan
- Change the article's voice or structure beyond what is specified
- Add filler phrases like "In conclusion" or "As we can see"

Preserve the exact meaning of all unchanged sections.
Output the COMPLETE revised article — every section, start to finish."""

SECTION_SYSTEM = """You are a technical editor revising ONE section of an article.
Apply ONLY the edits that target this section.
Preserve all other text in this section exactly.
Output the COMPLETE revised section text only — no headers like "Here is the revised section:"."""


def _extract_sections(article: str) -> list[tuple[str, str]]:
    """Split article into (header, body) pairs. Returns [(header, body), ...]."""
    # Match markdown headers (# ## ###)
    pattern = re.compile(r"^(#{1,3}\s+.+)$", re.MULTILINE)
    positions = [(m.start(), m.group(0)) for m in pattern.finditer(article)]

    if not positions:
        return [("", article)]

    sections = []
    for i, (pos, header) in enumerate(positions):
        start = pos + len(header) + 1  # skip newline after header
        end = positions[i + 1][0] if i + 1 < len(positions) else len(article)
        body = article[start:end].strip()
        sections.append((header, body))

    # Prepend any content before first header
    if positions[0][0] > 0:
        preamble = article[:positions[0][0]].strip()
        if preamble:
            sections.insert(0, ("", preamble))

    return sections


class RevisedOutputStage(BaseStage):
    name = "revised_output"

    def run(self, context: dict) -> dict:
        article = context["article_content"]
        edit_plan = context["previous_outputs"].get("edit_planning", "")
        impact = context["previous_outputs"].get("impact_assessment", "")
        retry_feedback = context.get("evaluator_feedback", "")
        retry_section = f"\n## Previous Attempt Feedback\n{retry_feedback}\n" if retry_feedback else ""

        plan_excerpt = edit_plan[:3500] if len(edit_plan) > 3500 else edit_plan
        impact_excerpt = impact[:1500] if len(impact) > 1500 else impact

        if len(article) <= 8000:
            revised = self._revise_whole(article, plan_excerpt, impact_excerpt, retry_section)
        else:
            revised = self._revise_by_section(article, plan_excerpt, impact_excerpt, retry_section)

        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "revised_article.md", revised))

        return {"content": revised, "artifacts": artifacts, "revised_article": revised}

    def _revise_whole(self, article: str, plan: str, impact: str, retry: str) -> str:
        prompt = f"""## Original Article
---
{article}
---

## Edit Plan to Apply
{plan}

## Impact Assessment Notes
{impact}
{retry}
## Instructions
Apply the edit plan to the article above.
Output the COMPLETE revised article — every word, from title to last paragraph."""

        return call_llm(prompt, system=SYSTEM, model=self.model, max_tokens=8000)

    def _revise_by_section(self, article: str, plan: str, impact: str, retry: str) -> str:
        sections = _extract_sections(article)
        revised_sections = []

        for header, body in sections:
            # Check if this section is targeted by the edit plan
            header_lower = header.lower()
            plan_mentions_section = (
                header_lower in plan.lower() or
                (body[:80].lower() in plan.lower() if body else False)
            )

            if plan_mentions_section or not header:
                # Revise this section
                section_prompt = f"""## Full Edit Plan (for context)
{plan[:2000]}

## Section Being Revised
{header}
{body}

## Instruction
Apply ONLY the edits from the plan that target this section.
If no edits target this section, output the section UNCHANGED.
Output the section text only (include the header if present)."""

                revised_body = call_llm(
                    section_prompt, system=SECTION_SYSTEM,
                    model=self.model, max_tokens=3000
                )
                revised_sections.append(revised_body)
            else:
                # No edits target this section — keep verbatim
                if header:
                    revised_sections.append(f"{header}\n\n{body}")
                else:
                    revised_sections.append(body)

        return "\n\n".join(revised_sections)
