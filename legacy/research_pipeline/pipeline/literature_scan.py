"""Stage A: Literature scan."""
import json

from ..llm_client import call_llm, parse_json_response
from .base import BaseStage

SYSTEM = """You are a research librarian conducting a thorough literature scan.
Your goal: identify key existing work, open questions, and knowledge gaps relevant to the topic.
Be specific — name papers, authors, findings where possible (even if simulated for MVP purposes).
Do not be vague. Do not list generic categories. Be concrete.

COMPLETENESS RULE: You MUST complete every section, including the last knowledge gap.
If approaching token limits, write shorter entries — but COMPLETE all items.
Never end mid-sentence or leave a section incomplete."""


class LiteratureScanStage(BaseStage):
    name = "literature_scan"

    def run(self, context: dict) -> dict:
        topic = context["topic"]
        # Skills are structural guidance (HOW to do the scan) — safe to inject even for Stage A
        # Raw lessons are NOT injected here to avoid topic bias on content discovery
        skill_text = context.get("retrieved_lessons", "")
        skill_section = f"\n{skill_text}\n" if skill_text else ""

        prompt = f"""## Research Topic
{topic}
{skill_section}
## Your Task
Conduct a comprehensive literature scan on this topic.

Produce:
1. **Key Papers / Work**: List 6-10 relevant papers or bodies of work with specific citations
   (author, year, venue, key finding). Include seminal works AND recent advances (2022-2024).
2. **Synthesis Table**: A markdown table with columns: Method/Approach | Key Finding | Gap/Limitation
3. **Core Findings**: What is already established? (be specific, cite sources)
4. **Open Questions**: What remains unresolved? (3-5 concrete questions)
5. **Knowledge Gaps**: What is clearly missing from the literature? (explicitly note gaps)
6. **Relevant Methods**: What experimental approaches have been used?

Note: This is an MVP pipeline. Use known real papers where possible; for gaps,
indicate "[estimated from trends]". Prioritize rigor and specificity over breadth.
Use consistent citation format throughout: Author et al. (year), "Title", Venue."""

        content = call_llm(prompt, system=SYSTEM, model=self.model, max_tokens=8000)

        # Save main output
        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "literature_notes.md", content))

        # Save citations as structured JSON (extracted from content)
        citations_prompt = f"""Extract citations from this literature scan and return as JSON array.
Each item: {{"title": "...", "authors": "...", "year": "...", "key_finding": "..."}}

Literature scan:
{content}

Return ONLY the JSON array."""
        citations_raw = call_llm(citations_prompt, model=self.model, max_tokens=1024)
        citations = parse_json_response(citations_raw)
        if isinstance(citations, list):
            citations_json = json.dumps(citations, indent=2, ensure_ascii=False)
        else:
            citations_json = json.dumps([], indent=2)
        artifacts.append(self._save_artifact(context["run_dir"], "citations.json", citations_json))

        return {"content": content, "artifacts": artifacts}
