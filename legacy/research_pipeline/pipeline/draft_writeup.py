"""Stage E: Draft writeup — section-by-section generation to prevent truncation.

Each section is generated in a separate LLM call, so no section gets cut off.
Lesson injection applied as warnings before generation.
"""
from ..llm_client import call_llm
from .base import BaseStage

# Each tuple: (section_id, section_title, word_guidance, max_tokens)
# Token limits include generous overhead for reasoning models (<think> blocks count against max_tokens).
# Rule of thumb: visible output tokens + ~800 thinking overhead tokens.
SECTIONS = [
    ("abstract",      "Abstract",      "3–4 sentences (~150 words) summarizing motivation, method, key finding, and implication.", 1200),
    ("introduction",  "Introduction",  "3–4 paragraphs (~400 words): background, problem statement, motivation, and scope.", 2000),
    ("related_work",  "Related Work",  "3–4 paragraphs (~400 words) synthesizing related work. ONLY cite papers from the provided Literature Scan — do NOT add new citations from memory.", 2000),
    ("methodology",   "Methodology",   "4–5 paragraphs (~500 words): experimental design, procedures, metrics (NOT BLEU/BERTScore), baselines.", 2500),
    ("results",       "Results",       "4–5 paragraphs (~500 words): simulated findings per hypothesis. Do NOT report F-statistics, t-statistics, or p-values for simulated data. Use approximate ranges (e.g., 'roughly 5–10%') not exact numbers.", 2500),
    ("discussion",    "Discussion",    "3–4 paragraphs (~400 words): interpretation of simulated patterns, limitations, threats to validity, future empirical work needed.", 2000),
    ("conclusion",    "Conclusion",    "2–3 paragraphs (~250 words): key takeaways as methodological contributions (not empirical findings), next steps including needed empirical validation.", 1800),
    ("references",    "References",    "8–12 references from the provided Literature Scan only, in numbered APA/ACL format. Do NOT add papers not in the context.", 1500),
]

SECTION_SYSTEM = """You are a research writer producing one specific section of a research report.
Write ONLY the requested section. Be concrete and specific. Write complete sentences and complete paragraphs.
Do not add section headers from other parts of the paper. Do not summarize other sections.

CRITICAL SCIENTIFIC INTEGRITY RULES:
1. If the experimental results are SIMULATED/HYPOTHETICAL (as described in context), you MUST:
   - Frame the paper as a "methodological framework" or "simulation study" — NOT an "empirical evaluation"
   - Use language like "our simulation suggests..." or "the model predicts..." — NEVER "our experiments show..."
   - DO NOT report precise F-statistics, p-values, t-statistics, or bootstrapped CIs — simulated data cannot support these
   - Use approximate effect descriptions: "roughly X% improvement" rather than "71.4 ± 9.3"
   - Include a brief disclosure: "Note: These findings are based on simulation; empirical validation is needed."
2. CITATION DISCIPLINE: Only cite papers that appear in the provided Literature Scan.
   Do not add papers from your training knowledge. If you need more citations, use the ones in context.
3. Always write complete paragraphs — never stop mid-sentence. If approaching length limit,
   complete your current paragraph before stopping."""


class DraftWriteupStage(BaseStage):
    name = "draft_writeup"

    def run(self, context: dict) -> dict:
        topic = context["topic"]
        lit_scan   = context["previous_outputs"].get("literature_scan", "")
        hypotheses = context["previous_outputs"].get("hypothesis_generation", "")
        experiment = context["previous_outputs"].get("experiment_plan_or_code", "")
        results    = context["previous_outputs"].get("experiment_result_summary", "")
        lessons_text = context.get("retrieved_lessons", "")
        evaluator_feedback = context.get("evaluator_feedback", "")

        # Shared context block passed to each section LLM call
        # Generous limits: writeup quality depends on having full context
        shared_context = f"""## Research Topic
{topic}

## Literature Scan (summary)
{lit_scan[:2000]}

## Hypotheses
{hypotheses[:3000]}

## Experiment Design (summary)
{experiment[:1500]}

## Results (simulated)
{results[:3000]}"""

        warnings = ""
        if lessons_text:
            warnings = f"\n## Prior-Run Warnings (Advisory)\n{lessons_text}\n"
        if evaluator_feedback:
            warnings += f"\n## Evaluator Feedback to Address\n{evaluator_feedback}\n"

        # Generate each section independently
        sections: dict[str, str] = {}
        for sec_id, sec_title, guidance, sec_max_tokens in SECTIONS:
            prompt = f"""{shared_context}
{warnings}
---
Write ONLY the **{sec_title}** section of the research report.
Guidance: {guidance}

Format: Start directly with the content (no "## {sec_title}" header needed — it will be added automatically).
Be specific. Reference actual hypotheses, results, and literature from the context above.
IMPORTANT: Write complete sentences and complete paragraphs. Do not truncate."""

            section_content = call_llm(
                prompt, system=SECTION_SYSTEM, model=self.model, max_tokens=sec_max_tokens
            )
            sections[sec_id] = section_content.strip()

        # Assemble full draft
        title_prompt = f"""Given this research topic, write a concise, informative paper title (max 12 words):

Topic: {topic}
Abstract: {sections.get('abstract', '')[:300]}

Return only the title, nothing else."""
        title = call_llm(title_prompt, model=self.model, max_tokens=50).strip().strip('"').strip()

        full_draft = _assemble_draft(title, sections)

        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "draft.md", full_draft))

        # Save sections separately for inspection
        for sec_id, content in sections.items():
            artifacts.append(
                self._save_artifact(context["run_dir"], f"sections/{sec_id}.md", content)
            )

        return {"content": full_draft, "artifacts": artifacts}


def _assemble_draft(title: str, sections: dict[str, str]) -> str:
    parts = [f"# {title}", ""]
    for sec_id, sec_title, _, _max_tok in SECTIONS:
        content = sections.get(sec_id, "")
        if content:
            parts.append(f"## {sec_title}")
            parts.append(content)
            parts.append("")
    return "\n".join(parts)
