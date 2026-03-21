"""Stage A: Article Analysis — identify weaknesses against the 5-dimension rubric.

No lesson injection here. This stage must see the article fresh to avoid bias
toward what past runs thought was weak. Skills ARE injected (HOW to analyze,
not WHAT to find).
"""
import json

from ..llm_client import call_llm, parse_json_response
from .base import BaseStage

SYSTEM = """You are a rigorous academic editor analyzing an article.
Your job: identify concrete, specific weaknesses in the article against a 5-dimension rubric.
Be specific — quote exact sentences or paragraphs that are weak. Do not be vague.
A weakness like "the argument could be clearer" is useless. Name the specific claim,
the specific section, and the specific reason it fails the rubric criterion.

COMPLETENESS RULE: Cover all 5 dimensions (A through E). Never skip a dimension.
If a dimension is strong, say so briefly — but still evaluate it."""


RUBRIC = """## Evaluation Rubric (5 Dimensions)

**A: Argumentative Rigor** (target ≥8/10)
Every core claim must have a clear derivation chain or empirical support.
Weakness: claims that require the reader to accept on faith, logical jumps,
missing intermediate steps, or unsupported generalizations.

**B: Conceptual Clarity** (target ≥8/10)
Key terms must be defined; the reader cannot misinterpret the central concept.
Weakness: undefined jargon, ambiguous pronouns for key concepts, terms used
inconsistently across the article.

**C: Cross-Article Consistency** (target ≥8/10)
No contradictions with the companion articles (Article 1 and Article 2).
Cross-references must be accurate and bidirectional.
Weakness: claims that contradict the companion articles, missing cross-references
where the companion articles use the same concept differently.

**D: Insight Novelty** (target ≥7/10)
The article says something non-obvious that the reader cannot derive in 5 minutes
from first principles.
Weakness: the "insight" is just restating what the reader already knows, or the
novel claim is buried and never made explicit.

**E: Actionability** (target ≥7/10)
The reader knows what to do or how to update their thinking after reading.
Weakness: the article ends without a clear implication for practice or future work,
or the implication is too vague to act on."""


class ArticleAnalysisStage(BaseStage):
    name = "article_analysis"

    def run(self, context: dict) -> dict:
        article = context["article_content"]
        skill_text = context.get("retrieved_lessons", "")  # HOW to analyze (structural guidance)
        skill_section = f"\n{skill_text}\n" if skill_text else ""
        retry_feedback = context.get("evaluator_feedback", "")
        retry_section = f"\n## Previous Attempt Feedback\n{retry_feedback}\n" if retry_feedback else ""

        prompt = f"""## Article to Analyze
---
{article}
---

{RUBRIC}
{skill_section}{retry_section}
## Your Task
Analyze the article above against the 5 rubric dimensions.

For each dimension provide:
1. **Current Score Estimate** (0–10): Your assessment of the current state
2. **Specific Weaknesses** (quote the problematic text):
   - Location: section name or first few words of the relevant paragraph
   - Quote: the exact problematic sentence or claim
   - Problem: why this fails the rubric criterion
3. **Specific Strengths**: What the article already does well on this dimension

After all 5 dimensions, provide:
- **Priority Ranking**: Which 2 dimensions have the most room for improvement?
- **Overall Assessment**: In 2 sentences, what is the single biggest barrier
  to this article reaching 8/10 across all dimensions?"""

        content = call_llm(prompt, system=SYSTEM, model=self.model, max_tokens=8000)

        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "analysis.md", content))

        # Extract structured scores
        score_prompt = f"""Extract dimension scores from this article analysis.
Return JSON with keys A, B, C, D, E (integers 0-10) and a "priority_dimensions" array
(2 dimension letters with most room for improvement).

Analysis:
{content[:3000]}

Return ONLY the JSON object."""
        score_raw = call_llm(score_prompt, model=self.model, max_tokens=2000)
        scores = parse_json_response(score_raw)
        if isinstance(scores, dict) and "raw_content" not in scores:
            artifacts.append(self._save_artifact(
                context["run_dir"], "scores.json",
                json.dumps(scores, indent=2, ensure_ascii=False)
            ))

        return {"content": content, "artifacts": artifacts, "scores": scores}
