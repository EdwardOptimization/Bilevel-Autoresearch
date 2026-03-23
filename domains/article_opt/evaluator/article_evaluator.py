"""Article quality evaluator — isolated from lesson memory.

DESIGN PRINCIPLE: This evaluator NEVER receives inner lessons or skills.
It scores only the current article against the fixed 5-dimension rubric.
Isolation is what makes the score a reliable signal for both inner and outer loops.
"""

from core.llm_client import call_llm, parse_json_response

EVALUATOR_SYSTEM = """You are a rigorous academic editor evaluating an article.
Score objectively. Do NOT be lenient — a 7/10 means genuinely good, not average.
Identify specific weaknesses with quotes. Respond with valid JSON only."""

RUBRIC = """## Scoring Rubric

**A: Argumentative Rigor** (0–10, target ≥8)
10: Every claim has an explicit derivation chain or citation; no logical gaps.
8: Most claims supported; minor gaps that don't undermine the argument.
6: Key claims have support but 1-2 important jumps require reader to accept on faith.
4: Multiple unsupported claims; argument depends on reader goodwill.
2: Core thesis is asserted, not argued.

**B: Conceptual Clarity** (0–10, target ≥8)
10: All key terms defined at first use; no ambiguity possible.
8: Key terms clear; minor ambiguity in secondary concepts.
6: Central concept clear but 1-2 important terms undefined or inconsistently used.
4: Reader must guess the meaning of important terms.
2: Core concept undefined; different readers will understand different things.

**C: Cross-Article Consistency** (0–10, target ≥8)
10: No contradictions; cross-references explicit and accurate.
8: Consistent overall; minor omissions in cross-referencing.
6: Mostly consistent but 1 claim contradicts or is inconsistent with companion articles.
4: Notable contradictions or missing cross-references for central claims.
2: Significant contradictions with companion articles.
(Note: if this is a standalone article without companions, score C based on internal consistency.)

**D: Insight Novelty** (0–10, target ≥7)
10: Contains at least one insight that is genuinely surprising and non-derivable in advance.
8: Main insight is non-obvious; clearly adds something the reader didn't know.
6: Insight present but either buried or derivable from first principles in a few minutes.
4: Article mostly restates established knowledge with light synthesis.
2: No novel insight; reader learns nothing new.

**E: Actionability** (0–10, target ≥7)
10: Explicit, concrete implications for practice AND future research.
8: Clear implications for at least one audience (practitioner or researcher).
6: Implications present but vague ("future work should explore this").
4: Article ends without clear implication.
2: Reader cannot determine what to do with the information."""


class ArticleEvaluator:
    """
    Evaluates a complete revised article against the 5-dimension rubric.
    Never receives lesson memory — isolation is mandatory.
    """

    def __init__(self, model: str = ""):
        self.model = model

    def evaluate(self, article: str, article_id: str) -> dict:
        """
        Score the article on dimensions A–E.
        Returns dict with keys: scores (A-E), overall, feedback, weaknesses, verdict.
        """
        prompt = f"""## Article to Evaluate (ID: {article_id})
---
{article}
---

{RUBRIC}

Evaluate this article on all 5 dimensions.

Return ONLY this JSON:
{{
  "scores": {{"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}},
  "overall": 0.0,
  "verdict": "weak|pass|strong",
  "feedback": {{
    "A": {{"score": 0, "strength": "...", "weakness": "...", "quote": "..."}},
    "B": {{"score": 0, "strength": "...", "weakness": "...", "quote": "..."}},
    "C": {{"score": 0, "strength": "...", "weakness": "...", "quote": "..."}},
    "D": {{"score": 0, "strength": "...", "weakness": "...", "quote": "..."}},
    "E": {{"score": 0, "strength": "...", "weakness": "...", "quote": "..."}}
  }},
  "below_threshold": ["A", "B"],
  "summary": "One paragraph overall assessment."
}}

For "overall": compute the average of all 5 scores (1 decimal place).
For "verdict": "weak" if overall < 6, "pass" if 6-7, "strong" if ≥ 8.
For "below_threshold": list dimensions where score is below target (A,B,C < 8; D,E < 7)."""

        # MiniMax M2.7-highspeed is a reasoning model: <think> blocks consume
        # many tokens before the actual JSON output begins.  Use a large budget.
        raw = call_llm(prompt, system=EVALUATOR_SYSTEM, model=self.model, max_tokens=6000)
        result = parse_json_response(raw)

        if not isinstance(result, dict) or "raw_content" in result:
            return self._fallback()

        # Coerce scores to int
        scores = result.get("scores", {})
        for dim in "ABCDE":
            scores[dim] = int(scores.get(dim, 5))
        result["scores"] = scores

        # Compute overall if missing or wrong
        if scores:
            result["overall"] = round(sum(scores.values()) / len(scores), 1)

        result.setdefault("verdict", "weak" if result["overall"] < 6 else
                          "pass" if result["overall"] < 8 else "strong")
        result.setdefault("below_threshold", [])
        result.setdefault("summary", "")

        return result

    def _fallback(self) -> dict:
        return {
            "scores": {"A": 5, "B": 5, "C": 5, "D": 5, "E": 5},
            "overall": 5.0,
            "verdict": "weak",
            "feedback": {},
            "below_threshold": ["A", "B", "C", "D", "E"],
            "summary": "Evaluation parsing failed.",
        }
