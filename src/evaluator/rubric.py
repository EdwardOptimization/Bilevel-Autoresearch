"""Stage-specific evaluation rubrics.

These criteria are given to the evaluator LLM for each stage.
The evaluator is ISOLATED from lesson memory — rubrics must never reference
historical lessons or memory store content.
"""

STAGE_RUBRICS: dict[str, str] = {
    "literature_scan": """
Evaluate this literature scan output on:
1. Coverage: Does it identify 6+ key relevant works, including recent advances?
2. Specificity: Are papers/sources cited with enough detail to be useful?
3. Synthesis: Does it include a synthesis table mapping Method/Approach → Key Finding → Gap/Limitation? Is the table complete with 6+ rows?
4. Gaps: Does it explicitly identify what is NOT yet known, with specific gap statements?
5. Relevance: Is content tightly focused on the research topic?

NOTE: A truncated output that cuts off mid-sentence or is missing major sections should score ≤5.
""",
    "hypothesis_generation": """
Evaluate this hypothesis generation output on:
1. Completeness: Are ALL hypotheses fully stated with no truncation? A hypothesis cut off mid-sentence is a critical failure.
2. Measurability: Does EACH hypothesis include: (a) named statistical test, (b) effect size threshold, (c) explicit confirmation AND refutation criteria?
3. Operationalization: Are independent and dependent variables specified with concrete measurement methods?
4. Grounding: Are hypotheses informed by the literature scan with specific connections?
5. Consistency: Are constructs (e.g., "task complexity") defined consistently across all hypotheses?

CRITICAL: If any hypothesis is truncated or missing evaluation criteria, score ≤5 (weak).
A score of 8+ requires ALL hypotheses to be complete with statistical test specifications.
""",
    "experiment_plan_or_code": """
Evaluate this experiment plan/code on:
1. Coverage: Does the plan address EVERY hypothesis with specific experiments? Missing hypotheses = ≤5 score.
2. Code completeness: If code is included, is it complete and executable? Truncated functions = critical failure.
3. Metric validity: Does it avoid BLEU/BERTScore for evaluating research quality? Uses construct-valid metrics?
4. Controls: Are comparison baselines or controls specified for each experiment?
5. Methodology rigor: Are procedures specific enough to reproduce?

CRITICAL: Incomplete code (functions defined but not implemented, or task corpus partially defined) should score ≤6.
""",
    "experiment_result_summary": """
Evaluate this result summary on:
1. Completeness: Are results reported for ALL hypotheses with no missing entries?
2. Quantification: Are results described with specific numbers and uncertainty ranges (e.g., "34.2% ± 4.1%")?
3. Honesty: Are negative results and unexpected observations included (≥3 anomalies expected)?
4. Hypothesis linkage: Are results explicitly tied to confirmation/refutation of each hypothesis?
5. Limitations: Are confounds, threats to validity, and reliability concerns acknowledged?
""",
    "draft_writeup": """
Evaluate this draft writeup on:
1. Completeness: Are ALL 8 sections present and non-truncated? (Abstract, Introduction, Related Work, Methodology, Results, Discussion, Conclusion, References)
2. Scientific integrity: Does it correctly use simulation-appropriate language? Reward "our simulation suggests..." — penalize p-values or F-statistics on simulated data. No false empirical claims.
3. Framing consistency: Is the paper framed as a methodological/simulation study throughout? "Empirical evaluation" language should NOT appear if results are simulated.
4. Coherence: Does it flow logically from problem → method → simulated results → theoretical implications?
5. Contribution: Does it clearly articulate methodological contributions and what empirical validation is still needed?

SCORING GUIDANCE:
- Score 8-9: Well-framed simulation study with consistent language, all sections complete, appropriate uncertainty expressed
- Score 6-7: Most sections complete, minor framing inconsistencies or one section is shallow
- Score ≤5: Critical integrity violation (empirical framing for simulated data), or truncated/missing sections

NOTE: A section that ends mid-sentence or is clearly incomplete should lower the score significantly.
""",
}

GLOBAL_RUBRIC = """
Score the stage from 0-10:
- 0-3: fail — major deficiencies, output is not useful or severely incomplete
- 4-5: weak — output has a critical gap: truncated content, missing required sections, or fundamental methodological flaw
- 6: weak — output exists but has significant gaps that limit usefulness
- 7-8: pass — output is adequate to good with only minor issues; all required elements present
- 9-10: pass — output is thorough, high quality, and complete in all respects

IMPORTANT SCORING NOTES:
- Any truncated output (ending mid-sentence, mid-function, or mid-hypothesis) = score ≤ 5
- Missing a required hypothesis, experiment, or section = score ≤ 5
- All required elements present but shallow = score 6-7
- All required elements present and detailed = score 7-8
- Exceptional depth, novel insights, rigorous methodology = score 9-10

Return your evaluation as JSON with these fields:
{
  "verdict": "<pass|weak|fail>",
  "score": <0-10>,
  "feedback": "<2-3 sentence overall assessment>",
  "strengths": ["<strength 1>", ...],
  "weaknesses": ["<weakness 1>", ...]
}

For verdict: score ≥ 7 = "pass", score 4-6 = "weak", score ≤ 3 = "fail"
"""
