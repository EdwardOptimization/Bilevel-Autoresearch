"""
Test script: GLM5 vs DeepSeek for outer loop suitability.

The outer loop needs to:
  1. Analyze an inner cycle trace and identify root causes of slow convergence
  2. Select an appropriate strategy from reference_frameworks.md
  3. Propose specific, actionable pipeline config changes
  4. Output structured outer-loop lessons (JSON)

We test both models on the same task and compare:
  - Reasoning quality (does it correctly identify root causes?)
  - Specificity (does it propose concrete config changes?)
  - JSON structure compliance (does it output valid, well-formed JSON?)
  - Framework reference (does it use reference_frameworks.md vocabulary?)
  - Token efficiency (how verbose is the output?)

Usage:
  # Set keys first:
  export $(grep -v '^#' ~/.env | xargs)
  # Run with both models:
  python article_optimizer/test_outer_model.py --models glm deepseek
  # Run with just glm:
  python article_optimizer/test_outer_model.py --models glm
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Add parent src to path so we reuse the shared llm_client
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from llm_client import call_llm, configure, parse_json_response

# ── Test payload: a realistic inner cycle trace ───────────────────────────────
# Based on the V1 science pipeline runs 8-12. Realistic enough to test reasoning.

INNER_TRACE = """
# Outer Loop Context — Cycle 1

## Inner Cycle Summary
- Article: article15_meta_optimization
- Total runs: 12
- Peak score: 7/10
- Runs to reach 8/10: never
- Converged: False

## Convergence Trace
run | A | B | C | D | E | overall
----|---|---|---|---|---|--------
  1 | 6 | 6 | 5 | 5 | 5 | 5
  2 | 7 | 6 | 5 | 6 | 5 | 6
  3 | 7 | 7 | 5 | 6 | 6 | 6
  4 | 7 | 7 | 6 | 6 | 6 | 6
  5 | 7 | 7 | 5 | 7 | 6 | 6
  6 | 7 | 7 | 5 | 7 | 7 | 7
  7 | 7 | 7 | 6 | 7 | 7 | 7
  8 | 7 | 8 | 5 | 7 | 7 | 7
  9 | 7 | 8 | 4 | 7 | 7 | 7
 10 | 8 | 8 | 5 | 7 | 7 | 7
 11 | 8 | 8 | 5 | 7 | 7 | 7
 12 | 8 | 8 | 5 | 7 | 7 | 7

## Stage Failure Patterns
- Stage A: mean=7.17, std=0.58, retries=0, range=[6,8]
- Stage B: mean=7.17, std=0.72, retries=1, range=[6,8]
- Stage C: mean=5.08, std=0.49, retries=3, range=[4,6]
- Stage D: mean=6.5, std=0.67, retries=0, range=[5,7]
- Stage E: mean=6.33, std=0.78, retries=0, range=[5,7]

## Lesson Quality
- Total lessons: 48
- High-confidence (≥0.85): 31
- Fraction: 0.65

## Prior Outer Lessons
(none — this is the first outer cycle)
"""

REFERENCE_FRAMEWORKS_EXCERPT = """
Key strategies available (from reference_frameworks.md):

1. **Reflexion** — Generate verbal failure postmortems after each failed stage;
   store in memory; inject at top of next run. Best for: high-variance stages.

2. **Self-Refine** — Add internal critique-refine loop inside a stage call.
   Best for: structurally correct but shallow output.

3. **OPRO** — Feed (prompt, score) history to a meta-LLM; ask for a better prompt.
   Best for: plateaued stages where token budget changes have no effect.

4. **DSPy** — Inject 2-3 high-scoring stage outputs as few-shot examples.
   Best for: inconsistent stages with ≥3 high-quality examples available.

5. **TextGrad** — Propagate "textual gradients" backward across stages to identify
   root-cause stage when downstream stages fail.
   Best for: cross-stage failures where one stage's bad output causes downstream issues.
"""

OUTER_SYSTEM = """You are the outer optimization loop for a dual-layer autoresearch system.
Your job: analyze the inner cycle trace and produce actionable pipeline improvements.

You must output valid JSON only — no markdown, no explanation outside the JSON.
"""

OUTER_PROMPT = f"""
{INNER_TRACE}

## Available Optimization Strategies
{REFERENCE_FRAMEWORKS_EXCERPT}

## Your Task

Analyze the inner cycle trace above and produce:
1. Root cause analysis: why did the pipeline fail to reach 8/10 in 12 runs?
2. Strategy selection: which strategy from the reference will most help?
3. Specific config changes: what exactly should change in the pipeline?
4. Outer lessons: structured lessons about the process.

Output JSON with this exact structure:
{{
  "root_cause": {{
    "primary_bottleneck_stage": "X",
    "diagnosis": "...",
    "evidence": ["...", "..."]
  }},
  "strategy_selected": {{
    "name": "...",
    "rationale": "...",
    "decision_rule_from_reference": "..."
  }},
  "config_changes": [
    {{
      "target": "stage_prompt | token_budget | rubric_sub_criteria | pipeline_structure",
      "stage": "X",
      "change": "...",
      "expected_effect": "..."
    }}
  ],
  "outer_lessons": [
    {{
      "lesson_type": "failure_pattern | strategy_effectiveness | config_change",
      "stage_affected": "X",
      "summary": "...",
      "reuse_rule": "...",
      "confidence": 0.0
    }}
  ]
}}
"""

# ── Evaluation rubric for comparing models ────────────────────────────────────

EVAL_CRITERIA = {
    "json_valid": "Output is valid parseable JSON",
    "correct_bottleneck": "Correctly identifies Stage C as primary bottleneck (consistently 5/10, highest retry rate)",
    "specific_changes": "Proposes at least 2 concrete, specific config changes (not vague 'improve the prompt')",
    "framework_reference": "References a strategy by name from the reference frameworks",
    "reuse_rule_quality": "Reuse rules are specific enough to apply in a future cycle without re-reading the trace",
    "confidence_calibrated": "Confidence values are not all 0.9+ (shows calibration)",
}


def score_response(parsed: dict, raw: str) -> dict[str, bool | str]:
    results = {}

    # JSON valid
    results["json_valid"] = isinstance(parsed, dict) and "root_cause" in parsed

    # Correct bottleneck
    bottleneck = parsed.get("root_cause", {}).get("primary_bottleneck_stage", "")
    results["correct_bottleneck"] = bottleneck.upper() == "C"

    # Specific changes
    changes = parsed.get("config_changes", [])
    specific = [c for c in changes if len(c.get("change", "")) > 30]
    results["specific_changes"] = len(specific) >= 2

    # Framework reference
    known_frameworks = ["reflexion", "self-refine", "opro", "dspy", "textgrad", "voyager"]
    strategy = parsed.get("strategy_selected", {}).get("name", "").lower()
    results["framework_reference"] = any(f in strategy for f in known_frameworks)

    # Reuse rule quality
    lessons = parsed.get("outer_lessons", [])
    good_rules = [l for l in lessons if len(l.get("reuse_rule", "")) > 40]
    results["reuse_rule_quality"] = len(good_rules) >= 1

    # Confidence calibrated
    confidences = [l.get("confidence", 0) for l in lessons]
    results["confidence_calibrated"] = len(confidences) > 0 and not all(c >= 0.9 for c in confidences)

    results["output_length"] = len(raw)
    return results


# ── Runner ────────────────────────────────────────────────────────────────────

MODEL_CONFIGS = {
    "glm": {
        "provider": "glm",
        "api_key_env": "GLM_API_KEY",
        "model": os.environ.get("GLM_MODEL", "glm-5"),
        "label": f"GLM5 ({os.environ.get('GLM_MODEL', 'glm-5')})",
    },
    "deepseek": {
        "provider": "deepseek",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
        "label": "DeepSeek v3 (deepseek-chat)",
    },
    "minimax": {
        "provider": "minimax",
        "api_key_env": "MINIMAX_API_KEY",
        "model": "MiniMax-M2.7-highspeed",
        "label": "MiniMax M2.7-highspeed",
    },
}


def run_test(model_key: str) -> dict:
    cfg = MODEL_CONFIGS[model_key]
    api_key = os.environ.get(cfg["api_key_env"], "")
    if not api_key:
        return {"error": f"Missing {cfg['api_key_env']}"}

    configure(cfg["provider"], api_key, cfg["model"])
    print(f"\n{'='*60}")
    print(f"Testing: {cfg['label']}")
    print(f"{'='*60}")

    t0 = time.time()
    try:
        raw = call_llm(OUTER_PROMPT, system=OUTER_SYSTEM, max_tokens=3000)
    except Exception as e:
        return {"error": str(e)}
    elapsed = time.time() - t0

    parsed = parse_json_response(raw)
    scores = score_response(parsed, raw)

    passed = sum(1 for k, v in scores.items() if k != "output_length" and v is True)
    total_criteria = sum(1 for k in scores if k != "output_length")

    print(f"Time: {elapsed:.1f}s | Output: {len(raw)} chars | Score: {passed}/{total_criteria}")
    print("\nCriteria:")
    for criterion, result in scores.items():
        if criterion == "output_length":
            continue
        icon = "✓" if result else "✗"
        print(f"  {icon} {criterion}")

    print("\nParsed output preview:")
    if isinstance(parsed, dict):
        print(f"  Root cause stage: {parsed.get('root_cause', {}).get('primary_bottleneck_stage', '?')}")
        print(f"  Strategy: {parsed.get('strategy_selected', {}).get('name', '?')}")
        print(f"  Config changes: {len(parsed.get('config_changes', []))}")
        print(f"  Outer lessons: {len(parsed.get('outer_lessons', []))}")
    else:
        print(f"  (raw) {raw[:200]}...")

    return {
        "model": cfg["label"],
        "elapsed": round(elapsed, 1),
        "output_length": len(raw),
        "score": f"{passed}/{total_criteria}",
        "criteria": scores,
        "parsed": parsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODEL_CONFIGS), choices=list(MODEL_CONFIGS))
    args = parser.parse_args()

    # Load ~/.env if present
    env_path = Path.home() / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip())

    results = {}
    for model_key in args.models:
        results[model_key] = run_test(model_key)

    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Model':<35} {'Score':<10} {'Time':<8} {'Chars'}")
    print("-" * 65)
    for key, r in results.items():
        if "error" in r:
            print(f"{MODEL_CONFIGS[key]['label']:<35} ERROR: {r['error']}")
        else:
            print(f"{r['model']:<35} {r['score']:<10} {r['elapsed']:<8} {r['output_length']}")

    # Save results
    out_path = Path(__file__).parent / "test_outer_model_results.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump({k: {**v, "parsed": str(v.get("parsed", ""))} for k, v in results.items()},
                  f, indent=2, ensure_ascii=False, default=str)
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
