---
name: research-evo
description: >
  Run an automated, self-improving AI research pipeline on a topic. Use when the user
  asks to research a topic, investigate a question, generate and test hypotheses, or
  produce a research report. The pipeline learns from previous runs via an evolution
  memory layer — each run gets better than the last. Also use when the user wants to
  query what the research system has already learned about a topic.
---

You have access to the `research-evo` CLI tool, which runs a 5-stage AI research
pipeline with evolutionary memory. Each run extracts lessons, stores them, and
injects them into the next run automatically.

## When to use this skill

- User asks to research a topic or question
- User wants to generate and test hypotheses about a subject
- User wants to see what EvoResearch has learned about a topic
- User asks about the research pipeline's memory or lessons
- User wants to run multiple research iterations on a topic

## Core workflow

### 1. Check memory first (before running)
Always check if EvoResearch has prior lessons on the topic:
```bash
research-evo list-lessons --limit 10
```
Or query specifically:
```bash
# Show what skills have been promoted from repeated patterns
research-evo list-skills
```

### 2. Run the pipeline
```bash
# Basic run
research-evo run "your research topic here"

# Show which lessons from memory are being injected
research-evo run "your topic" --show-lessons

# Run without quality gate (faster, less thorough)
research-evo run "your topic" --no-quality-gate
```

### 3. After the run — check lessons and consider promoting
```bash
# View lessons from this and all runs
research-evo list-lessons

# Promote high-confidence lessons to reusable skills (do this after 2+ runs)
research-evo promote-lessons --min-confidence 0.8
```

### 4. Run again to see evolution
```bash
# Second run — lessons from Run 1 will be injected automatically
research-evo run "same topic" --show-lessons
```

## What the pipeline does

> Note: This skill is for the **article optimization domain** (`src/`). The stages
> below are specific to that domain. Other domains (e.g. `domains/train_opt/`) have
> their own inner/outer loops.

| Stage | Output | Notes |
|-------|--------|-------|
| 1: Article Analysis | Analysis of current article | Identifies weaknesses |
| 2: Improvement Hypotheses | Proposed improvements | ← Lessons injected here |
| 3: Edit Planning | Concrete edit plan | ← Lessons injected here |
| 4: Impact Assessment | Predicted impact of edits | Evaluates proposed changes |
| 5: Revised Output | `revised_article.md` | ← Lessons injected here |

After pipeline: Evaluator scores each stage (0–10), Run Reviewer gives holistic verdict,
Lesson Extractor pulls structured lessons (failure/success/guardrail/warning/decision).

## Key architectural guarantees

- **Evaluator is isolated from memory**: scores reflect actual quality, not memory bias
- **Lessons are advisory**: they influence proposals, never override evaluation
- **Quality gate**: stages scoring below threshold auto-retry with evaluator feedback
- **Skills > Lessons**: promoted skills are distilled from many lessons — more reliable

## View results

```bash
# List all runs with scores
research-evo list-runs

# Compare two runs
research-evo compare <run_id_1> <run_id_2>

# Start web dashboard
research-evo serve

# Export a run
research-evo export <run_id>
```

## MCP integration (for agent tool use)

The system can run as an MCP server, exposing these tools:
- `research_run(topic)` — run full pipeline
- `memory_query(topic, stage, limit)` — query lessons
- `run_list()` — list recent runs
- `skill_list()` — list promoted skills
- `promote_lessons()` — trigger skill promotion

Add to Claude Code settings.json:
```json
{
  "mcpServers": {
    "evo-research": {
      "command": "research-evo",
      "args": ["mcp-server"],
      "cwd": "/path/to/research_evo_mvp"
    }
  }
}
```

## Interpreting results

| Verdict | Score | Meaning |
|---------|-------|---------|
| pass | 7–10 | Good quality, proceed |
| weak | 5–6 | Usable but has gaps, lessons extracted |
| fail | 0–4 | Major issues, auto-retried by quality gate |

A run's value is not just the report — it's the lessons extracted that improve future runs.
Even a "weak" run that produces good lessons is valuable.

## Important notes

- Artifacts are saved to `artifacts/run-{id}/`
- Memory persists across sessions in `memory/lessons.jsonl`
- Skills are saved to `memory/skills/` and auto-injected
- Config is in `config/local.yaml` (gitignored, put API keys here)
