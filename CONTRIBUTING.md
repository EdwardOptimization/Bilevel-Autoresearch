# Contributing to EvoResearch

Thank you for your interest in contributing!

## Development Setup

```bash
git clone https://github.com/your-username/evo-research.git
cd evo-research
pip install -e ".[dev]"
```

## Running Tests

```bash
python -m pytest tests/ -v
```

Tests cover: LLM client (think-tag stripping, JSON parsing), lesson retrieval, memory persistence.
They run offline — no API key required.

## Key Architecture Constraints

Before contributing, understand these design invariants:

1. **Evaluator isolation**: `StageEvaluator` and `RunReviewer` must NEVER receive lesson memory.
   Lessons only influence *proposals* (stages that produce research content), never *judgments*.

2. **Skills vs Lessons**: Promoted skills are distilled from many lessons. Skills are structural
   (how to do something); lessons are empirical (what went wrong/right in specific runs).

3. **Skill injection order**: Skills are prepended before lessons in the injection block.
   The order matters — skills are higher-authority guidance.

4. **No hardcoded models**: Stage implementations use `self.model` (inherits from config).
   Never hardcode a model name in pipeline stages.

## Adding a New Pipeline Stage

1. Create `src/pipeline/your_stage.py` extending `BaseStage`
2. Add `name = "your_stage"` class attribute
3. Register in `src/orchestrator/run_manager.py`
4. Add a rubric entry in `src/evaluator/rubric.py`
5. Decide if this stage receives lesson injection (add to `INJECTION_STAGES` if yes)
6. Write tests if the stage has non-trivial logic

## Adding a New LLM Provider

Edit `src/llm_client.py` — add an entry to `PROVIDERS`:

```python
"yourprovider": {
    "base_url": "https://api.example.com/v1",
    "default_model": "model-name",
    "api_key_env": "YOURPROVIDER_API_KEY",
    "native_sdk": False,  # True only for Anthropic
},
```

All OpenAI-compatible providers work with `native_sdk: False`.

## Code Style

```bash
ruff check src/ tests/    # lint
ruff format src/ tests/   # format
```

## Pull Request Guidelines

- Keep PRs focused — one concern per PR
- Tests must pass: `python -m pytest tests/ -v`
- Lint must pass: `ruff check src/ tests/`
- Update `OPTIMIZATION_LOG.md` if you ran evaluation loops
- Do not commit `config/local.yaml` (contains API keys) or `artifacts/` or `memory/`

## Reporting Issues

Please include:
- Python version
- Provider being used
- Full error traceback
- Whether the issue is reproducible
