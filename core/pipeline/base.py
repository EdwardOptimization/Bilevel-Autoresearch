"""Base class for article optimizer pipeline stages."""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseStage(ABC):
    """
    Base class for all article optimizer stages.

    context keys:
      - article_id (str)
      - article_content (str)          — current working copy of the article
      - previous_outputs (dict)        — stage_name → content str from earlier stages this run
      - retrieved_lessons (str)        — inner skills/lessons (cleared each outer iteration)
      - evaluator_feedback (str)       — feedback from quality gate retry
      - run_dir (Path)
      - run_number (int)

    Returns dict:
      - content (str)                  — primary output text
      - artifacts (list[str])          — paths relative to run_dir
    """

    name: str = "base"
    max_retries: int | None = None  # None = use global config

    def __init__(self, model: str = ""):
        self.model = model

    @abstractmethod
    def run(self, context: dict) -> dict: ...

    def _outer_guidance(self, context: dict) -> str:
        """Return outer loop guidance for this stage, or empty string."""
        overrides = context.get("outer_guidance", {})
        text = overrides.get(self.name, "")
        if text:
            return f"\n## Outer Loop Guidance (from process optimization)\n{text}\n"
        return ""

    def _save_artifact(self, run_dir: Path, filename: str, content: str) -> str:
        stage_dir = run_dir / "stages" / self.name
        full_path = stage_dir / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return str(full_path.relative_to(run_dir))
