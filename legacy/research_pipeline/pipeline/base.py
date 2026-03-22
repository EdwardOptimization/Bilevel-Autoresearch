"""Base class for pipeline stages."""
from abc import ABC, abstractmethod
from pathlib import Path


class BaseStage(ABC):
    """
    Base class for all research pipeline stages.

    context keys:
      - topic (str)
      - previous_outputs (dict[stage_name, content_str])
      - retrieved_lessons (str)       — pre-formatted lesson text (advisory)
      - evaluator_feedback (str)      — feedback from previous failed attempt (quality gate)
      - run_dir (Path)

    Returns dict:
      - content (str)
      - artifacts (list[str])         — paths relative to run_dir
    """

    name: str = "base"
    max_retries: int | None = None  # None = use global config value

    def __init__(self, model: str = ""):
        self.model = model  # empty = use provider default

    @abstractmethod
    def run(self, context: dict) -> dict: ...

    def _save_artifact(self, run_dir: Path, filename: str, content: str) -> str:
        """Save content to stage artifact directory, return relative path."""
        stage_dir = run_dir / "stages" / self.name
        full_path = stage_dir / filename
        full_path.parent.mkdir(parents=True, exist_ok=True)
        full_path.write_text(content, encoding="utf-8")
        return str(full_path.relative_to(run_dir))
