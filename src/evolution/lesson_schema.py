"""Lesson data structure."""
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

LESSON_TYPES = [
    "failure_pattern",
    "successful_pattern",
    "decision",
    "warning",
    "guardrail",
]


@dataclass
class Lesson:
    id: str
    created_from_run: str
    lesson_type: str          # one of LESSON_TYPES
    scope: str                # "topic" | "stage" | "global"
    stage: str                # pipeline stage this lesson applies to
    topic_tags: list[str]
    summary: str
    evidence: list[str]       # paths or descriptions of supporting artifacts
    confidence: float         # 0.0 - 1.0
    reuse_rule: str           # actionable instruction for next run
    anti_pattern: str = ""    # what NOT to do (mainly for failure_pattern)
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Lesson":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def make_lesson_id() -> str:
    return f"lesson_{uuid.uuid4().hex[:8]}"
