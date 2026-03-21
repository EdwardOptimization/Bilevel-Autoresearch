"""Run and stage state enums."""
from enum import Enum


class RunState(str, Enum):
    CREATED = "created"
    RUNNING = "running"
    REVIEWED = "reviewed"
    LESSONS_EXTRACTED = "lessons_extracted"
    COMPLETED = "completed"
    FAILED = "failed"


class StageState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    REVIEWED = "reviewed"
