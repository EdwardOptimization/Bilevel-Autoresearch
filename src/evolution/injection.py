"""Format retrieved lessons into prompt-injectable text."""
from .lesson_schema import Lesson


def format_lessons_for_injection(lessons: list[Lesson], context: str = "") -> str:
    """
    Format lessons as a readable section for injecting into LLM prompts.
    Memory is ADVISORY: lessons inform proposals, they do NOT override evaluation.
    """
    if not lessons:
        return ""

    header = "## Lessons from Previous Runs (Advisory)"
    if context:
        header += f" — {context}"

    lines = [header, ""]
    lines.append(
        "These lessons were extracted from prior research runs on similar topics. "
        "Use them to improve your proposals and avoid known failure patterns. "
        "They are advisory: use your own judgment."
    )
    lines.append("")

    for i, lesson in enumerate(lessons, 1):
        type_label = lesson.lesson_type.replace("_", " ").title()
        lines.append(f"### Lesson {i} [{type_label}] (confidence: {lesson.confidence:.0%})")
        lines.append(f"**Stage**: {lesson.stage}")
        lines.append(f"**Summary**: {lesson.summary}")
        lines.append(f"**Reuse Rule**: {lesson.reuse_rule}")
        if lesson.anti_pattern:
            lines.append(f"**Avoid**: {lesson.anti_pattern}")
        lines.append("")

    return "\n".join(lines)
