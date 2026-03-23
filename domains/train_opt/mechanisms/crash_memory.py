"""Crash memory — tracks which parameter changes caused crashes."""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CrashRecord:
    """Records a single crash event for crash memory."""
    param: str
    value: str  # the value that caused the crash
    iteration: int
    error_hint: str  # short description of failure mode (OOM, timeout, etc.)


class CrashMemory:
    """Tracks which parameter changes caused crashes so the LLM can avoid them.

    This is distinct from the outer loop's freeze mechanism: the outer loop
    freezes params entirely, while crash memory gives the LLM a warning and
    lets it decide (e.g. try a *smaller* change instead of the same crash).
    """

    def __init__(self):
        self._crashes: list[CrashRecord] = []
        # param -> count of crashes involving that param
        self._param_crash_counts: dict[str, int] = defaultdict(int)

    def record(self, changes: dict, iteration: int, error_hint: str = "timeout/OOM") -> None:
        """Record a crash caused by the given parameter changes."""
        for param, value in changes.items():
            rec = CrashRecord(
                param=param,
                value=str(value),
                iteration=iteration,
                error_hint=error_hint,
            )
            self._crashes.append(rec)
            self._param_crash_counts[param] += 1
            logger.info(f"[CrashMemory] Recorded crash for {param}={value} (total: {self._param_crash_counts[param]})")

    @property
    def crash_count(self) -> int:
        return len(self._crashes)

    def get_warning_text(self) -> str:
        """Generate a warning block to inject into the proposal prompt.

        Returns empty string if no crashes recorded.
        """
        if not self._crashes:
            return ""

        lines = ["## Crash History (IMPORTANT — read before proposing)"]
        lines.append(
            "The following parameter changes caused training crashes (OOM, timeout, divergence). "
            "Avoid repeating these mistakes. If you must change a crash-prone parameter, "
            "use a MUCH more conservative value."
        )

        # Group by param
        by_param: dict[str, list[CrashRecord]] = defaultdict(list)
        for rec in self._crashes:
            by_param[rec.param].append(rec)

        for param, records in by_param.items():
            count = len(records)
            values = [r.value for r in records]
            lines.append(
                f"- **{param}** crashed {count} time(s) with values: {', '.join(values)}. "
                f"Reason: {records[0].error_hint}."
            )
            if count >= 2:
                lines.append(
                    f"  WARNING: {param} has crashed {count}+ times. "
                    f"Strongly consider NOT changing this parameter."
                )

        return "\n".join(lines)
