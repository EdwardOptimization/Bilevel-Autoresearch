"""Register adaptation — formal vs informal prompt style affects proposal quality.

Translation Theory Mechanism 38:
The "register" of the prompt (formal academic vs casual engineering vs numerical
terse) affects the quality and character of the LLM's proposals. Formal prompts
elicit conservative, well-reasoned but timid proposals. Informal prompts elicit
creative but riskier ideas. Terse numerical prompts elicit precise but narrow
changes. This mechanism tracks which prompt register produces the best outcomes
and adaptively selects the register that matches the current search phase.
During early exploration, use informal/creative register. During late exploitation,
use formal/precise register. During stagnation, switch register to break patterns.
"""
from __future__ import annotations

import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# Three register styles that wrap the guidance text differently
REGISTER_FORMAL = (
    "You are conducting a rigorous hyperparameter optimization study. "
    "Each proposal should be grounded in established ML theory (learning rate "
    "schedules, batch size scaling laws, weight decay regularization theory). "
    "State your theoretical justification precisely. Prefer conservative, "
    "well-motivated changes over speculative ones."
)

REGISTER_INFORMAL = (
    "You're an ML hacker trying stuff out. What's your gut telling you? "
    "Don't overthink it -- try something bold that might surprise us. "
    "The trace shows what's been working; riff on that. If something weird "
    "might work, go for it. We can always revert."
)

REGISTER_TERSE = (
    "Respond with minimal text. Focus entirely on the numbers. "
    "Look at the numerical patterns in the trace: which values cluster "
    "near improvements? What interpolation or extrapolation of those "
    "numbers would you try next? Ignore theory; follow the data."
)

_REGISTERS = {
    "formal": REGISTER_FORMAL,
    "informal": REGISTER_INFORMAL,
    "terse": REGISTER_TERSE,
}


class RegisterAdaptation:
    """Adaptively selects prompt register based on search phase and outcomes.

    Tracks which register produces the best val_bpb improvements and switches
    register when the current one stagnates. Also considers the search phase:
    early iterations favor informal (exploration), late iterations favor
    formal (exploitation), stagnation triggers a register switch.
    """

    def __init__(self, switch_window: int = 4, min_samples_per_register: int = 2):
        """
        Args:
            switch_window: Number of iterations to evaluate before considering switch.
            min_samples_per_register: Min observations before judging a register.
        """
        self._switch_window = switch_window
        self._min_samples = min_samples_per_register

        # register_name -> list of {iteration, delta_bpb, status}
        self._outcomes: dict[str, list[dict]] = defaultdict(list)

        # Current active register
        self._current_register: str = "formal"

        # History of register switches for debugging
        self._switch_log: list[dict] = []

        # Iteration counter within current register
        self._iters_in_register: int = 0

        # Track consecutive non-improvements within a register
        self._stagnation_count: int = 0

    def get_current_register(self) -> str:
        """Return the name of the currently active register."""
        return self._current_register

    def get_register_prefix(self) -> str:
        """Return the prompt prefix text for the current register."""
        return _REGISTERS.get(self._current_register, REGISTER_FORMAL)

    def record(self, val_bpb: float, best_bpb_before: float,
               status: str, iteration: int) -> None:
        """Record the outcome of a proposal made under the current register.

        Args:
            val_bpb: Observed validation bpb.
            best_bpb_before: Best bpb before this run.
            status: "keep" | "discard" | "crash".
            iteration: Current iteration number.
        """
        delta = 0.0 if status == "crash" else (val_bpb - best_bpb_before)
        improved = status != "crash" and delta < 0

        self._outcomes[self._current_register].append({
            "iteration": iteration,
            "delta_bpb": delta,
            "status": status,
        })

        self._iters_in_register += 1

        if improved:
            self._stagnation_count = 0
        else:
            self._stagnation_count += 1

        # Check if we should switch register
        if self._iters_in_register >= self._switch_window:
            self._maybe_switch(iteration)

    def _maybe_switch(self, iteration: int) -> None:
        """Evaluate whether to switch to a different register."""
        # Compute recent performance of current register
        recent = self._outcomes[self._current_register][-self._switch_window:]
        recent_improvements = sum(1 for e in recent if e["delta_bpb"] < 0)

        # Switch if current register is stagnating
        should_switch = (
            self._stagnation_count >= self._switch_window
            or (recent_improvements == 0 and len(recent) >= self._switch_window)
        )

        if not should_switch:
            return

        # Pick the register with best historical performance, or cycle if untested
        best_register = None
        best_avg = float("inf")

        for reg_name in _REGISTERS:
            if reg_name == self._current_register:
                continue

            outcomes = self._outcomes.get(reg_name, [])
            if len(outcomes) < self._min_samples:
                # Untested register — try it (exploration)
                best_register = reg_name
                break

            avg_delta = sum(e["delta_bpb"] for e in outcomes) / len(outcomes)
            if avg_delta < best_avg:
                best_avg = avg_delta
                best_register = reg_name

        if best_register is None:
            # Cycle through registers in order
            register_order = list(_REGISTERS.keys())
            cur_idx = register_order.index(self._current_register)
            best_register = register_order[(cur_idx + 1) % len(register_order)]

        old_register = self._current_register
        self._current_register = best_register
        self._iters_in_register = 0
        self._stagnation_count = 0

        self._switch_log.append({
            "iteration": iteration,
            "from": old_register,
            "to": best_register,
            "reason": f"stagnation ({self._switch_window} iters without improvement)",
        })

        logger.info(
            f"[RegisterAdaptation] Switching register: {old_register} -> {best_register} "
            f"at iteration {iteration}"
        )

    def get_register_text(self) -> str:
        """Generate register context for the proposal prompt.

        Returns the register-specific prefix plus a summary of which registers
        have worked best historically.
        """
        prefix = self.get_register_prefix()

        # Build performance summary if we have enough data
        perf_lines = []
        for reg_name in _REGISTERS:
            outcomes = self._outcomes.get(reg_name, [])
            if not outcomes:
                continue
            improvements = sum(1 for e in outcomes if e["delta_bpb"] < 0)
            avg_delta = sum(e["delta_bpb"] for e in outcomes) / len(outcomes)
            perf_lines.append(
                f"  {reg_name}: {improvements}/{len(outcomes)} improvements, "
                f"avg delta={avg_delta:+.6f}"
            )

        lines = [f"## Prompt Register: {self._current_register.upper()}"]
        lines.append(prefix)

        if perf_lines:
            lines.append("\nRegister performance history:")
            lines.extend(perf_lines)

        if self._switch_log:
            last_switch = self._switch_log[-1]
            lines.append(
                f"\n(Switched from '{last_switch['from']}' at iter "
                f"{last_switch['iteration']}: {last_switch['reason']})"
            )

        return "\n".join(lines)
