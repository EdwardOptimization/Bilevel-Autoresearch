class PhaseDetector:
    """Detects optimization phases based on performance trends."""
    
    def __init__(
        self,
        exploration_window: int = 3,
        exploitation_window: int = 4,
        stagnation_threshold: int = 3,
        min_improvement: float = 0.0003
    ):
        """
        Args:
            exploration_window: Consecutive improvements needed to enter "exploitation"
            exploitation_window: Consecutive non-improvements needed to enter "exploration"
            stagnation_threshold: Iterations with < min_improvement to detect "plateau"
            min_improvement: Minimum bpb improvement to count as progress
        """
        self.exploration_window = exploration_window
        self.exploitation_window = exploitation_window
        self.stagnation_threshold = stagnation_threshold
        self.min_improvement = min_improvement
        
        # State tracking
        self.improvement_streak = 0
        self.non_improvement_streak = 0
        self.plateau_counter = 0
        self.last_best_bpb = float('inf')
    
    def update_and_detect(self, iteration: int, current_bpb: float) -> str:
        """
        Update internal state and return current phase.
        
        Returns:
            One of: "exploration", "exploitation", "plateau", "reflection"
        """
        # Calculate improvement
        improvement = self.last_best_bpb - current_bpb
        has_improved = improvement > self.min_improvement
        
        # Update streaks
        if has_improved:
            self.improvement_streak += 1
            self.non_improvement_streak = 0
            self.plateau_counter = 0
            self.last_best_bpb = current_bpb
        else:
            self.non_improvement_streak += 1
            self.improvement_streak = 0
            if improvement > -self.min_improvement:  # Within noise margin
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0
        
        # Phase detection logic
        if self.plateau_counter >= self.stagnation_threshold:
            return "plateau"
        elif self.improvement_streak >= self.exploration_window:
            return "exploitation"
        elif self.non_improvement_streak >= self.exploitation_window:
            return "exploration"
        elif iteration % 5 == 0:  # Periodic reflection
            return "reflection"
        else:
            # Default: continue current strategy
            return "exploitation" if self.improvement_streak > 0 else "exploration"

def _build_proposal_prompt(
    self,
    current_config: dict[str, any],
    iteration: int,
    phase: str
) -> str:
    """
    Build phase-aware prompt for LLM proposals.
    
    Args:
        current_config: Current hyperparameter values
        iteration: Current iteration number
        phase: One of "exploration", "exploitation", "plateau", "reflection"
    
    Returns:
        Formatted prompt string
    """
    # Base prompt components
    lines = [
        "# Hyperparameter Optimization Proposal",
        f"Iteration: {iteration}",
        "",
        "## Current Configuration",
        json.dumps(current_config, indent=2),
        "",
        "## Crash Memory (recent failures to avoid)",
        self.crash_memory.get_warning_text(),
        "",
        "## Elite Pool (top configs found so far — look for PATTERNS)",
        self.elite_pool.get_elite_text(),
        "",
        "## Momentum (recent trends)",
        self.momentum.get_momentum_text(),
        "",
        "## Step Size Calibration",
        self.step_calibrator.get_step_size_text(),
        "",
        "## Plateau Detection",
        self.plateau_detector.check_plateau(),
        "",
        "## Phase-Aware Guidance"
    ]
    
    # Phase-specific instructions
    if phase == "exploration":
        lines.extend([
            "You are in EXPLORATION phase. Prioritize DIVERSITY over refinement.",
            "Try significantly different hyperparameter combinations, even if risky.",
            "Consider changing multiple parameters at once.",
            "Break out of local patterns and test new regions of the search space."
        ])
    elif phase == "exploitation":
        lines.extend([
            "You are in EXPLOITATION phase. Focus on REFINEMENT.",
            "Make small, targeted adjustments to the best-performing configuration.",
            "Change only 1-2 parameters at a time.",
            "Fine-tune values based on patterns from the elite pool."
        ])
    elif phase == "plateau":
        lines.extend([
            "You are in PLATEAU phase. The optimization has stagnated.",
            "CRITICALLY ANALYZE why progress stopped. Consider:",
            "- Are we in a local minimum?",
            "- Should we change optimization strategy?",
            "- Are there parameter interactions we're missing?",
            "- Could constraints or boundaries be limiting progress?",
            "Propose bolder changes that might escape the plateau."
        ])
    elif phase == "reflection":
        lines.extend([
            "You are in REFLECTION phase. Review the last 5 iterations.",
            "What patterns do you see in successful vs failed proposals?",
            "What assumptions might be wrong?",
            "Propose 1-2 'hypothesis tests' to validate your understanding.",
            "Think about the search strategy itself, not just parameter values."
        ])
    else:
        lines.append(f"Phase '{phase}' not recognized. Default to exploration mindset.")
    
    # Add phase context
    lines.extend([
        "",
        "## Phase Context",
        f"Current phase: {phase}"
    ])
    
    # Show recent phase history if available
    if hasattr(self, '_phase_history') and self._phase_history:
        recent = self._phase_history[-5:]  # Last 5 transitions
        lines.append("Recent phase history:")
        for iter_num, ph in recent:
            lines.append(f"  Iteration {iter_num}: {ph}")
    
    # Final instructions
    lines.extend([
        "",
        "## Your Task",
        "Propose exactly ONE new hyperparameter configuration.",
        "Return a JSON object with the following structure:",
        "{",
        '  "reasoning": "Your step-by-step reasoning here",',
        '  "changes": {"PARAM_NAME": value, ...}',
        "}",
        "",
        "Important constraints:",
        f"Active parameters: {list(self.search_config.active_params.keys())}",
        f"Frozen parameters: {list(self.search_config.frozen_params.keys())}",
        "Changes must be within the allowed ranges for each parameter.",
        "Do NOT propose changes to frozen parameters.",
        "",
        "Current train.py is available for reference if needed."
    ])
    
    return "\n".join(lines)

def get_phase_summary(self) -> dict:
    """Return phase detection state for debugging."""
    return {
        "current_phase": self._current_phase,
        "phase_history": self._phase_history[-10:],  # Last 10
        "detector_state": {
            "improvement_streak": self.phase_detector.improvement_streak,
            "non_improvement_streak": self.phase_detector.non_improvement_streak,
            "plateau_counter": self.phase_detector.plateau_counter,
            "last_best_bpb": self.phase_detector.last_best_bpb
        }
    }