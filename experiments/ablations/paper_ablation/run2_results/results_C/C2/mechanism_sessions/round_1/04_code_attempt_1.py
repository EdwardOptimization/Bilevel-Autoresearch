import json
import logging
import math
import random
import re
import shutil
import subprocess
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path


class MultiScaleBanditProposer:
    """Bandit system for selecting proposal scales based on information gain."""

    def __init__(
        self,
        scale_levels: list[str] = None,
        initial_exploration_bias: float = 0.7,
        exploration_decay: float = 0.95,
        min_exploration: float = 0.1,
        optimistic_init_value: float = 2.0
    ):
        """
        Args:
            scale_levels: List of scale identifiers ['tiny', 'small', 'medium', 'large', 'xlarge']
            initial_exploration_bias: Initial probability of choosing exploration over exploitation
            exploration_decay: Multiplicative decay per iteration for exploration bias
            min_exploration: Minimum exploration probability
            optimistic_init_value: Initial Q-value for untried actions (optimistic initialization)
        """
        if scale_levels is None:
            scale_levels = ['tiny', 'small', 'medium', 'large', 'xlarge']
        self.scale_levels = scale_levels
        self.initial_exploration_bias = initial_exploration_bias
        self.exploration_decay = exploration_decay
        self.min_exploration = min_exploration
        self.optimistic_init_value = optimistic_init_value

        self.q_values = {scale: optimistic_init_value for scale in scale_levels}
        self.visit_counts = {scale: 0 for scale in scale_levels}
        self.total_visits = 0

        self.scale_parameters = {
            'tiny': {'max_delta': 0.01, 'num_changes': (1, 2), 'allow_new_params': False},
            'small': {'max_delta': 0.05, 'num_changes': (2, 3), 'allow_new_params': False},
            'medium': {'max_delta': 0.15, 'num_changes': (3, 4), 'allow_new_params': True},
            'large': {'max_delta': 0.3, 'num_changes': (4, 5), 'allow_new_params': True},
            'xlarge': {'max_delta': 0.5, 'num_changes': (5, 7), 'allow_new_params': True}
        }

    def select_scale(
        self,
        current_config: dict,
        iteration: int,
        recent_performance: list[float] = None
    ) -> tuple[str, dict]:
        """
        Select a scale level for the next proposal.

        Returns:
            Tuple of (scale_level, scale_parameters)
        """
        p_explore = max(
            self.min_exploration,
            self.initial_exploration_bias * (self.exploration_decay ** iteration)
        )

        if random.random() < p_explore:
            unexplored = [s for s in self.scale_levels if self.visit_counts[s] == 0]
            if unexplored:
                selected = random.choice(unexplored)
            else:
                ucb_scores = {}
                for scale in self.scale_levels:
                    if self.visit_counts[scale] == 0:
                        ucb_scores[scale] = float('inf')
                    else:
                        exploration_bonus = math.sqrt(2 * math.log(self.total_visits) / self.visit_counts[scale])
                        ucb_scores[scale] = self.q_values[scale] + exploration_bonus
                selected = max(ucb_scores.items(), key=lambda x: x[1])[0]
        else:
            selected = max(self.q_values.items(), key=lambda x: x[1])[0]

        return selected, self.get_scale_parameters(selected)

    def update_reward(
        self,
        scale_level: str,
        was_accepted: bool,
        information_gain: float,
        performance_change: float = None
    ):
        """
        Update bandit Q-values based on outcome.

        Args:
            scale_level: The scale that was used
            was_accepted: Whether proposal was accepted
            information_gain: Metric for how much we learned (0-1)
            performance_change: Change in bpb (positive = improvement)
        """
        self.visit_counts[scale_level] += 1
        self.total_visits += 1

        base_reward = 1.0 if was_accepted else 0.2

        if scale_level in ['large', 'xlarge']:
            bonus_multiplier = 1.5
        elif not was_accepted:
            bonus_multiplier = 1.2
        else:
            bonus_multiplier = 1.0

        performance_bonus = 0.0
        if performance_change is not None:
            performance_bonus = max(-0.5, min(0.5, performance_change * 0.5))

        total_reward = base_reward * bonus_multiplier + performance_bonus
        total_reward = max(0.0, min(2.0, total_reward))

        alpha = 1.0 / self.visit_counts[scale_level]
        self.q_values[scale_level] += alpha * (total_reward - self.q_values[scale_level])

    def get_scale_parameters(self, scale_level: str) -> dict:
        """Get mutation parameters for a given scale level."""
        params = self.scale_parameters.get(scale_level)
        if params is None:
            raise ValueError(f"Unknown scale level: {scale_level}")
        return params.copy()

    def _calculate_information_gain(
        self,
        scale_level: str,
        was_accepted: bool,
        performance_change: float = None
    ) -> float:
        """Calculate information gain metric (0-1)."""
        if was_accepted:
            gain = 0.3
        else:
            if scale_level in ['large', 'xlarge']:
                gain = 0.8
            else:
                gain = 0.5

        if performance_change is not None:
            if performance_change < -0.1:
                gain += 0.2
            elif abs(performance_change) < 0.05:
                gain += 0.1

        return min(gain, 1.0)

    def get_stats(self) -> dict:
        """Return current bandit statistics for debugging."""
        return {
            'q_values': self.q_values.copy(),
            'visit_counts': self.visit_counts.copy(),
            'total_visits': self.total_visits
        }