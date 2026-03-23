import json, logging, math, random, re, shutil, subprocess, time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple, List, Optional


class OrthogonalExplorer:
    """Manages systematic exploration of hyperparameter space using orthogonal sampling."""

    def __init__(
        self,
        param_ranges: Dict[str, Tuple[float, float]],
        n_samples: int = 8,
        exploration_bonus_weight: float = 0.3,
        decay_rate: float = 0.85
    ):
        """
        Args:
            param_ranges: Dict mapping param_name -> (min_value, max_value) in log10 space
            n_samples: Number of orthogonal samples to generate initially
            exploration_bonus_weight: Weight for exploration bonus (0-1)
            decay_rate: Exponential decay rate for step sizes per parameter
        """
        self.param_ranges = param_ranges
        self.n_samples = n_samples
        self.exploration_bonus_weight = exploration_bonus_weight
        self.decay_rate = decay_rate

        # Track exploration history
        self.param_visit_counts: Dict[str, int] = {name: 0 for name in param_ranges}
        self.param_step_sizes: Dict[str, float] = {}
        self.orthogonal_samples: List[Dict[str, float]] = []
        self.sample_index = 0

        self._initialize_step_sizes()
        self._generate_orthogonal_samples()

    def get_exploration_bonus(self, param_name: str) -> float:
        """Calculate exploration bonus for a parameter based on visit frequency."""
        total_visits = sum(self.param_visit_counts.values())
        if total_visits == 0:
            return 1.0

        param_ratio = self.param_visit_counts[param_name] / total_visits
        expected_ratio = 1.0 / len(self.param_visit_counts)

        # Bonus is higher for underexplored parameters
        bonus = max(0, expected_ratio - param_ratio) / expected_ratio
        return self.exploration_bonus_weight * bonus

    def get_step_size(self, param_name: str) -> float:
        """Get exponentially decayed step size for parameter."""
        return self.param_step_sizes[param_name]

    def record_visit(self, param_name: str):
        """Record that a parameter was modified."""
        self.param_visit_counts[param_name] += 1
        # Decay step size for this parameter
        self.param_step_sizes[param_name] *= self.decay_rate

    def get_orthogonal_sample(self) -> Optional[Dict[str, float]]:
        """Get next orthogonal sample, or None if all used."""
        if self.sample_index >= len(self.orthogonal_samples):
            return None

        sample = self.orthogonal_samples[self.sample_index]
        self.sample_index += 1
        return sample

    def _initialize_step_sizes(self):
        """Initialize step sizes based on parameter ranges."""
        for param_name, (min_val, max_val) in self.param_ranges.items():
            # Step size = 10% of parameter range (in log space)
            range_size = max_val - min_val
            self.param_step_sizes[param_name] = 0.1 * range_size

    def _generate_orthogonal_samples(self):
        """Generate Latin hypercube samples for systematic exploration."""
        n_params = len(self.param_ranges)
        param_names = list(self.param_ranges.keys())

        for i in range(self.n_samples):
            sample = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.param_ranges[param_name]
                # Stagger samples across the range
                offset = (i + j / n_params) % 1.0
                value = min_val + offset * (max_val - min_val)
                sample[param_name] = 10 ** value  # Convert from log space
            self.orthogonal_samples.append(sample)