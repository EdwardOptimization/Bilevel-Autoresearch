import json
import logging
import math
import random
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


class TabuSearchManager:
    """Manages tabu lists to prevent revisiting recently explored parameter regions."""

    def __init__(
        self,
        max_tabu_size: int = 10,
        tabu_tenure: int = 3,
        distance_thresholds: Optional[dict[str, float]] = None,
        enable_adaptive_thresholds: bool = True
    ):
        """
        Args:
            max_tabu_size: Maximum number of tabu entries to maintain
            tabu_tenure: Number of iterations an entry stays in the tabu list
            distance_thresholds: Parameter-specific distance thresholds for "closeness"
            enable_adaptive_thresholds: Whether to adjust thresholds based on search progress
        """
        self.max_tabu_size = max_tabu_size
        self.tabu_tenure = tabu_tenure
        self.distance_thresholds = distance_thresholds or {
            "weight_decay": 0.02,
            "embed_lr": 5e-4,
            "lr": 1e-4,
            "batch_size": 4,
            "dropout": 0.05,
        }
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.tabu_list: list[dict] = []  # each entry: config, added_at, expires_at, reason
        self.iteration_history: list[dict] = []
        self.logger = logging.getLogger(__name__)

    def is_tabu(self, config: dict[str, Any], current_iteration: int) -> tuple[bool, str]:
        """
        Check if a configuration is tabu (forbidden).

        Args:
            config: Hyperparameter configuration to check
            current_iteration: Current iteration number

        Returns:
            Tuple of (is_tabu: bool, reason: str)
        """
        # Clean expired entries first
        self.tabu_list = [entry for entry in self.tabu_list if current_iteration <= entry["expires_at"]]

        for entry in self.tabu_list:
            tabu_config = entry["config"]
            for param, tabu_value in tabu_config.items():
                if param in config:
                    current_value = config[param]
                    try:
                        diff = abs(float(current_value) - float(tabu_value))
                        threshold = self.distance_thresholds.get(param, 0.0)
                        if diff <= threshold:
                            reason = (f"Parameter {param} value {current_value} too close to "
                                      f"previously explored {tabu_value} (diff={diff:.6f} <= {threshold:.6f})")
                            return True, reason
                    except (ValueError, TypeError):
                        # If values are not numeric, skip distance check
                        if current_value == tabu_value:
                            reason = f"Parameter {param} value {current_value} matches previously explored value"
                            return True, reason
        return False, ""

    def add_tabu_entry(
        self,
        config: dict[str, Any],
        iteration: int,
        reason: str = "explored"
    ) -> None:
        """
        Add a configuration to the tabu list.

        Args:
            config: Configuration to forbid
            iteration: Current iteration number
            reason: Why this is being added (e.g., "explored", "poor_performance")
        """
        entry = {
            "config": config.copy(),
            "added_at": iteration,
            "expires_at": iteration + self.tabu_tenure,
            "reason": reason
        }
        self.tabu_list.append(entry)

        # Enforce max size by removing oldest entries
        if len(self.tabu_list) > self.max_tabu_size:
            self.tabu_list.sort(key=lambda x: x["added_at"])
            self.tabu_list = self.tabu_list[-self.max_tabu_size:]

        self.logger.debug(f"Added tabu entry at iteration {iteration}: {reason}")

    def update_distance_thresholds(
        self,
        elite_configs: list[dict[str, Any]],
        current_iteration: int
    ) -> None:
        """
        Adaptively update distance thresholds based on elite configurations.

        Args:
            elite_configs: List of elite configurations from ElitePool
            current_iteration: Current iteration number
        """
        if not self.enable_adaptive_thresholds:
            return
        if len(elite_configs) < 2:
            return

        # Collect all parameters present in elite configs
        all_params = set()
        for config in elite_configs:
            all_params.update(config.keys())

        # For each parameter, calculate std and adjust threshold
        updates = {}
        for param in all_params:
            values = []
            for config in elite_configs:
                if param in config:
                    try:
                        values.append(float(config[param]))
                    except (ValueError, TypeError):
                        # Non-numeric parameter, skip adaptive update
                        continue

            if len(values) >= 2:
                # Calculate standard deviation
                mean_val = sum(values) / len(values)
                variance = sum((v - mean_val) ** 2 for v in values) / len(values)
                std = math.sqrt(variance)

                # Get default range for parameter (approximate from current thresholds)
                # If we had param_ranges we could use them, but we don't have them here
                # Use a conservative approach: threshold = max(0.5*std, 0.1*default_threshold)
                default_threshold = self.distance_thresholds.get(param, 0.0)
                if default_threshold > 0:
                    new_threshold = max(0.5 * std, 0.1 * default_threshold)
                else:
                    new_threshold = 0.5 * std

                # Ensure threshold is not zero
                if new_threshold <= 0:
                    new_threshold = default_threshold if default_threshold > 0 else 0.01

                updates[param] = new_threshold

        # Apply updates
        self.distance_thresholds.update(updates)

        # Record history
        self.iteration_history.append({
            "iteration": current_iteration,
            "updates": updates.copy()
        })

    def get_suggested_alternatives(
        self,
        current_config: dict[str, Any],
        param_ranges: dict[str, tuple[float, float]]
    ) -> dict[str, list[float]]:
        """
        Suggest alternative values for parameters that are currently tabu.

        Args:
            current_config: Current hyperparameter configuration
            param_ranges: Valid ranges for each parameter

        Returns:
            Dictionary mapping parameter names to lists of suggested alternative values
        """
        alternatives = {}
        for param, current_value in current_config.items():
            if param not in param_ranges:
                continue

            try:
                current_val_float = float(current_value)
            except (ValueError, TypeError):
                # Non-numeric parameter, skip
                continue

            # Check if this parameter value would be tabu
            # Create a test config with just this parameter changed
            test_config = current_config.copy()
            # The actual check is done in is_tabu, which compares against all tabu entries
            # We'll rely on the caller to have called is_tabu first

            # Get parameter range and threshold
            param_min, param_max = param_ranges[param]
            threshold = self.distance_thresholds.get(param, 0.0)

            # Generate three alternatives
            low_alt = max(param_min, current_val_float - 2 * threshold)
            high_alt = min(param_max, current_val_float + 2 * threshold)

            # Random alternative that avoids tabu regions
            random_alt = None
            attempts = 0
            while attempts < 10 and random_alt is None:
                candidate = random.uniform(param_min, param_max)
                # Check if candidate is too close to any tabu value for this parameter
                too_close = False
                for entry in self.tabu_list:
                    if param in entry["config"]:
                        try:
                            tabu_val = float(entry["config"][param])
                            if abs(candidate - tabu_val) <= threshold:
                                too_close = True
                                break
                        except (ValueError, TypeError):
                            pass
                if not too_close:
                    random_alt = candidate
                attempts += 1

            if random_alt is None:
                # Fallback to midpoint if can't find non-tabu value
                random_alt = (param_min + param_max) / 2

            alternatives[param] = [low_alt, high_alt, random_alt]

        return alternatives