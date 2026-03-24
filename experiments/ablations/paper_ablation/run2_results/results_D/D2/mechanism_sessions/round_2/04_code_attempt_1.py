import json
import logging
import math
import random
import re
import shutil
import subprocess
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple


class TrustRegionConstraint:
    """Manages trust regions for hyperparameter exploration."""
    
    def __init__(
        self,
        initial_radii: Dict[str, float],
        contraction_factor: float = 0.8,
        expansion_factor: float = 1.2,
        min_radius: Optional[Dict[str, float]] = None,
        max_radius: Optional[Dict[str, float]] = None,
        success_threshold: float = 0.0,
        window_size: int = 3,
    ):
        """
        Args:
            initial_radii: Dict mapping hyperparameter names to initial trust radii.
                          Example: {"EMBEDDING_LR": 0.1, "BATCH_SIZE": 32}
            contraction_factor: Multiply radius by this when region fails (0-1)
            expansion_factor: Multiply radius by this when region succeeds (>1)
            min_radius: Minimum allowed radii (prevents collapse to zero)
            max_radius: Maximum allowed radii (prevents unbounded expansion)
            success_threshold: Minimum bpb improvement to count as "successful" step
            window_size: How many recent iterations to consider for success rate
        """
        self.radii = initial_radii.copy()
        self.initial_radii = initial_radii.copy()
        self.contraction_factor = contraction_factor
        self.expansion_factor = expansion_factor
        self.min_radius = min_radius.copy() if min_radius else {}
        self.max_radius = max_radius.copy() if max_radius else {}
        self.success_threshold = success_threshold
        self.window_size = window_size
        
        self.history = deque(maxlen=window_size)
        self.success_count = 0
        self.total_updates = 0
        self.logger = logging.getLogger(__name__)
        
        self.log_scale_params = {"EMBEDDING_LR", "LR_DECAY_GAMMA", "WEIGHT_DECAY"}
        self.bounded_params = {
            "DROPOUT": (0.0, 0.5),
            "LR_DECAY_GAMMA": (0.5, 1.0),
        }
    
    def constrain_proposal(
        self,
        current_params: Dict[str, float],
        proposed_params: Dict[str, float],
    ) -> Dict[str, float]:
        """Clip proposed parameters to stay within trust region of current best."""
        constrained = {}
        
        for param, proposed_val in proposed_params.items():
            if param not in current_params or param not in self.radii:
                constrained[param] = proposed_val
                continue
            
            current_val = current_params[param]
            radius = self.radii.get(param, self.initial_radii.get(param, 0.0))
            
            if param in self.log_scale_params:
                log_current = math.log10(current_val)
                lower = log_current - radius
                upper = log_current + radius
                log_proposed = math.log10(proposed_val)
                log_clipped = max(lower, min(upper, log_proposed))
                constrained_val = 10 ** log_clipped
            else:
                lower = current_val - radius
                upper = current_val + radius
                constrained_val = max(lower, min(upper, proposed_val))
            
            if param in self.bounded_params:
                param_min, param_max = self.bounded_params[param]
                constrained_val = max(param_min, min(param_max, constrained_val))
            
            constrained[param] = constrained_val
        
        return constrained
    
    def update_region(
        self,
        current_params: Dict[str, float],
        new_params: Dict[str, float],
        improvement: float,
        iteration: int,
    ) -> None:
        """Update trust radii based on success/failure of last step."""
        is_success = improvement > self.success_threshold
        self.history.append((is_success, iteration))
        
        recent_successes = sum(1 for success, _ in self.history if success)
        recent_total = len(self.history)
        success_rate = recent_successes / recent_total if recent_total > 0 else 0.0
        
        self.total_updates += 1
        if is_success:
            self.success_count += 1
        
        radius_changes = {}
        for param in self.radii:
            current_radius = self.radii[param]
            
            if success_rate > 0.66:
                new_radius = current_radius * self.expansion_factor
                if param in self.max_radius:
                    new_radius = min(new_radius, self.max_radius[param])
                self.radii[param] = new_radius
                if abs(new_radius - current_radius) > 1e-9:
                    radius_changes[param] = (current_radius, new_radius, "expanded")
            
            elif success_rate < 0.33:
                new_radius = current_radius * self.contraction_factor
                if param in self.min_radius:
                    new_radius = max(new_radius, self.min_radius[param])
                self.radii[param] = new_radius
                if abs(new_radius - current_radius) > 1e-9:
                    radius_changes[param] = (current_radius, new_radius, "contracted")
        
        if radius_changes:
            self.logger.info(f"Iteration {iteration}: Trust region updated. Success rate: {success_rate:.2f}")
            for param, (old, new, action) in radius_changes.items():
                self.logger.info(f"  {param}: {old:.4g} -> {new:.4g} ({action})")
    
    def get_region_status(self) -> Dict[str, Any]:
        """Return current radii and success statistics for logging."""
        recent_successes = sum(1 for success, _ in self.history if success)
        recent_total = len(self.history)
        success_rate = recent_successes / recent_total if recent_total > 0 else 0.0
        
        return {
            "radii": self.radii.copy(),
            "success_rate": success_rate,
            "total_updates": self.total_updates,
            "success_count": self.success_count,
            "history_size": len(self.history),
        }
    
    def reset_region(self, param: Optional[str] = None) -> None:
        """Reset trust region for specific parameter or all parameters."""
        if param:
            if param in self.initial_radii:
                self.radii[param] = self.initial_radii[param]
        else:
            self.radii = self.initial_radii.copy()
        self.logger.info(f"Trust region reset for {param if param else 'all parameters'}")