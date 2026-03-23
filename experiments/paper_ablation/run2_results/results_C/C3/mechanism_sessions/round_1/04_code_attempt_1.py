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
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler, OneHotEncoder


class BayesianOptimizer:
    """Bayesian optimization with constrained action space for hyperparameter tuning."""
    
    def __init__(
        self,
        param_bounds: Dict[str, Tuple[float, float]],
        discrete_params: Dict[str, List[float]],
        init_samples: int = 5,
        exploration_weight: float = 0.1,
        max_consecutive_failures: int = 3
    ):
        """
        Args:
            param_bounds: Continuous parameter bounds, e.g., {"lr": (1e-5, 1e-2)}
            discrete_params: Discrete parameter options, e.g., {"optimizer": ["adam", "sgd"]}
            init_samples: Number of random samples before using GP
            exploration_weight: Weight for acquisition function (higher = more exploration)
            max_consecutive_failures: Reset to random search after this many failures
        """
        self.param_bounds = param_bounds
        self.discrete_params = discrete_params
        self.init_samples = init_samples
        self.exploration_weight = exploration_weight
        self.max_consecutive_failures = max_consecutive_failures
        
        # Track state
        self.consecutive_failures = 0
        self.recent_attempts = []
        self.history = []
        self.param_names = list(param_bounds.keys()) + list(discrete_params.keys())
        
        # Preprocess discrete params mapping
        self.discrete_mapping = {}
        for param, values in discrete_params.items():
            self.discrete_mapping[param] = {v: i for i, v in enumerate(values)}
        
        # Initialize GP and scalers
        self.gp = None
        self.X_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        
        # LLM constraint cache
        self.llm_constraints = {}
        
    def _params_to_features(self, params: Dict[str, Any]) -> np.ndarray:
        """Convert params dict to feature vector."""
        features = []
        
        # Continuous params (normalized to [0, 1])
        for param, (low, high) in self.param_bounds.items():
            val = params.get(param, (low + high) / 2)
            # Clip to bounds
            val = max(low, min(high, val))
            # Normalize
            norm_val = (val - low) / (high - low)
            features.append(norm_val)
        
        # Discrete params (one-hot encoded)
        for param, values in self.discrete_params.items():
            val = params.get(param, values[0])
            if val not in values:
                val = values[0]
            # One-hot encoding
            idx = self.discrete_mapping[param][val]
            one_hot = [0] * len(values)
            one_hot[idx] = 1
            features.extend(one_hot)
        
        return np.array(features).reshape(1, -1)
    
    def _features_to_params(self, features: np.ndarray) -> Dict[str, Any]:
        """Convert feature vector back to params dict."""
        features = features.flatten()
        idx = 0
        params = {}
        
        # Continuous params
        for param, (low, high) in self.param_bounds.items():
            norm_val = features[idx]
            # Clip to [0, 1]
            norm_val = max(0.0, min(1.0, norm_val))
            # Denormalize
            val = low + norm_val * (high - low)
            params[param] = val
            idx += 1
        
        # Discrete params
        for param, values in self.discrete_params.items():
            one_hot = features[idx:idx + len(values)]
            # Find max index
            max_idx = np.argmax(one_hot)
            params[param] = values[max_idx]
            idx += len(values)
        
        return params
    
    def _random_sample(self) -> Dict[str, Any]:
        """Generate random sample within bounds."""
        params = {}
        
        # Continuous params
        for param, (low, high) in self.param_bounds.items():
            params[param] = random.uniform(low, high)
        
        # Discrete params
        for param, values in self.discrete_params.items():
            params[param] = random.choice(values)
        
        return params
    
    def _apply_llm_constraints(self, params: Dict[str, Any], llm_insights: Optional[str]) -> float:
        """Apply penalty based on LLM insights."""
        if not llm_insights:
            return 0.0
        
        penalty = 0.0
        insight_lower = llm_insights.lower()
        
        # Parse constraints from insights
        if "lower learning rate" in insight_lower or "lower lr" in insight_lower:
            if "lr" in params:
                # Penalize high learning rates
                lr = params["lr"]
                max_lr = self.param_bounds["lr"][1]
                penalty += (lr / max_lr) * 10.0
        
        if "higher learning rate" in insight_lower or "higher lr" in insight_lower:
            if "lr" in params:
                # Penalize low learning rates
                lr = params["lr"]
                min_lr = self.param_bounds["lr"][0]
                penalty += (1.0 - lr / min_lr) * 10.0 if min_lr > 0 else 0.0
        
        if "avoid" in insight_lower:
            # Simple pattern matching for avoidance
            if "batch" in insight_lower and "batch_size" in params:
                # Example: "avoid large batch sizes"
                batch = params["batch_size"]
                max_batch = self.param_bounds["batch_size"][1]
                penalty += (batch / max_batch) * 5.0
        
        return penalty
    
    def _is_too_similar(self, params1: Dict[str, Any], params2: Dict[str, Any], threshold: float = 0.1) -> bool:
        """Check if two parameter sets are too similar."""
        if not params1 or not params2:
            return False
        
        total_diff = 0
        count = 0
        
        # Compare continuous params
        for param in self.param_bounds:
            if param in params1 and param in params2:
                low, high = self.param_bounds[param]
                norm_diff = abs(params1[param] - params2[param]) / (high - low)
                total_diff += norm_diff
                count += 1
        
        # Compare discrete params
        for param in self.discrete_params:
            if param in params1 and param in params2:
                if params1[param] != params2[param]:
                    total_diff += 1.0
                count += 1
        
        if count == 0:
            return False
        
        avg_diff = total_diff / count
        return avg_diff < threshold
    
    def propose_next(
        self,
        history: List[Dict[str, Any]],
        current_best: float,
        llm_insights: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Propose next hyperparameter configuration.
        
        Args:
            history: List of dicts with keys: "params", "score", "iteration"
            current_best: Best score seen so far
            llm_insights: Optional LLM interpretation of patterns
            
        Returns:
            Dict with parameter changes to apply
        """
        # Filter successful samples
        successful_history = [h for h in history if "score" in h and isinstance(h["score"], (int, float))]
        
        # Check if we have enough data for GP
        if len(successful_history) < self.init_samples:
            # Generate random sample, avoiding recent failures
            for _ in range(10):  # Try up to 10 times
                candidate = self._random_sample()
                
                # Check against recent attempts
                too_similar = any(
                    self._is_too_similar(candidate, attempt.get("params", {}))
                    for attempt in self.recent_attempts[-3:]
                )
                
                if not too_similar:
                    self.recent_attempts.append({"params": candidate, "type": "random"})
                    return candidate
            
            # Fallback
            return self._random_sample()
        
        # We have enough data, use GP
        # Prepare training data
        X = []
        y = []
        
        for entry in successful_history:
            if "params" in entry and "score" in entry:
                try:
                    features = self._params_to_features(entry["params"])
                    X.append(features.flatten())
                    y.append(entry["score"])
                except Exception as e:
                    logging.debug(f"Failed to process entry for GP: {e}")
                    continue
        
        if len(X) < 2:
            # Not enough valid data, fallback to random
            return self._random_sample()
        
        X = np.array(X)
        y = np.array(y).reshape(-1, 1)
        
        # Scale features and targets
        X_scaled = self.X_scaler.fit_transform(X)
        y_scaled = self.y_scaler.fit_transform(y)
        
        # Fit GP
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, alpha=1e-2)
        self.gp.fit(X_scaled, y_scaled)
        
        # Generate candidate points
        n_candidates = 100
        candidates = []
        
        for _ in range(n_candidates):
            candidate_params = self._random_sample()
            candidates.append(candidate_params)
        
        # Evaluate acquisition function
        best_candidate = None
        best_acquisition = -float('inf')
        
        for candidate in candidates:
            # Convert to features
            features = self._params_to_features(candidate)
            features_scaled = self.X_scaler.transform(features.reshape(1, -1))
            
            # GP prediction
            if self.gp:
                y_pred, y_std = self.gp.predict(features_scaled, return_std=True)
                y_pred = self.y_scaler.inverse_transform(y_pred.reshape(-1, 1))[0, 0]
                y_std = max(y_std[0], 1e-6)
                
                # Upper Confidence Bound acquisition
                # Lower scores are better, so we use negative of prediction
                acquisition = -y_pred + self.exploration_weight * y_std
            else:
                # Fallback to random
                acquisition = random.random()
            
            # Apply LLM constraint penalty
            penalty = self._apply_llm_constraints(candidate, llm_insights)
            acquisition -= penalty
            
            # Check against recent attempts
            too_similar = any(
                self._is_too_similar(candidate, attempt.get("params", {}))
                for attempt in self.recent_attempts[-3:]
            )
            
            if too_similar:
                acquisition -= 5.0  # Strong penalty for similarity
            
            if acquisition > best_acquisition:
                best_acquisition = acquisition
                best_candidate = candidate
        
        if best_candidate is None:
            best_candidate = self._random_sample()
        
        self.recent_attempts.append({"params": best_candidate, "type": "gp"})
        return best_candidate
    
    def register_result(
        self,
        params: Dict[str, Any],
        score: float,
        success: bool
    ) -> None:
        """Register the outcome of a proposed configuration."""
        # Update consecutive failures counter
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
        
        # Adjust exploration weight based on failures
        if self.consecutive_failures >= self.max_consecutive_failures:
            # Reset to more exploration
            self.exploration_weight *= 2.0
            logging.info(f"Reset Bayesian optimizer to random search due to {self.consecutive_failures} failures")
        
        # Gradually reduce exploration weight when things are going well
        if success and self.consecutive_failures == 0:
            self.exploration_weight = max(0.01, self.exploration_weight * 0.9)