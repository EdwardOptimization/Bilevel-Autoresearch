import json, logging, math, random, re, shutil, subprocess, time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional


class DiversityEnforcer:
    """
    Enforces diversity in hyperparameter proposals to avoid repetitive configurations.
    """
    
    # Predefined min/max bounds for common hyperparameters (can be extended)
    PARAM_BOUNDS = {
        'learning_rate': (1e-5, 1.0),
        'weight_decay': (0.0, 0.1),
        'batch_size': (1, 1024),
        'gradient_accumulation_steps': (1, 32),
        'warmup_steps': (0, 10000),
        'max_steps': (1000, 1000000),
        'beta1': (0.8, 0.999),
        'beta2': (0.9, 0.9999),
        'epsilon': (1e-9, 1e-6),
        'clip_grad_norm': (0.0, 5.0),
        'dropout': (0.0, 0.5),
        'attention_dropout': (0.0, 0.5),
        'hidden_dropout': (0.0, 0.5),
        'lr_decay_factor': (0.1, 1.0),
        'lr_decay_steps': (1000, 100000),
        'lr_decay_patience': (1, 20),
        'min_lr': (1e-7, 1e-4),
    }
    
    # Categorical parameter options
    CATEGORICAL_OPTIONS = {
        'optimizer': ['adam', 'adamw', 'sgd', 'rmsprop', 'adagrad'],
        'scheduler': ['linear', 'cosine', 'constant', 'reduce_on_plateau', 'exponential'],
        'activation': ['relu', 'gelu', 'silu', 'tanh'],
        'weight_init': ['normal', 'xavier', 'kaiming', 'orthogonal'],
    }
    
    def __init__(
        self,
        history_window: int = 5,
        min_distance_threshold: float = 0.3,
        max_retries: int = 3
    ):
        """
        Args:
            history_window: Number of recent iterations to consider for diversity checking
            min_distance_threshold: Minimum normalized distance (0-1) required from recent configs
            max_retries: Maximum attempts to generate a diverse proposal before accepting the best
        """
        self.history_window = history_window
        self.min_distance_threshold = min_distance_threshold
        self.max_retries = max_retries
        self.logger = logging.getLogger(__name__)
        
    def enforce_diversity(
        self,
        proposed_config: Dict[str, Any],
        recent_configs: List[Dict[str, Any]],
        iteration: int
    ) -> Tuple[Dict[str, Any], bool, str]:
        """
        Check if proposed config is sufficiently diverse from recent history.
        
        Args:
            proposed_config: Hyperparameter configuration from LLM proposal
            recent_configs: List of recent configurations (most recent last)
            iteration: Current iteration number
            
        Returns:
            Tuple of (final_config, was_modified, reason)
            - final_config: Either original or modified configuration
            - was_modified: True if config was modified for diversity
            - reason: Human-readable explanation of what happened
        """
        # Step 1: Check if we have enough history
        if len(recent_configs) < 2:
            return proposed_config, False, "insufficient history for diversity check"
        
        # Normalize the proposed config
        norm_proposed, param_info = self._normalize_config(proposed_config)
        
        # Get recent configs within window
        window_configs = recent_configs[-self.history_window:]
        
        # Calculate minimum distance to recent configs
        min_distance = float('inf')
        closest_config_idx = -1
        
        for i, recent_config in enumerate(window_configs):
            norm_recent, _ = self._normalize_config(recent_config)
            distance = self._compute_distance(norm_proposed, norm_recent)
            if distance < min_distance:
                min_distance = distance
                closest_config_idx = i
        
        # Step 2: Diversity check
        if min_distance >= self.min_distance_threshold:
            reason = f"sufficiently diverse (distance={min_distance:.3f} >= {self.min_distance_threshold})"
            return proposed_config, False, reason
        
        # Step 3: Diversification needed
        reason = f"insufficient diversity (distance={min_distance:.3f} < {self.min_distance_threshold})"
        self.logger.info(f"[Iter {iteration}] {reason}")
        
        # Get the closest config for analysis
        closest_config = window_configs[closest_config_idx]
        norm_closest, _ = self._normalize_config(closest_config)
        
        # Identify parameters contributing most to similarity
        param_contributions = []
        for param_name in norm_proposed:
            if param_name in norm_closest:
                diff = abs(norm_proposed[param_name] - norm_closest[param_name])
                param_contributions.append((param_name, diff))
        
        # Sort by smallest difference (most similar)
        param_contributions.sort(key=lambda x: x[1])
        
        best_config = proposed_config.copy()
        best_distance = min_distance
        best_reason = reason
        
        # Try diversification up to max_retries
        for attempt in range(self.max_retries):
            modified_config = proposed_config.copy()
            modified_norm = norm_proposed.copy()
            
            # Modify top 2-3 most similar parameters
            num_to_modify = min(3, len(param_contributions))
            modified_params = []
            
            for i in range(num_to_modify):
                param_name, _ = param_contributions[i]
                original_value = proposed_config.get(param_name)
                
                if original_value is None:
                    continue
                    
                # Get parameter info
                param_type = param_info[param_name]['type']
                bounds = param_info[param_name].get('bounds')
                options = param_info[param_name].get('options')
                
                # Apply targeted perturbation
                new_value = self._perturb_parameter(
                    param_name, original_value, param_type, bounds, options,
                    norm_proposed[param_name], norm_closest.get(param_name)
                )
                
                if new_value != original_value:
                    modified_config[param_name] = new_value
                    modified_params.append(param_name)
                    
                    # Update normalized value
                    if param_type == 'numeric':
                        if param_name == 'learning_rate':
                            # Special handling for log-scale learning rate
                            log_val = math.log10(new_value)
                            min_log = math.log10(self.PARAM_BOUNDS['learning_rate'][0])
                            max_log = math.log10(self.PARAM_BOUNDS['learning_rate'][1])
                            modified_norm[param_name] = (log_val - min_log) / (max_log - min_log)
                        else:
                            min_val, max_val = bounds
                            modified_norm[param_name] = (new_value - min_val) / (max_val - min_val)
                    elif param_type == 'categorical':
                        # Re-encode categorical
                        if options:
                            idx = options.index(new_value) if new_value in options else 0
                            modified_norm[param_name] = idx / max(1, len(options) - 1)
                    elif param_type == 'boolean':
                        modified_norm[param_name] = 1.0 if new_value else 0.0
            
            # Recalculate distance with modified config
            new_min_distance = float('inf')
            for recent_config in window_configs:
                norm_recent, _ = self._normalize_config(recent_config)
                distance = self._compute_distance(modified_norm, norm_recent)
                if distance < new_min_distance:
                    new_min_distance = distance
            
            # Update best found configuration
            if new_min_distance > best_distance:
                best_config = modified_config
                best_distance = new_min_distance
                best_reason = f"diversified {modified_params} (new distance={new_min_distance:.3f})"
            
            # Check if we meet threshold
            if new_min_distance >= self.min_distance_threshold:
                final_reason = f"successfully diversified after {attempt+1} attempts: {best_reason}"
                return modified_config, True, final_reason
        
        # If we get here, we couldn't meet threshold
        if best_distance > min_distance:
            final_reason = f"improved diversity to {best_distance:.3f} but below threshold {self.min_distance_threshold}"
            return best_config, True, final_reason
        else:
            final_reason = f"could not improve diversity, accepting original (distance={min_distance:.3f})"
            return proposed_config, True, final_reason
    
    def _normalize_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, Dict]]:
        """
        Normalize a configuration to [0,1] range.
        Returns normalized dict and parameter info dict.
        """
        normalized = {}
        param_info = {}
        
        for param_name, value in config.items():
            if isinstance(value, (int, float)):
                # Check if it's learning rate (log scale)
                if param_name == 'learning_rate':
                    min_val, max_val = self.PARAM_BOUNDS.get('learning_rate', (1e-5, 1.0))
                    log_val = math.log10(max(value, min_val))
                    min_log = math.log10(min_val)
                    max_log = math.log10(max_val)
                    normalized[param_name] = (log_val - min_log) / (max_log - min_log)
                    param_info[param_name] = {
                        'type': 'numeric',
                        'bounds': (min_val, max_val),
                        'log_scale': True
                    }
                else:
                    # Get bounds for this parameter
                    bounds = self.PARAM_BOUNDS.get(param_name)
                    if bounds:
                        min_val, max_val = bounds
                        normalized_val = (value - min_val) / (max_val - min_val)
                        normalized_val = max(0.0, min(1.0, normalized_val))
                        normalized[param_name] = normalized_val
                        param_info[param_name] = {
                            'type': 'numeric',
                            'bounds': bounds,
                            'log_scale': False
                        }
                    else:
                        # Unknown numeric parameter, use default normalization
                        normalized[param_name] = 0.5
                        param_info[param_name] = {
                            'type': 'numeric',
                            'bounds': (value * 0.5, value * 1.5),
                            'log_scale': False
                        }
            
            elif isinstance(value, bool):
                normalized[param_name] = 1.0 if value else 0.0
                param_info[param_name] = {'type': 'boolean'}
            
            elif isinstance(value, str):
                # Categorical parameter
                options = self.CATEGORICAL_OPTIONS.get(param_name)
                if options:
                    idx = options.index(value) if value in options else 0
                    normalized[param_name] = idx / max(1, len(options) - 1)
                    param_info[param_name] = {
                        'type': 'categorical',
                        'options': options
                    }
                else:
                    # Unknown categorical, treat as binary
                    normalized[param_name] = 0.5
                    param_info[param_name] = {
                        'type': 'categorical',
                        'options': [value]
                    }
        
        return normalized, param_info
    
    def _compute_distance(self, norm_vec1: Dict[str, float], norm_vec2: Dict[str, float]) -> float:
        """
        Compute Euclidean distance between two normalized vectors.
        Only considers parameters present in both vectors.
        """
        common_params = set(norm_vec1.keys()) & set(norm_vec2.keys())
        if not common_params:
            return 1.0  # Maximum distance if no common parameters
        
        squared_diff = 0.0
        for param in common_params:
            diff = norm_vec1[param] - norm_vec2[param]
            squared_diff += diff * diff
        
        return math.sqrt(squared_diff / len(common_params))
    
    def _perturb_parameter(
        self,
        param_name: str,
        original_value: Any,
        param_type: str,
        bounds: Optional[Tuple[float, float]],
        options: Optional[List[str]],
        norm_value: float,
        norm_closest: Optional[float]
    ) -> Any:
        """
        Apply targeted perturbation to a parameter.
        """
        if param_type == 'numeric' and bounds:
            min_val, max_val = bounds
            current_norm = norm_value
            
            # Determine perturbation direction
            if norm_closest is not None and random.random() < 0.7:
                # Bias toward moving away from closest config
                direction = 1.0 if current_norm < norm_closest else -1.0
            else:
                # Random direction
                direction = 1.0 if random.random() < 0.5 else -1.0
            
            # Perturbation magnitude: 20-50% of range
            magnitude = random.uniform(0.2, 0.5)
            new_norm = current_norm + direction * magnitude
            
            # Clip to [0, 1]
            new_norm = max(0.0, min(1.0, new_norm))
            
            # Convert back to original scale
            if param_name == 'learning_rate':
                # Log scale for learning rate
                min_log = math.log10(min_val)
                max_log = math.log10(max_val)
                log_val = min_log + new_norm * (max_log - min_log)
                return 10 ** log_val
            else:
                return min_val + new_norm * (max_val - min_val)
        
        elif param_type == 'categorical' and options and len(options) > 1:
            if random.random() < 0.6:  # 60% chance to change category
                current_idx = options.index(original_value) if original_value in options else 0
                # Choose a different option
                other_options = [opt for opt in options if opt != original_value]
                if other_options:
                    return random.choice(other_options)
            
            return original_value
        
        elif param_type == 'boolean':
            if random.random() < 0.4:  # 40% chance to flip
                return not original_value
            return original_value
        
        return original_value