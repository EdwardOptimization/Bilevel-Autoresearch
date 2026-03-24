class StepSizeCalibrator:
    def __init__(self, initial_step_size: float = 0.1, 
                 min_step: float = 0.01, max_step: float = 0.5,
                 adaptation_window: int = 3):
        """
        Args:
            initial_step_size: Starting step size for parameter modifications
            min_step: Minimum allowed step size (prevent over-refinement)
            max_step: Maximum allowed step size (prevent erratic jumps)
            adaptation_window: Number of recent iterations to consider for calibration
        """
        self.step_size = initial_step_size
        self.min_step = min_step
        self.max_step = max_step
        self.adaptation_window = adaptation_window
        
        # Track recent performance for step size adaptation
        self.recent_results: List[Tuple[float, float]] = []  # (step_magnitude, bpb_delta)
        self.last_config: Dict[str, Any] = {}
        
    def calibrate(self, current_bpb: float, new_bpb: float, 
                  current_config: Dict[str, Any], new_config: Dict[str, Any]) -> float:
        """
        Adjust step size based on performance of recent modifications.
        
        Args:
            current_bpb: BPB before modification
            new_bpb: BPB after modification
            current_config: Hyperparameters before modification
            new_config: Hyperparameters after modification
            
        Returns:
            Updated step size for next iteration
        """
        import logging
        logger = logging.getLogger(__name__)
        
        # 1a. Calculate bpb_delta = new_bpb - current_bpb (negative = improvement)
        bpb_delta = new_bpb - current_bpb
        
        # 1b. Calculate step_magnitude
        step_magnitude = self._calculate_step_magnitude(current_config, new_config)
        
        # 1c. Append to recent results
        self.recent_results.append((step_magnitude, bpb_delta))
        
        # 1d. Trim to adaptation window
        if len(self.recent_results) > self.adaptation_window:
            self.recent_results = self.recent_results[-self.adaptation_window:]
        
        # 2. Calculate metrics if we have enough data
        success_rate = 0.0
        correlation = 0.0
        
        if len(self.recent_results) >= 2:
            # Calculate success rate (improvements count)
            improvements = sum(1 for _, delta in self.recent_results if delta < 0)
            success_rate = improvements / len(self.recent_results)
            
            # Calculate correlation between step_magnitude and abs(bpb_delta)
            magnitudes = [m for m, _ in self.recent_results]
            deltas_abs = [abs(d) for _, d in self.recent_results]
            
            if len(set(magnitudes)) > 1 and len(set(deltas_abs)) > 1:
                # Simple correlation calculation
                mean_mag = sum(magnitudes) / len(magnitudes)
                mean_delta = sum(deltas_abs) / len(deltas_abs)
                
                numerator = sum((m - mean_mag) * (d - mean_delta) for m, d in zip(magnitudes, deltas_abs))
                denom_mag = sum((m - mean_mag) ** 2 for m in magnitudes)
                denom_delta = sum((d - mean_delta) ** 2 for d in deltas_abs)
                
                if denom_mag > 0 and denom_delta > 0:
                    correlation = numerator / (denom_mag * denom_delta) ** 0.5
        
        # 3. Adjust step_size based on rules
        original_step = self.step_size
        
        # 3a. Poor success rate
        if success_rate < 0.2:
            self.step_size = max(self.min_step, self.step_size * 0.7)
            if self.step_size != original_step:
                logger.info("Reducing step size due to low success rate")
        
        # 3b. Good success rate with positive correlation
        elif success_rate > 0.6 and correlation > 0.3:
            self.step_size = min(self.max_step, self.step_size * 1.2)
            if self.step_size != original_step:
                logger.info("Increasing step size due to consistent improvements")
        
        # 3c. Tiny change
        elif abs(bpb_delta) < 0.001:
            if step_magnitude < 0.1:
                self.step_size = min(self.max_step, self.step_size * 1.15)
        
        # 3d. Large regression
        elif abs(bpb_delta) > 0.01 and bpb_delta > 0:
            self.step_size = max(self.min_step, self.step_size * 0.75)
            if self.step_size != original_step:
                logger.info("Reducing step size due to large regression")
        
        # 3e. Moderate changes: keep step_size unchanged
        
        # 4. Return updated step_size
        return self.step_size
        
    def get_step_size(self) -> float:
        """Return current step size."""
        return self.step_size
        
    def _calculate_step_magnitude(self, old_config: Dict[str, Any], 
                                  new_config: Dict[str, Any]) -> float:
        """
        Calculate the magnitude of change between two configurations.
        Normalized to [0, 1] range.
        """
        if not old_config or not new_config:
            return 0.0
        
        changes = []
        
        # Get all unique keys
        all_keys = set(old_config.keys()) | set(new_config.keys())
        
        for key in all_keys:
            old_val = old_config.get(key)
            new_val = new_config.get(key)
            
            # Handle missing values
            if old_val is None or new_val is None:
                changes.append(1.0)  # Full change if key appears/disappears
                continue
            
            # Numeric values
            if isinstance(old_val, (int, float)) and isinstance(new_val, (int, float)):
                # Try to normalize based on typical ranges
                # For common hyperparameters, use known ranges
                param_ranges = {
                    'lr': (1e-5, 1e-1),
                    'weight_decay': (0.0, 0.1),
                    'dropout': (0.0, 0.5),
                    'warmup_epochs': (0, 10),
                    'epochs': (1, 100),
                    'batch_size': (8, 256),
                    'gradient_accumulation': (1, 16),
                }
                
                if key in param_ranges:
                    min_val, max_val = param_ranges[key]
                    if max_val > min_val:
                        old_norm = (old_val - min_val) / (max_val - min_val)
                        new_norm = (new_val - min_val) / (max_val - min_val)
                        change = abs(new_norm - old_norm)
                        changes.append(min(change, 1.0))
                    else:
                        changes.append(1.0 if old_val != new_val else 0.0)
                else:
                    # For unknown numeric params, use relative change capped at 1.0
                    if old_val != 0:
                        rel_change = abs(new_val - old_val) / abs(old_val)
                        changes.append(min(rel_change, 1.0))
                    else:
                        changes.append(1.0 if new_val != 0 else 0.0)
            
            # Categorical/boolean/string values
            else:
                changes.append(0.0 if old_val == new_val else 1.0)
        
        # Average all normalized changes
        if not changes:
            return 0.0
        return sum(changes) / len(changes)