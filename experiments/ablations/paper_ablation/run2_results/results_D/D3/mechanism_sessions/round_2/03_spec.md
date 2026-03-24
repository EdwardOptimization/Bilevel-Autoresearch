## Implementation Specification: Step-Size Calibrator Enhancement

**1. Mechanism name**: `adaptive_step_calibrator`

**2. Implementation strategy**: `replace_method` (enhance existing StepSizeCalibrator class)

**3. Target**: `StepSizeCalibrator` class (currently at Improvement 11)

**4. Interface**:

```python
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
        
    def get_step_size(self) -> float:
        """Return current step size."""
        return self.step_size
        
    def _calculate_step_magnitude(self, old_config: Dict[str, Any], 
                                  new_config: Dict[str, Any]) -> float:
        """
        Calculate the magnitude of change between two configurations.
        Normalized to [0, 1] range.
        """
```

**5. Step-by-step logic**:

```
1. In calibrate() method:
   a. Calculate bpb_delta = new_bpb - current_bpb (negative = improvement)
   b. Calculate step_magnitude = _calculate_step_magnitude(current_config, new_config)
   c. Append (step_magnitude, bpb_delta) to recent_results
   d. Trim recent_results to keep only last adaptation_window entries
   
2. If we have at least 2 entries in recent_results:
   a. Calculate correlation between step_magnitude and abs(bpb_delta)
   b. Calculate success_rate = count(improvements) / len(recent_results)
   
3. Adjust step_size based on rules:
   a. IF success_rate < 0.2 (poor success):
      - Reduce step_size by 30% (minimum min_step)
      - Log: "Reducing step size due to low success rate"
      
   b. ELSE IF success_rate > 0.6 (good success) AND correlation > 0.3:
      - Increase step_size by 20% (maximum max_step)
      - Log: "Increasing step size due to consistent improvements"
      
   c. ELSE IF abs(bpb_delta) < 0.001 (tiny change):
      - IF step_magnitude < 0.1: Increase step_size by 15%
      - ELSE: Keep step_size unchanged
      
   d. ELSE IF abs(bpb_delta) > 0.01 (large regression):
      - Reduce step_size by 25%
      - Log: "Reducing step size due to large regression"
      
   e. ELSE (moderate changes):
      - Keep step_size unchanged

4. Return updated step_size

5. In _calculate_step_magnitude():
   a. For each hyperparameter in configs:
      - If numeric: normalize change relative to parameter range
      - If categorical: 0 if same, 1 if different
      - If boolean: 0 if same, 1 if different
   b. Average all normalized changes
   c. Return magnitude in [0, 1] range
```

**6. Integration points**:

A. **TrainRunner.__init__ modification**:
```python
# Replace current StepSizeCalibrator initialization:
# self.step_calibrator = StepSizeCalibrator()
# With enhanced version:
self.step_calibrator = StepSizeCalibrator(
    initial_step_size=0.15,  # Slightly more aggressive default
    min_step=0.02,           # Prevent over-refinement
    max_step=0.4,            # Cap erratic jumps
    adaptation_window=4       # Consider last 4 iterations
)
```

B. **In run_iteration() method** (after training completes):
```python
# After obtaining train_result in run_iteration():
if iteration > 0:  # Need previous iteration for comparison
    prev_result = self.trace.results[-1] if self.trace.results else None
    if prev_result:
        # Extract configs from current and previous code
        current_config = self._extract_hyperparams(self.current_code)
        prev_config = self._extract_hyperparams(prev_result.code_snapshot)
        
        # Calibrate step size based on performance
        new_step = self.step_calibrator.calibrate(
            current_bpb=train_result.bpb,
            new_bpb=prev_result.bpb,
            current_config=current_config,
            new_config=prev_config
        )
        logger.info(f"[Iter {iteration}] Step size adjusted to: {new_step:.3f}")
```

C. **In LLM prompt generation** (where step size is used):
```python
# When constructing prompt for LLM, include current step size:
current_step = self.step_calibrator.get_step_size()
prompt += f"\nCurrent step size setting: {current_step:.3f} "
prompt += "(lower = conservative changes, higher = aggressive changes)"
```

D. **In parameter modification logic** (when applying LLM suggestions):
```python
# Use step_size to scale numeric parameter changes:
def _apply_parameter_change(self, param_name: str, current_value: float, 
                           suggested_value: float) -> float:
    step_size = self.step_calibrator.get_step_size()
    
    if isinstance(current_value, (int, float)):
        # Scale change by step_size
        delta = suggested_value - current_value
        scaled_delta = delta * step_size * 2  # Factor of 2 for reasonable scaling
        return current_value + scaled_delta
    else:
        return suggested_value  # Categorical/boolean unchanged
```

**Key Design Decisions**:
1. **Adaptive window**: Uses last N iterations (not all history) to respond to recent trends
2. **Multi-factor calibration**: Considers success rate, correlation, and magnitude of changes
3. **Conservative defaults**: Prevents wild oscillations in step size
4. **Integration-friendly**: Works with existing momentum tracking and elite pool
5. **Transparent logging**: Each adjustment is logged for debugging

This implementation directly addresses the "erratic step sizes" problem by making the step size responsive to recent search performance, creating a self-regulating system that becomes more conservative when changes are harmful and more aggressive when they're beneficial.