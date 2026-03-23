# Implementation Specification: Directed Forgetting & Tabu Search

## 1. **Mechanism name**
`tabu_search_manager`

## 2. **Implementation strategy**
`new_helper_class` - Create a standalone helper class similar to ElitePool and CrashMemory

## 3. **Target**
New class: `TabuSearchManager`

## 4. **Interface**

```python
class TabuSearchManager:
    """Manages tabu lists to prevent revisiting recently explored parameter regions."""
    
    def __init__(
        self,
        max_tabu_size: int = 10,
        tabu_tenure: int = 3,
        distance_thresholds: dict[str, float] | None = None,
        enable_adaptive_thresholds: bool = True
    ):
        """
        Args:
            max_tabu_size: Maximum number of tabu entries to maintain
            tabu_tenure: Number of iterations an entry stays in the tabu list
            distance_thresholds: Parameter-specific distance thresholds for "closeness"
            enable_adaptive_thresholds: Whether to adjust thresholds based on search progress
        """
        
    def is_tabu(self, config: dict[str, Any], current_iteration: int) -> tuple[bool, str]:
        """
        Check if a configuration is tabu (forbidden).
        
        Args:
            config: Hyperparameter configuration to check
            current_iteration: Current iteration number
            
        Returns:
            Tuple of (is_tabu: bool, reason: str)
        """
        
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
```

## 5. **Step-by-step logic**

### 5.1 `__init__` method
1. Store initialization parameters as attributes
2. Initialize empty tabu list: `self.tabu_list = []` (list of dicts with keys: `config`, `added_at`, `expires_at`, `reason`)
3. Set default distance thresholds if not provided:
   ```python
   self.distance_thresholds = distance_thresholds or {
       "weight_decay": 0.02,      # ±0.02 considered "close"
       "embed_lr": 5e-4,          # ±0.0005 considered "close"
       "lr": 1e-4,                # ±0.0001 considered "close"
       "batch_size": 4,           # ±4 considered "close"
       "dropout": 0.05,           # ±0.05 considered "close"
   }
   ```
4. Initialize adaptive tracking: `self.iteration_history = []`

### 5.2 `is_tabu` method
1. For each entry in `self.tabu_list`:
   - If `current_iteration > entry["expires_at"]`, skip (entry expired)
   - For each parameter in the entry's config:
     - If parameter exists in both configs:
       - Calculate absolute difference: `abs(config[param] - entry["config"][param])`
       - If difference ≤ `self.distance_thresholds.get(param, 0.0)`:
         - Return `(True, f"Parameter {param} value {config[param]} too close to previously explored {entry['config'][param]}")`
2. Return `(False, "")`

### 5.3 `add_tabu_entry` method
1. Create tabu entry:
   ```python
   entry = {
       "config": config.copy(),
       "added_at": iteration,
       "expires_at": iteration + self.tabu_tenure,
       "reason": reason
   }
   ```
2. Append to `self.tabu_list`
3. If `len(self.tabu_list) > self.max_tabu_size`:
   - Remove oldest entry (lowest `added_at`)
4. Log the addition

### 5.4 `update_distance_thresholds` method (adaptive)
1. If not `self.enable_adaptive_thresholds`, return
2. If `len(elite_configs) < 2`, return (need at least 2 for variance)
3. For each parameter:
   - Collect values from elite configs
   - Calculate standard deviation of values
   - Adjust threshold: `new_threshold = max(0.5 * std, 0.1 * (param_range[1] - param_range[0]))`
   - Update `self.distance_thresholds[param] = new_threshold`
4. Record update in `self.iteration_history`

### 5.5 `get_suggested_alternatives` method
1. Initialize `alternatives = {}`
2. For each parameter in `current_config`:
   - Check if current value would be tabu against any entry
   - If tabu:
     - Get parameter range from `param_ranges`
     - Generate 3 alternative values:
       - Low alternative: `max(param_range[0], current_value - 2*threshold)`
       - High alternative: `min(param_range[1], current_value + 2*threshold)`
       - Random alternative: random value in range, avoiding tabu regions
     - Add to `alternatives[param] = [low, high, random]`
3. Return `alternatives`

## 6. **Integration points**

### 6.1 Modify `TrainRunner.__init__`:
```python
def __init__(self, ...):
    # ... existing code ...
    
    # Improvement 17: Tabu search manager
    self.tabu_manager = TabuSearchManager(
        max_tabu_size=8,
        tabu_tenure=4,
        enable_adaptive_thresholds=True
    )
```

### 6.2 Modify `run_iteration` method (after line 1):
```python
def run_iteration(self, iteration: int) -> TrainResult:
    # ... existing code until current_config extraction ...
    
    current_config = self._extract_hyperparams(self.current_code)
    
    # Update tabu manager with elite configurations
    if self.elite_pool.elites:
        elite_configs = [e["config"] for e in self.elite_pool.elites]
        self.tabu_manager.update_distance_thresholds(elite_configs, iteration)
    
    # ... rest of existing code ...
```

### 6.3 Create new method `_generate_tabu_aware_prompt`:
```python
def _generate_tabu_aware_prompt(
    self,
    base_prompt: str,
    current_config: dict[str, Any],
    iteration: int
) -> str:
    """
    Enhance the LLM prompt with tabu search guidance.
    """
    # Check if current config would be tabu
    is_tabu, reason = self.tabu_manager.is_tabu(current_config, iteration)
    
    if not is_tabu:
        return base_prompt
    
    # Get suggested alternatives for tabu parameters
    param_ranges = self.search_config.param_ranges  # Assuming this exists
    alternatives = self.tabu_manager.get_suggested_alternatives(current_config, param_ranges)
    
    # Build tabu guidance
    tabu_guidance = "\n\n## TABU SEARCH GUIDANCE\n"
    tabu_guidance += f"Current configuration is in a recently explored region: {reason}\n"
    tabu_guidance += "To escape local optima, consider these alternative values:\n"
    
    for param, alt_values in alternatives.items():
        tabu_guidance += f"- {param}: try {alt_values[0]:.4f} (low), {alt_values[1]:.4f} (high), or {alt_values[2]:.4f} (random)\n"
    
    tabu_guidance += "\nAvoid values close to recently explored ones to diversify search."
    
    return base_prompt + tabu_guidance
```

### 6.4 Modify the prompt generation in `run_iteration`:
```python
# In the proposal generation section, replace prompt generation with:
prompt = self._generate_proposal_prompt(current_config, iteration)  # Existing method
prompt = self._generate_tabu_aware_prompt(prompt, current_config, iteration)  # New wrapper
```

### 6.5 Add tabu entry after evaluation:
```python
# After evaluating a configuration in run_iteration:
if result.bpb < 100:  # Only tabu reasonable configurations
    self.tabu_manager.add_tabu_entry(
        config=result.config,
        iteration=iteration,
        reason="explored" if result.bpb > best_bpb else "poor_performance"
    )
```

### 6.6 Add to plateau detection handling:
```python
# In plateau/stagnation handling:
if self.plateau_detector.is_plateau():
    # Force exploration by adding current config to tabu
    self.tabu_manager.add_tabu_entry(
        config=current_config,
        iteration=iteration,
        reason="plateau_detected"
    )
```

## Key Design Decisions:

1. **Parameter-specific thresholds**: Different parameters need different "closeness" definitions
2. **Adaptive thresholds**: Automatically adjust based on elite configuration diversity
3. **Time-limited tabu**: Entries expire after `tabu_tenure` iterations
4. **Size-limited tabu**: Prevents memory explosion
5. **Suggestive guidance**: Doesn't just forbid, but suggests alternatives
6. **Integration with existing systems**: Works with ElitePool, PlateauDetector

This implementation directly addresses the "stuck-in-a-rut" pattern observed in the trace by forcing exploration away from recently tested values while providing intelligent guidance on where to explore next.