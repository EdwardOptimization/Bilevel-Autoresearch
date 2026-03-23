# Implementation Specification: Systematic Orthogonal Exploration

## 1. **Mechanism name**: `systematic_orthogonal_exploration`

## 2. **Implementation strategy**: `new_helper_class`

## 3. **Target**: New class `OrthogonalExplorer` + integration into `TrainRunner`

## 4. **Interface**:

```python
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
    
    def get_orthogonal_sample(self) -> Dict[str, float] | None:
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
        # Simplified implementation - in practice would use scipy or similar
        n_params = len(self.param_ranges)
        param_names = list(self.param_ranges.keys())
        
        for i in range(self.n_samples):
            sample = {}
            for j, param_name in enumerate(param_names):
                min_val, max_val = self.param_ranges[param_name]
                # Stagger samples across the range
                offset = (i + j/self.n_params) % 1.0
                value = min_val + offset * (max_val - min_val)
                sample[param_name] = 10 ** value  # Convert from log space
            self.orthogonal_samples.append(sample)
```

## 5. **Step-by-step logic**:

### Integration into TrainRunner.__init__:

1. **Add to imports**:
   ```python
   from typing import Dict, Tuple
   ```

2. **Add to __init__ method**:
   ```python
   # After other initializations in __init__:
   
   # Improvement 17: Systematic orthogonal exploration
   param_ranges = {
       'learning_rate': (-5, -1),      # 1e-5 to 0.1 in log10
       'weight_decay': (-5, -1),       # 1e-5 to 0.1
       'batch_size_log2': (4, 10),     # 16 to 1024
       'warmup_iters': (0, 3),         # 1 to 1000 in log10
       'grad_clip': (-2, 2),           # 0.01 to 100
   }
   self.orthogonal_explorer = OrthogonalExplorer(
       param_ranges=param_ranges,
       n_samples=8,
       exploration_bonus_weight=0.3,
       decay_rate=0.85
   )
   
   # Track if we're still in orthogonal exploration phase
   self._orthogonal_phase = True
   ```

### Integration into run_iteration/LLM prompting:

3. **Modify the LLM prompt generation** (in the proposal phase):
   ```python
   def _generate_exploration_prompt(self, current_config: Dict) -> str:
       """Generate prompt with exploration guidance."""
       
       prompt_parts = []
       
       # 1. Add orthogonal sample if still in exploration phase
       if self._orthogonal_phase:
           sample = self.orthogonal_explorer.get_orthogonal_sample()
           if sample:
               prompt_parts.append("EXPLORATION PHASE: Try this systematic sample:")
               for param, value in sample.items():
                   prompt_parts.append(f"  - Set {param} = {value:.6g}")
               prompt_parts.append("")
           else:
               self._orthogonal_phase = False  # Exploration phase complete
       
       # 2. Add exploration bonuses for underexplored parameters
       if not self._orthogonal_phase:
           prompt_parts.append("EXPLORATION GUIDANCE (focus on underexplored):")
           for param_name in self.orthogonal_explorer.param_ranges:
               bonus = self.orthogonal_explorer.get_exploration_bonus(param_name)
               step_size = self.orthogonal_explorer.get_step_size(param_name)
               if bonus > 0.1:  # Only mention parameters needing exploration
                   prompt_parts.append(
                       f"  - {param_name}: exploration bonus={bonus:.2f}, "
                       f"suggested step={step_size:.3f} (log10 scale)"
                   )
           prompt_parts.append("")
       
       # 3. Add current best configuration context
       if self.elite_pool.best_config:
           prompt_parts.append("CURRENT BEST CONFIGURATION:")
           for param, value in self.elite_pool.best_config.items():
               prompt_parts.append(f"  - {param} = {value}")
       
       return "\n".join(prompt_parts)
   ```

4. **Update parameter modification tracking**:
   ```python
   def _record_parameter_changes(self, old_config: Dict, new_config: Dict):
       """Record which parameters were changed for exploration tracking."""
       for param_name in old_config:
           if param_name in new_config and old_config[param_name] != new_config[param_name]:
               self.orthogonal_explorer.record_visit(param_name)
   ```

5. **Modify run_iteration to use exploration guidance**:
   ```python
   # In run_iteration method, before calling LLM:
   
   # Generate exploration-enhanced prompt
   exploration_guidance = self._generate_exploration_prompt(current_config)
   
   # Add to existing prompt context
   full_prompt = f"""
   Current configuration:
   {current_config}
   
   {exploration_guidance}
   
   Previous best: {self.elite_pool.best_bpb if self.elite_pool.best_bpb else 'N/A'}
   
   Propose specific changes to improve validation loss...
   """
   ```

## 6. **Integration points**:

### A. **TrainRunner.__init__ additions**:
- Add `orthogonal_explorer` attribute
- Add `_orthogonal_phase` flag
- Define parameter ranges for systematic exploration

### B. **New methods to add to TrainRunner**:
- `_generate_exploration_prompt()` - generates exploration guidance
- `_record_parameter_changes()` - tracks which parameters were modified

### C. **Modifications to existing flow**:
1. In `run_iteration()`:
   - Call `_generate_exploration_prompt()` before LLM proposal
   - Inject exploration guidance into LLM prompt
   - After proposal is accepted, call `_record_parameter_changes()`

2. In proposal evaluation/acceptance:
   - Use orthogonal samples during initial exploration phase
   - Transition to exploration-bonus guided search after samples exhausted

### D. **Key behaviors**:
1. **First 8 iterations**: Use systematic orthogonal samples (Latin hypercube)
2. **Subsequent iterations**: Guide LLM toward underexplored parameters
3. **Step size decay**: Each time a parameter is modified, its step size decays by 15%
4. **Exploration bonus**: Parameters visited less frequently get higher bonus in prompts

### E. **Expected outcomes**:
- Better coverage of hyperparameter space in early iterations
- Avoids myopic focus on recently successful parameters
- Exponential decay prevents overshooting in well-explored dimensions
- Simple implementation with no external dependencies