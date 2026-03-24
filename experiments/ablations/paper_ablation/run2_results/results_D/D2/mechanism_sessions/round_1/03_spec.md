## Implementation Specification: Adaptive Prompt Specialization

### 1. **Mechanism name**
`adaptive_prompt_specializer`

### 2. **Implementation strategy**
**new_helper_class** - Create a standalone helper class that manages phase detection and prompt specialization, similar to ElitePool and CrashMemory.

### 3. **Target**
New class: `AdaptivePromptSpecializer`

### 4. **Interface**
```python
class AdaptivePromptSpecializer:
    def __init__(
        self,
        stagnation_window: int = 4,
        exploration_window: int = 3,
        min_improvement_threshold: float = 0.0003
    ):
        """
        Args:
            stagnation_window: Number of recent iterations to analyze for stagnation
            exploration_window: Number of recent iterations to analyze for exploration patterns
            min_improvement_threshold: Minimum bpb improvement to consider as progress
        """
        
    def update(self, iteration: int, result: TrainResult, current_config: dict) -> None:
        """Update internal state with latest iteration results."""
        
    def get_specialized_instructions(self) -> str:
        """Return specialized prompt instructions based on current phase."""
        
    def get_phase(self) -> str:
        """Return current phase: 'exploration', 'exploitation', or 'stagnation'."""
```

### 5. **Step-by-step logic**

#### **5.1 Class initialization**
```python
def __init__(self, stagnation_window=4, exploration_window=3, min_improvement_threshold=0.0003):
    self.stagnation_window = stagnation_window
    self.exploration_window = exploration_window
    self.min_improvement = min_improvement_threshold
    
    # State tracking
    self.history = []  # List of dicts: {'iteration', 'bpb', 'config', 'accepted', 'discard_reason'}
    self.phase = 'exploration'  # Initial phase
    self.last_phase_change = 0
    
    # Pattern detection
    self.discard_patterns = {
        'parameter': {},  # e.g., {'weight_decay': {'count': 3, 'last_iterations': [5,6,7]}}
        'category': {}   # e.g., {'learning_rate': {'count': 2, 'last_iterations': [4,6]}}
    }
    
    # Phase-specific counters
    self.iterations_in_current_phase = 0
    self.improvements_in_phase = 0
```

#### **5.2 Update method**
```python
def update(self, iteration: int, result: TrainResult, current_config: dict):
    # 1. Record history entry
    entry = {
        'iteration': iteration,
        'bpb': result.bpb if result else None,
        'config': current_config,
        'accepted': result.accepted if result else False,
        'discard_reason': result.discard_reason if hasattr(result, 'discard_reason') else None
    }
    self.history.append(entry)
    
    # 2. Update discard patterns for recent failures
    if result and not result.accepted and hasattr(result, 'discard_reason'):
        self._update_discard_patterns(iteration, current_config, result.discard_reason)
    
    # 3. Detect phase transition
    self._detect_phase_transition(iteration)
    
    # 4. Increment phase counter
    self.iterations_in_current_phase += 1
```

#### **5.3 Phase detection logic**
```python
def _detect_phase_transition(self, current_iteration: int):
    # Need at least stagnation_window entries to make decisions
    if len(self.history) < self.stagnation_window:
        return
    
    recent = self.history[-self.stagnation_window:]
    
    # Check for stagnation: no improvement for N iterations
    improvements = [entry for entry in recent 
                   if entry['bpb'] and entry['accepted']]
    
    if len(improvements) == 0:
        # No accepted improvements in window → stagnation
        if self.phase != 'stagnation':
            self.phase = 'stagnation'
            self.last_phase_change = current_iteration
            self.iterations_in_current_phase = 0
            self.improvements_in_phase = 0
        return
    
    # Check if we should switch to exploitation
    if self.phase == 'exploration':
        # If we found promising region, switch to exploitation
        recent_best = min([entry['bpb'] for entry in recent if entry['bpb']])
        if self._is_significant_improvement(recent_best):
            self.phase = 'exploitation'
            self.last_phase_change = current_iteration
            self.iterations_in_current_phase = 0
            self.improvements_in_phase = 0
    
    # Check if we should switch back to exploration
    elif self.phase in ['exploitation', 'stagnation']:
        if self.iterations_in_current_phase >= 3 and self.improvements_in_phase == 0:
            self.phase = 'exploration'
            self.last_phase_change = current_iteration
            self.iterations_in_current_phase = 0
            self.improvements_in_phase = 0
```

#### **5.4 Get specialized instructions**
```python
def get_specialized_instructions(self) -> str:
    base = "Your goal is to improve validation bpb by modifying hyperparameters in train.py."
    
    if self.phase == 'exploration':
        instructions = self._get_exploration_instructions()
    elif self.phase == 'exploitation':
        instructions = self._get_exploitation_instructions()
    else:  # stagnation
        instructions = self._get_stagnation_instructions()
    
    # Add pattern warnings if any
    pattern_warnings = self._get_pattern_warnings()
    
    return f"{base}\n\n{instructions}\n\n{pattern_warnings}"

def _get_exploration_instructions(self):
    return """EXPLORATION PHASE: We're searching for promising regions.
    - Try substantially different hyperparameter combinations
    - Consider changing multiple parameters at once
    - Explore less common values (e.g., very small/large learning rates)
    - Don't be afraid to make bold changes - we need to find new promising areas"""

def _get_exploitation_instructions(self):
    return """EXPLOITATION PHASE: We've found a promising region.
    - Make small, incremental adjustments to fine-tune
    - Focus on one parameter at a time
    - Use gradient-like thinking: if increasing helped, try increasing more
    - Look for optimal values within this neighborhood"""

def _get_stagnation_instructions(self):
    recent_discards = self._get_recent_discard_summary()
    return f"""STAGNATION PHASE: We're stuck in a local optimum.
    - Break out of current patterns completely
    - Avoid parameters that have failed recently: {recent_discards}
    - Try orthogonal changes (if changing learning rate failed, try changing batch size)
    - Consider resetting some parameters to their original values
    - Look for combinations you haven't tried before"""
```

#### **5.5 Pattern warning generation**
```python
def _get_pattern_warnings(self):
    warnings = []
    
    # Check for parameter-specific patterns
    for param, data in self.discard_patterns['parameter'].items():
        if data['count'] >= 3:
            last_tries = data['last_iterations'][-3:]
            warnings.append(
                f"Warning: Last 3 attempts with {param} were discarded "
                f"(iterations {last_tries}). Consider avoiding this parameter."
            )
    
    # Check for category patterns
    for category, data in self.discard_patterns['category'].items():
        if data['count'] >= 4:
            warnings.append(
                f"Warning: Multiple recent failures in {category} adjustments. "
                f"Try a different type of change."
            )
    
    if warnings:
        return "RECENT PATTERNS TO CONSIDER:\n" + "\n".join(warnings)
    return ""
```

### 6. **Integration points**

#### **6.1 Modify TrainRunner.__init__**
```python
def __init__(self, ...):
    # ... existing code ...
    
    # Improvement 4: Adaptive prompt specialization
    self.prompt_specializer = AdaptivePromptSpecializer(
        stagnation_window=4,
        exploration_window=3,
        min_improvement_threshold=0.0003
    )
```

#### **6.2 Modify run_iteration method**
```python
def run_iteration(self, iteration: int) -> TrainResult:
    # ... existing code ...
    
    # Before LLM proposal, get specialized instructions
    specialized_instructions = self.prompt_specializer.get_specialized_instructions()
    
    # Modify the prompt generation to include these instructions
    # (Assuming there's a method that builds the prompt for LLM)
    prompt = self._build_proposal_prompt(
        current_config=current_config,
        trace=self.trace,
        specialized_instructions=specialized_instructions,
        # ... other context ...
    )
    
    # ... rest of iteration logic ...
    
    # After iteration completes, update the specializer
    self.prompt_specializer.update(
        iteration=iteration,
        result=result,
        current_config=current_config
    )
    
    return result
```

#### **6.3 Example prompt integration**
```python
def _build_proposal_prompt(self, current_config, trace, specialized_instructions, ...):
    prompt_template = """
You are optimizing hyperparameters for a neural language model.

{specialized_instructions}

Current hyperparameters:
{current_config}

Recent history (last {n} iterations):
{history}

Recent best configurations:
{elites}

Please propose changes to improve validation bpb...
"""
    
    return prompt_template.format(
        specialized_instructions=specialized_instructions,
        # ... other formatting ...
    )
```

### **Key Design Decisions**

1. **Phase detection uses simple heuristics** based on improvement patterns rather than complex statistical tests
2. **Pattern warnings are concrete and actionable** (e.g., "Last 3 attempts with weight_decay failed")
3. **Instructions are phase-specific** but maintain the same overall goal
4. **Integration is minimal** - only requires adding the specializer to prompt generation
5. **State is self-contained** in the helper class for easy testing and debugging

### **Testing Considerations**
- Unit test phase transitions with synthetic history sequences
- Test pattern detection with simulated discard patterns  
- Verify instructions are appropriate for each phase
- Ensure no interference with existing improvements (ElitePool, PlateauDetector, etc.)