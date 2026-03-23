## **Implementation Specification: Adaptive Meta-Prompting with Phase Detection**

### 1. **Mechanism name**
`adaptive_meta_prompting`

### 2. **Implementation strategy**
`new_helper_class` + `modify_init` + `replace_method`

### 3. **Target**
- New class: `PhaseDetector`
- Modified method: `TrainRunner._build_proposal_prompt()` (replaces existing prompt construction)
- New attributes in `TrainRunner.__init__`:
  - `phase_detector: PhaseDetector`
  - `_current_phase: str`
  - `_phase_history: list[tuple[int, str]]`  # (iteration, phase)

### 4. **Interface**

#### **PhaseDetector Class**
```python
class PhaseDetector:
    """Detects optimization phases based on performance trends."""
    
    def __init__(
        self,
        exploration_window: int = 3,
        exploitation_window: int = 4,
        stagnation_threshold: int = 3,
        min_improvement: float = 0.0003
    ):
        """
        Args:
            exploration_window: Consecutive improvements needed to enter "exploitation"
            exploitation_window: Consecutive non-improvements needed to enter "exploration"
            stagnation_threshold: Iterations with < min_improvement to detect "plateau"
            min_improvement: Minimum bpb improvement to count as progress
        """
        self.exploration_window = exploration_window
        self.exploitation_window = exploitation_window
        self.stagnation_threshold = stagnation_threshold
        self.min_improvement = min_improvement
        
        # State tracking
        self.improvement_streak = 0
        self.non_improvement_streak = 0
        self.plateau_counter = 0
        self.last_best_bpb = float('inf')
    
    def update_and_detect(self, iteration: int, current_bpb: float) -> str:
        """
        Update internal state and return current phase.
        
        Returns:
            One of: "exploration", "exploitation", "plateau", "reflection"
        """
        # Calculate improvement
        improvement = self.last_best_bpb - current_bpb
        has_improved = improvement > self.min_improvement
        
        # Update streaks
        if has_improved:
            self.improvement_streak += 1
            self.non_improvement_streak = 0
            self.plateau_counter = 0
            self.last_best_bpb = current_bpb
        else:
            self.non_improvement_streak += 1
            self.improvement_streak = 0
            if improvement > -self.min_improvement:  # Within noise margin
                self.plateau_counter += 1
            else:
                self.plateau_counter = 0
        
        # Phase detection logic
        if self.plateau_counter >= self.stagnation_threshold:
            return "plateau"
        elif self.improvement_streak >= self.exploration_window:
            return "exploitation"
        elif self.non_improvement_streak >= self.exploitation_window:
            return "exploration"
        elif iteration % 5 == 0:  # Periodic reflection
            return "reflection"
        else:
            # Default: continue current strategy
            return "exploitation" if self.improvement_streak > 0 else "exploration"
```

#### **Modified TrainRunner._build_proposal_prompt()**
```python
def _build_proposal_prompt(
    self,
    current_config: dict[str, Any],
    iteration: int,
    phase: str
) -> str:
    """
    Build phase-aware prompt for LLM proposals.
    
    Args:
        current_config: Current hyperparameter values
        iteration: Current iteration number
        phase: One of "exploration", "exploitation", "plateau", "reflection"
    
    Returns:
        Formatted prompt string
    """
```

### 5. **Step-by-step logic**

#### **PhaseDetector.update_and_detect()**
1. **Input**: `iteration` (int), `current_bpb` (float)
2. **Calculate improvement**: `improvement = last_best_bpb - current_bpb`
3. **Determine if improved**: `has_improved = improvement > min_improvement`
4. **Update tracking variables**:
   - If improved: increment `improvement_streak`, reset others, update `last_best_bpb`
   - If not improved: increment `non_improvement_streak`
   - If within noise margin (`abs(improvement) <= min_improvement`): increment `plateau_counter`
5. **Phase decision tree**:
   ```
   if plateau_counter >= stagnation_threshold:
       return "plateau"
   elif improvement_streak >= exploration_window:
       return "exploitation"
   elif non_improvement_streak >= exploitation_window:
       return "exploration"
   elif iteration % 5 == 0:
       return "reflection"
   else:
       return "exploitation" if improvement_streak > 0 else "exploration"
   ```

#### **TrainRunner._build_proposal_prompt()**
1. **Base prompt**: Include standard elements (current config, crash memory, elite pool, etc.)
2. **Phase-specific instructions**:
   ```
   if phase == "exploration":
       Add: "You are in EXPLORATION phase. Prioritize DIVERSITY over refinement. 
            Try significantly different hyperparameter combinations, even if risky.
            Consider changing multiple parameters at once."
   
   elif phase == "exploitation":
       Add: "You are in EXPLOITATION phase. Focus on REFINEMENT. Make small, 
            targeted adjustments to the best-performing configuration.
            Change only 1-2 parameters at a time."
   
   elif phase == "plateau":
       Add: "You are in PLATEAU phase. The optimization has stagnated.
            CRITICALLY ANALYZE why progress stopped. Consider:
            - Are we in a local minimum?
            - Should we change optimization strategy?
            - Are there parameter interactions we're missing?"
   
   elif phase == "reflection":
       Add: "You are in REFLECTION phase. Review the last 5 iterations.
            What patterns do you see in successful vs failed proposals?
            What assumptions might be wrong? Propose 1-2 'hypothesis tests'
            to validate your understanding."
   ```
3. **Phase context**: Append recent phase history
4. **Return**: Complete formatted prompt

#### **Integration in run_iteration()**
1. After getting trial result, update phase detector
2. Pass phase to `_build_proposal_prompt()`
3. Store phase in history for debugging

### 6. **Integration points**

#### **TrainRunner.__init__() modifications**
```python
def __init__(self, ...):
    # Existing code...
    
    # Improvement 17: Adaptive meta-prompting
    self.phase_detector = PhaseDetector(
        exploration_window=3,
        exploitation_window=4,
        stagnation_threshold=3,
        min_improvement=0.0003
    )
    self._current_phase = "exploration"
    self._phase_history = []  # (iteration, phase)
```

#### **TrainRunner.run_iteration() modifications**
```python
def run_iteration(self, iteration: int) -> TrainResult:
    # ... existing code until after trial execution ...
    
    # Update phase detection
    self._current_phase = self.phase_detector.update_and_detect(
        iteration, 
        result.bpb if result.success else float('inf')
    )
    self._phase_history.append((iteration, self._current_phase))
    
    # Build phase-aware prompt
    prompt = self._build_proposal_prompt(
        current_config, 
        iteration, 
        self._current_phase
    )
    
    # ... continue with existing LLM call ...
```

#### **TrainRunner._build_proposal_prompt() replacement**
Replace the existing prompt construction with the phase-aware version described above. The method should:
1. Accept the new `phase` parameter
2. Incorporate phase-specific guidance
3. Maintain all existing prompt components (crash memory, elite pool, etc.)
4. Add phase context section showing recent transitions

#### **Debugging integration**
Add to `TrainRunner`:
```python
def get_phase_summary(self) -> dict:
    """Return phase detection state for debugging."""
    return {
        "current_phase": self._current_phase,
        "phase_history": self._phase_history[-10:],  # Last 10
        "detector_state": {
            "improvement_streak": self.phase_detector.improvement_streak,
            "non_improvement_streak": self.phase_detector.non_improvement_streak,
            "plateau_counter": self.phase_detector.plateau_counter,
            "last_best_bpb": self.phase_detector.last_best_bpb
        }
    }
```

### **Fallback behavior**
- If phase detection fails (e.g., no valid bpb), default to "exploration"
- The LLM should still receive a valid prompt even with default phase
- Log phase transitions for monitoring: `logger.info(f"Phase transition: {old_phase} → {new_phase}")`

### **Testing considerations**
1. Unit test `PhaseDetector` with synthetic improvement sequences
2. Verify prompt templates contain phase-specific instructions
3. Test that phase transitions occur at expected thresholds
4. Ensure backward compatibility: system works even if phase detection is disabled