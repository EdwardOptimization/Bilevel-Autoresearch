**Mechanism name**: `fixation_detector`

**Implementation strategy**: new_helper_class

**Target**: New class `FixationDetector` to be instantiated in `TrainRunner.__init__`

**Interface**:
```python
class FixationDetector:
    def __init__(
        self,
        window_size: int = 5,
        similarity_threshold: float = 0.7,
        max_repetitions: int = 3
    ):
        """
        window_size: How many recent proposals to track
        similarity_threshold: Jaccard similarity threshold to consider proposals "similar"
        max_repetitions: How many similar proposals allowed before triggering fixation
        """
        
    def add_proposal(self, proposal_text: str, iteration: int) -> None:
        """Record a new proposal for analysis"""
        
    def check_fixation(self) -> tuple[bool, str, dict]:
        """
        Returns:
            - is_fixated (bool): True if fixation detected
            - pattern (str): Description of the fixation pattern
            - details (dict): Diagnostic information for logging
        """
        
    def get_intervention_suggestion(self) -> str:
        """Returns a concrete suggestion to break the fixation pattern"""
        
    def reset(self) -> None:
        """Clear history after successful intervention"""
```

**Step-by-step logic**:

1. **Initialization**:
   - Store `window_size`, `similarity_threshold`, `max_repetitions`
   - Initialize empty lists: `proposal_history` (strings), `iteration_history` (ints)
   - Initialize `fixation_counter = 0`

2. **add_proposal(proposal_text, iteration)**:
   - Clean proposal text: remove comments, normalize whitespace, extract parameter changes
   - Append cleaned text to `proposal_history`
   - Append iteration number to `iteration_history`
   - Trim both lists to last `window_size` entries

3. **check_fixation()**:
   - If history has fewer than 2 entries: return `(False, "", {})`
   - For each pair of recent proposals (last N vs previous N-1), compute similarity:
     a. Tokenize each proposal into parameter change operations
     b. Compute Jaccard similarity: `len(intersection) / len(union)`
   - If any similarity > `similarity_threshold`:
     - Increment `fixation_counter`
     - If `fixation_counter >= max_repetitions`:
       - Identify the most common parameter being modified
       - Return `(True, f"Repeated modifications to {parameter}", diagnostics)`
   - Else:
     - Reset `fixation_counter = 0`
     - Return `(False, "", {})`

4. **get_intervention_suggestion()**:
   - Analyze the fixation pattern:
     - If fixation on a specific parameter (e.g., "EMBEDDING_LR"):
       Return "Try modifying a different parameter like HIDDEN_SIZE or DROPOUT instead"
     - If fixation on direction (e.g., "always increasing"):
       Return "Try opposite direction or explore orthogonal dimensions"
     - Default: "Force exploration by modifying untouched parameters"

5. **reset()**:
   - Clear `proposal_history`, `iteration_history`
   - Reset `fixation_counter = 0`

**Integration points**:

1. **In TrainRunner.__init__**:
```python
# Add after other helper classes
self.fixation_detector = FixationDetector(
    window_size=5,
    similarity_threshold=0.7,
    max_repetitions=3
)
```

2. **In run_iteration()** (after LLM generates proposal):
```python
# After getting proposal from LLM, before applying changes
proposal_text = ...  # The LLM's proposed changes

# Track proposal
self.fixation_detector.add_proposal(proposal_text, iteration)

# Check for fixation
is_fixated, pattern, details = self.fixation_detector.check_fixation()
if is_fixated:
    logger.warning(f"Fixation detected: {pattern}")
    logger.debug(f"Fixation details: {details}")
    
    # Get intervention suggestion
    intervention = self.fixation_detector.get_intervention_suggestion()
    
    # Modify the prompt to include fixation-breaking instruction
    # This would integrate with the existing prompt engineering
    # For now, log and potentially override the proposal
    if self._should_override_fixation():
        # Generate alternative proposal
        alternative = self._generate_alternative_proposal(intervention)
        # Use alternative instead of original
        proposal_text = alternative
        
        # Reset detector after successful intervention
        self.fixation_detector.reset()
```

3. **New helper method in TrainRunner**:
```python
def _should_override_fixation(self) -> bool:
    """Decide whether to override a fixated proposal"""
    # Consider: iteration number, recent progress, exploration budget
    if self.plateau_detector.is_in_plateau():
        return True
    if self._exploration_budget_available():
        return True
    return False

def _generate_alternative_proposal(self, intervention_hint: str) -> str:
    """Generate an alternative proposal using the intervention hint"""
    # Could: modify the original proposal, generate fresh from LLM with hint,
    # or use a rule-based alternative
    # For simplicity, return a modified version of the original
    return f"{intervention_hint}\n\nOriginal: {self.current_code}"
```

**Key design decisions**:
1. Uses Jaccard similarity on parameter change operations (not raw text)
2. Configurable thresholds to avoid false positives
3. Provides concrete intervention suggestions
4. Resets after successful intervention to avoid perpetual fixation state
5. Integrates with existing plateau detection and exploration mechanisms