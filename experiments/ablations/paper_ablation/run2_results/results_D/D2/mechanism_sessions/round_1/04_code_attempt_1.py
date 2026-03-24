import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class GeneratedMechanism_20260324_142015:
    """
    Adaptive Prompt Specializer: Dynamically adjusts LLM instructions based on
    optimization phase (exploration, exploitation, stagnation) and failure patterns.
    """
    
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
        self.stagnation_window = stagnation_window
        self.exploration_window = exploration_window
        self.min_improvement = min_improvement_threshold
        
        # State tracking
        self.history: List[Dict[str, Any]] = []  # List of dicts: {'iteration', 'bpb', 'config', 'accepted', 'discard_reason'}
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
        
        # Parameter categories for pattern detection
        self._param_categories = {
            'learning_rate': 'learning_rate',
            'lr': 'learning_rate',
            'lr_schedule': 'learning_rate',
            'weight_decay': 'regularization',
            'dropout': 'regularization',
            'batch_size': 'batch_size',
            'gradient_clip': 'gradient',
            'optimizer': 'optimizer',
            'warmup_iters': 'schedule'
        }
    
    def update(self, iteration: int, result: Any, current_config: dict) -> None:
        """Update internal state with latest iteration results."""
        # 1. Record history entry
        entry = {
            'iteration': iteration,
            'bpb': result.val_bpb if result else None,
            'config': current_config,
            'accepted': result.status == "keep" if result else False,
            'discard_reason': getattr(result, 'discard_reason', None) if result else None
        }
        self.history.append(entry)
        
        # 2. Update discard patterns for recent failures
        if result and result.status != "keep" and hasattr(result, 'discard_reason'):
            self._update_discard_patterns(iteration, current_config, result.discard_reason)
        
        # 3. Update improvement counter for current phase
        if result and result.status == "keep":
            self.improvements_in_phase += 1
        
        # 4. Detect phase transition
        self._detect_phase_transition(iteration)
        
        # 5. Increment phase counter
        self.iterations_in_current_phase += 1
    
    def _update_discard_patterns(self, iteration: int, config: dict, discard_reason: str) -> None:
        """Track patterns in failed parameter changes."""
        if not config:
            return
            
        # Extract changed parameters from config (compared to previous)
        if self.history and len(self.history) > 1:
            prev_entry = self.history[-2]
            prev_config = prev_entry.get('config', {}) if prev_entry else {}
            
            # Simple detection: look at all params in current config
            for param, value in config.items():
                prev_value = prev_config.get(param)
                if prev_value is not None and prev_value != value:
                    # Parameter was changed
                    if param not in self.discard_patterns['parameter']:
                        self.discard_patterns['parameter'][param] = {
                            'count': 0,
                            'last_iterations': []
                        }
                    self.discard_patterns['parameter'][param]['count'] += 1
                    self.discard_patterns['parameter'][param]['last_iterations'].append(iteration)
                    
                    # Also track by category
                    category = self._param_categories.get(param, 'other')
                    if category not in self.discard_patterns['category']:
                        self.discard_patterns['category'][category] = {
                            'count': 0,
                            'last_iterations': []
                        }
                    self.discard_patterns['category'][category]['count'] += 1
                    self.discard_patterns['category'][category]['last_iterations'].append(iteration)
    
    def _detect_phase_transition(self, current_iteration: int) -> None:
        """Determine current optimization phase based on recent history."""
        # Need at least stagnation_window entries to make decisions
        if len(self.history) < self.stagnation_window:
            return
        
        recent = self.history[-self.stagnation_window:]
        
        # Check for stagnation: no improvement for N iterations
        improvements = [entry for entry in recent 
                       if entry['bpb'] is not None and entry['accepted']]
        
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
            recent_bpbs = [entry['bpb'] for entry in recent if entry['bpb'] is not None]
            if recent_bpbs:
                recent_best = min(recent_bpbs)
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
    
    def _is_significant_improvement(self, recent_best: float) -> bool:
        """Check if recent improvement is significant compared to history."""
        if len(self.history) < self.stagnation_window + 1:
            return False
        
        # Compare with earlier best
        earlier = self.history[:-self.stagnation_window]
        earlier_bpbs = [entry['bpb'] for entry in earlier if entry['bpb'] is not None]
        if not earlier_bpbs:
            return False
        
        earlier_best = min(earlier_bpbs)
        improvement = earlier_best - recent_best
        
        return improvement >= self.min_improvement
    
    def get_specialized_instructions(self) -> str:
        """Return specialized prompt instructions based on current phase."""
        base = "Your goal is to improve validation bpb by modifying hyperparameters in train.py."
        
        if self.phase == 'exploration':
            instructions = self._get_exploration_instructions()
        elif self.phase == 'exploitation':
            instructions = self._get_exploitation_instructions()
        else:  # stagnation
            instructions = self._get_stagnation_instructions()
        
        # Add pattern warnings if any
        pattern_warnings = self._get_pattern_warnings()
        
        if pattern_warnings:
            return f"{base}\n\n{instructions}\n\n{pattern_warnings}"
        else:
            return f"{base}\n\n{instructions}"
    
    def _get_exploration_instructions(self) -> str:
        return """EXPLORATION PHASE: We're searching for promising regions.
- Try substantially different hyperparameter combinations
- Consider changing multiple parameters at once
- Explore less common values (e.g., very small/large learning rates)
- Don't be afraid to make bold changes - we need to find new promising areas"""
    
    def _get_exploitation_instructions(self) -> str:
        return """EXPLOITATION PHASE: We've found a promising region.
- Make small, incremental adjustments to fine-tune
- Focus on one parameter at a time
- Use gradient-like thinking: if increasing helped, try increasing more
- Look for optimal values within this neighborhood"""
    
    def _get_stagnation_instructions(self) -> str:
        recent_discards = self._get_recent_discard_summary()
        return f"""STAGNATION PHASE: We're stuck in a local optimum.
- Break out of current patterns completely
- Avoid parameters that have failed recently: {recent_discards}
- Try orthogonal changes (if changing learning rate failed, try changing batch size)
- Consider resetting some parameters to their original values
- Look for combinations you haven't tried before"""
    
    def _get_recent_discard_summary(self) -> str:
        """Get summary of recently failed parameters."""
        if not self.discard_patterns['parameter']:
            return "none"
        
        recent_failures = []
        for param, data in self.discard_patterns['parameter'].items():
            if data['last_iterations'] and data['last_iterations'][-1] >= len(self.history) - 3:
                recent_failures.append(param)
        
        if recent_failures:
            return ", ".join(recent_failures[:3])  # Show at most 3
        return "none"
    
    def _get_pattern_warnings(self) -> str:
        """Generate warnings about repeated failure patterns."""
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
    
    def get_phase(self) -> str:
        """Return current phase: 'exploration', 'exploitation', or 'stagnation'."""
        return self.phase