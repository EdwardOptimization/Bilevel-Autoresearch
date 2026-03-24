import re
from collections import Counter
from typing import List, Tuple, Dict, Optional

class FixationDetector:
    """
    Detects when the LLM is stuck proposing similar parameter changes repeatedly.
    """
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
        self.window_size = window_size
        self.similarity_threshold = similarity_threshold
        self.max_repetitions = max_repetitions
        
        self.proposal_history: List[str] = []
        self.iteration_history: List[int] = []
        self.fixation_counter = 0
        
    def add_proposal(self, proposal_text: str, iteration: int) -> None:
        """Record a new proposal for analysis"""
        cleaned = self._clean_proposal(proposal_text)
        self.proposal_history.append(cleaned)
        self.iteration_history.append(iteration)
        
        # Keep only the most recent window_size entries
        if len(self.proposal_history) > self.window_size:
            self.proposal_history = self.proposal_history[-self.window_size:]
            self.iteration_history = self.iteration_history[-self.window_size:]
        
    def check_fixation(self) -> Tuple[bool, str, dict]:
        """
        Returns:
            - is_fixated (bool): True if fixation detected
            - pattern (str): Description of the fixation pattern
            - details (dict): Diagnostic information for logging
        """
        if len(self.proposal_history) < 2:
            return False, "", {}
        
        # Check similarity between recent proposals
        recent_similar = False
        if len(self.proposal_history) >= 2:
            # Compare the most recent proposal with previous ones in the window
            latest = self.proposal_history[-1]
            for prev in self.proposal_history[-min(3, len(self.proposal_history)-1):-1]:
                similarity = self._jaccard_similarity(latest, prev)
                if similarity > self.similarity_threshold:
                    recent_similar = True
                    break
        
        if recent_similar:
            self.fixation_counter += 1
            if self.fixation_counter >= self.max_repetitions:
                # Identify the most common parameter being modified
                common_param = self._identify_fixated_parameter()
                pattern = f"Repeated modifications to {common_param}"
                details = {
                    'fixation_counter': self.fixation_counter,
                    'window_size': len(self.proposal_history),
                    'common_parameter': common_param,
                    'recent_proposals': self.proposal_history[-3:] if len(self.proposal_history) >= 3 else self.proposal_history
                }
                return True, pattern, details
        else:
            self.fixation_counter = 0
        
        return False, "", {}
        
    def get_intervention_suggestion(self) -> str:
        """Returns a concrete suggestion to break the fixation pattern"""
        if not self.proposal_history:
            return "Force exploration by modifying untouched parameters"
        
        # Analyze the fixation pattern
        common_param = self._identify_fixated_parameter()
        
        if common_param and common_param != "unknown":
            # Fixation on a specific parameter
            suggestions = {
                'EMBEDDING_LR': "Try modifying a different parameter like HIDDEN_SIZE or DROPOUT instead",
                'HIDDEN_SIZE': "Try modifying a different parameter like EMBEDDING_LR or DROPOUT instead",
                'DROPOUT': "Try modifying a different parameter like EMBEDDING_LR or HIDDEN_SIZE instead",
                'LEARNING_RATE': "Try modifying a different parameter like BATCH_SIZE or WEIGHT_DECAY instead",
                'BATCH_SIZE': "Try modifying a different parameter like LEARNING_RATE or GRADIENT_ACCUMULATION instead",
            }
            return suggestions.get(common_param, f"Try modifying a different parameter instead of {common_param}")
        
        # Check for directional fixation
        direction_pattern = self._detect_directional_pattern()
        if direction_pattern:
            if "increasing" in direction_pattern or "decreasing" in direction_pattern:
                return "Try opposite direction or explore orthogonal dimensions"
        
        # Default suggestion
        return "Force exploration by modifying untouched parameters"
        
    def reset(self) -> None:
        """Clear history after successful intervention"""
        self.proposal_history.clear()
        self.iteration_history.clear()
        self.fixation_counter = 0
    
    def _clean_proposal(self, proposal_text: str) -> str:
        """Clean proposal text and extract parameter changes"""
        # Remove comments
        lines = proposal_text.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove Python comments
            if '#' in line:
                line = line[:line.index('#')]
            cleaned_lines.append(line.strip())
        
        # Join and normalize whitespace
        text = ' '.join(cleaned_lines)
        text = re.sub(r'\s+', ' ', text)
        
        # Extract parameter change patterns
        param_changes = []
        
        # Look for common parameter assignment patterns
        patterns = [
            r'(\w+)\s*=\s*[\d\.]+',  # param = value
            r'(\w+)\s*:\s*[\d\.]+',  # param: value
            r'(\w+)\s*\+=\s*[\d\.]+',  # param += value
            r'(\w+)\s*-=\s*[\d\.]+',  # param -= value
            r'(\w+)\s*\*=\s*[\d\.]+',  # param *= value
            r'(\w+)\s*/=\s*[\d\.]+',  # param /= value
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            param_changes.extend(matches)
        
        # Also look for common hyperparameter names
        common_params = ['EMBEDDING_LR', 'HIDDEN_SIZE', 'DROPOUT', 'LEARNING_RATE', 
                        'BATCH_SIZE', 'WEIGHT_DECAY', 'GRADIENT_ACCUMULATION',
                        'WARMUP_STEPS', 'MAX_STEPS', 'SEQ_LEN']
        
        for param in common_params:
            if param.lower() in text.lower():
                param_changes.append(param)
        
        # Return unique parameter changes
        return ' '.join(sorted(set(param_changes))) if param_changes else text[:100]  # Fallback to truncated text
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Compute Jaccard similarity between two texts based on tokens"""
        tokens1 = set(text1.split())
        tokens2 = set(text2.split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1.intersection(tokens2))
        union = len(tokens1.union(tokens2))
        
        return intersection / union if union > 0 else 0.0
    
    def _identify_fixated_parameter(self) -> str:
        """Identify the most common parameter in recent proposals"""
        if not self.proposal_history:
            return "unknown"
        
        # Extract all parameter names from recent proposals
        all_params = []
        for proposal in self.proposal_history[-self.window_size:]:
            # Simple token-based extraction (parameters are usually alphanumeric with underscores)
            tokens = proposal.split()
            for token in tokens:
                if re.match(r'^[A-Z_][A-Z0-9_]*$', token):
                    all_params.append(token)
        
        if not all_params:
            return "unknown"
        
        # Find most common parameter
        counter = Counter(all_params)
        most_common = counter.most_common(1)[0]
        return most_common[0]
    
    def _detect_directional_pattern(self) -> Optional[str]:
        """Detect if proposals show a consistent directional pattern"""
        # This is a simplified version - in practice would need more sophisticated analysis
        if len(self.proposal_history) < 3:
            return None
        
        # Look for patterns like "increase", "decrease", "larger", "smaller" in proposals
        recent_text = ' '.join(self.proposal_history[-3:]).lower()
        
        increase_indicators = ['increase', 'larger', 'bigger', 'higher', 'more', 'add', '+']
        decrease_indicators = ['decrease', 'smaller', 'lower', 'less', 'reduce', 'subtract', '-']
        
        increase_count = sum(1 for indicator in increase_indicators if indicator in recent_text)
        decrease_count = sum(1 for indicator in decrease_indicators if indicator in recent_text)
        
        if increase_count > decrease_count and increase_count >= 2:
            return "consistently increasing"
        elif decrease_count > increase_count and decrease_count >= 2:
            return "consistently decreasing"
        
        return None