import json

from src.llm_client import call_llm
from src.pipeline.base import BaseStage


class HypothesisValidationLoopStage(BaseStage):
    name = "hypothesis_validation_loop"

    def __init__(self, model: str = ""):
        super().__init__(model)

    def run(self, context: dict) -> dict:
        previous_outputs = context.get("previous_outputs", {})
        original_hypothesis = previous_outputs.get("improvement_hypotheses", {}).get("selected_hypothesis", "")
        if not original_hypothesis:
            original_hypothesis = "No hypothesis selected from previous stage."

        validation_rounds = []
        current_round = 1
        max_rounds = 3
        final_validation_status = None
        refined_hypothesis = original_hypothesis
        validation_summary = ""

        while current_round <= max_rounds:
            # Step 2.1: Generate validation test
            test_gen_prompt = f"""You are a rigorous hypothesis validator. Given this improvement hypothesis:

HYPOTHESIS: {original_hypothesis}

Generate ONE concrete, falsifiable test that would validate or disprove this hypothesis.

CRITERIA:
1. Test must be executable within the current system context
2. Test must have clear PASS/FAIL criteria
3. Test must directly test the core claim of the hypothesis
4. Test should be minimal and focused

Previous validation attempts (if any):
{self._format_validation_rounds(validation_rounds)}

Output format:
Test Description: [A specific action or check to perform]
Expected Outcome: [What should happen if hypothesis is correct]
Failure Conditions: [What would disprove the hypothesis]

Return ONLY the three labeled sections, no markdown, no additional commentary."""
            test_gen_response = call_llm(test_gen_prompt, model=self.model, max_tokens=300)
            test_description, expected_outcome, failure_conditions = self._parse_test_generation(test_gen_response)

            # Step 2.2: Execute validation test
            test_exec_prompt = f"""Execute this validation test for the hypothesis:

HYPOTHESIS: {original_hypothesis}
TEST: {test_description}

Based on the system's capabilities and the article analysis context, determine if this test PASSES or FAILS.

Consider:
1. Is the test logically sound?
2. Does available evidence support the expected outcome?
3. Are there any logical flaws in the test itself?

Provide specific evidence for your determination.

Output format:
Result: [PASS/FAIL/INCONCLUSIVE]
Evidence: [Specific reasoning and references]
Confidence: [HIGH/MEDIUM/LOW]

Return ONLY the three labeled lines, evidence 2-4 sentences."""
            test_exec_response = call_llm(test_exec_prompt, model=self.model, max_tokens=400)
            test_result, evidence, confidence = self._parse_test_execution(test_exec_response)

            # Step 2.3: Record round
            validation_rounds.append({
                "round": current_round,
                "test": test_description,
                "result": test_result,
                "evidence": evidence,
                "confidence": confidence
            })

            # Step 2.4: Check for high-confidence failure
            if test_result == "FAIL" and confidence == "HIGH":
                final_validation_status = "FAILED"
                break

            # Step 2.5: Check for high-confidence pass with at least 2 rounds
            if test_result == "PASS" and confidence == "HIGH":
                if current_round >= 2:
                    final_validation_status = "VALIDATED"
                    break

            current_round += 1

        # Step 3: Determine final outcome
        if final_validation_status is None:
            final_validation_status = "INCONCLUSIVE"

        if final_validation_status == "VALIDATED":
            refined_hypothesis = original_hypothesis
            validation_summary = f"Hypothesis validated through {len(validation_rounds)} successful tests"
        elif final_validation_status == "FAILED":
            # Generate refined hypothesis
            refine_prompt = f"""The original hypothesis has failed validation:

ORIGINAL HYPOTHESIS: {original_hypothesis}

Validation History:
{json.dumps(validation_rounds, indent=2)}

The hypothesis failed because: {validation_rounds[-1]['evidence'] if validation_rounds else 'Unknown'}

Generate a refined version of this hypothesis that:
1. Addresses the specific failure mode
2. Maintains the original intent
3. Is more specific and testable
4. Avoids the logical flaw that caused the failure

Output format:
Refined Hypothesis: [Your revised hypothesis statement]

Return ONLY the single line starting with 'Refined Hypothesis: '."""
            refine_response = call_llm(refine_prompt, model=self.model, max_tokens=200)
            refined_hypothesis = self._parse_refined_hypothesis(refine_response)
            validation_summary = f"Hypothesis failed test round {validation_rounds[-1]['round']}: {validation_rounds[-1]['evidence'][:100]}..."
        else:  # INCONCLUSIVE
            refined_hypothesis = original_hypothesis
            validation_summary = f"Insufficient evidence after {len(validation_rounds)} validation attempts"

        # Build output
        output = {
            "validated_hypothesis": {
                "original_hypothesis": original_hypothesis,
                "validation_rounds": validation_rounds,
                "final_validation_status": final_validation_status,
                "validation_summary": validation_summary,
                "refined_hypothesis": refined_hypothesis
            }
        }

        # Save artifacts
        artifacts = []
        artifacts.append(self._save_artifact(context["run_dir"], "validation_loop.md", json.dumps(output, indent=2)))
        artifacts.append(self._save_artifact(context["run_dir"], "validation_loop.json", json.dumps(output, indent=2)))

        return {"content": json.dumps(output, indent=2), "artifacts": artifacts}

    def _format_validation_rounds(self, validation_rounds):
        if not validation_rounds:
            return "None"
        formatted = []
        for r in validation_rounds:
            formatted.append(f"Round {r['round']}: {r['test']} -> {r['result']} ({r['confidence']})")
        return "\n".join(formatted)

    def _parse_test_generation(self, response):
        lines = response.strip().split('\n')
        test_desc = ""
        expected = ""
        failure = ""
        for line in lines:
            if line.startswith("Test Description:"):
                test_desc = line.replace("Test Description:", "", 1).strip()
            elif line.startswith("Expected Outcome:"):
                expected = line.replace("Expected Outcome:", "", 1).strip()
            elif line.startswith("Failure Conditions:"):
                failure = line.replace("Failure Conditions:", "", 1).strip()
        return test_desc, expected, failure

    def _parse_test_execution(self, response):
        lines = response.strip().split('\n')
        result = ""
        evidence = ""
        confidence = ""
        for line in lines:
            if line.startswith("Result:"):
                result = line.replace("Result:", "", 1).strip().upper()
            elif line.startswith("Evidence:"):
                evidence = line.replace("Evidence:", "", 1).strip()
            elif line.startswith("Confidence:"):
                confidence = line.replace("Confidence:", "", 1).strip().upper()
        return result, evidence, confidence

    def _parse_refined_hypothesis(self, response):
        lines = response.strip().split('\n')
        for line in lines:
            if line.startswith("Refined Hypothesis:"):
                return line.replace("Refined Hypothesis:", "", 1).strip()
        return "Refinement not generated."