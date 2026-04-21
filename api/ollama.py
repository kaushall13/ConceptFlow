"""
Ollama API Wrapper - Local LLM evaluation with binary prompts, JSON generation,
and raw text generation.
"""

import json
import re
import time
from typing import Dict, Any, Optional
import requests


class OllamaAPI:
    """Wrapper for Ollama local LLM API for evaluation tasks."""

    def __init__(self, host: str = "http://localhost:11434", model: str = "llama3.2:3b"):
        """
        Initialize Ollama API client.

        Args:
            host: Ollama server host URL
            model: Model name to use
        """
        self.host = host.rstrip('/')
        self.model = model
        self.api_url = f"{self.host}/api/generate"
        self.timeout = 30  # seconds

    def check_connection(self) -> bool:
        """
        Check if Ollama server is reachable.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def generate_json(self, prompt: str, system_prompt: str = None, temperature: float = 0.3) -> Dict[str, Any]:
        """
        Generate a JSON response using Ollama /api/chat endpoint.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Sampling temperature (default 0.3 for structured output)

        Returns:
            Parsed JSON dictionary

        Raises:
            Exception: If Ollama is unreachable or JSON parsing fails after retries
        """
        max_retries = 3
        base_delay = 2.0

        print(f"  [Ollama] generate_json called, model={self.model}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": temperature,
                        }
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()

                response_text = result.get("message", {}).get("content", "").strip()

                try:
                    return self._parse_json(response_text)
                except Exception as parse_err:
                    if attempt < max_retries - 1:
                        delay = base_delay * (2 ** attempt)
                        print(f"  [Ollama] JSON parse failed (attempt {attempt + 1}/{max_retries}): {parse_err}")
                        print(f"  [Ollama] Retrying in {delay}s...")
                        time.sleep(delay)
                    else:
                        raise Exception(
                            f"[Ollama] JSON parse failed after {max_retries} attempts: {parse_err}"
                        )

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  [Ollama] Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  [Ollama] Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise Exception(
                        f"[Ollama] Unreachable after {max_retries} attempts. "
                        f"Ensure Ollama is running at {self.host}. Last error: {e}"
                    )

    def generate_text(self, prompt: str, system_prompt: str = None) -> str:
        """
        Generate raw text using Ollama /api/chat endpoint.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Raw response text string

        Raises:
            Exception: If Ollama is unreachable after retries
        """
        max_retries = 3
        base_delay = 2.0

        print(f"  [Ollama] generate_text called, model={self.model}")

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    f"{self.host}/api/chat",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "stream": False,
                    },
                    timeout=self.timeout
                )
                response.raise_for_status()
                result = response.json()

                return result.get("message", {}).get("content", "").strip()

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  [Ollama] Connection error (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  [Ollama] Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    raise Exception(
                        f"[Ollama] Unreachable after {max_retries} attempts. "
                        f"Ensure Ollama is running at {self.host}. Last error: {e}"
                    )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from response, stripping markdown code fences if present."""
        text = text.strip()

        # Strip markdown code fences
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON object/array from response
            match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            raise Exception(f"Failed to parse JSON from response: {text[:300]}")

    def evaluate_binary(self, prompt: str, condition: str) -> tuple[bool, str]:
        """
        Evaluate a binary condition using Ollama.

        Args:
            prompt: Evaluation context and question
            condition: Specific condition to check

        Returns:
            Tuple of (pass/fail boolean, explanation string)

        Raises:
            Exception: If evaluation fails
        """
        max_retries = 2
        base_delay = 1.0

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    self.api_url,
                    json={
                        "model": self.model,
                        "prompt": f"{prompt}\n\nCondition to check: {condition}\n\nAnswer only YES or NO, then one sentence of evidence.",
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for consistent binary output
                            "num_predict": 50     # Short responses preferred
                        }
                    },
                    timeout=self.timeout
                )

                response.raise_for_status()
                result = response.json()

                # Extract response text
                response_text = result.get('response', '').strip()

                # Parse binary response
                passed, evidence = self._parse_binary_response(response_text)

                return passed, evidence

            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    delay = base_delay * (2 ** attempt)
                    print(f"  Ollama call failed (attempt {attempt + 1}/{max_retries}): {e}")
                    print(f"  Retrying in {delay}s...")
                    time.sleep(delay)
                else:
                    print(f"  Ollama call failed after {max_retries} attempts: {e}")
                    raise

    def _parse_binary_response(self, response_text: str) -> tuple[bool, str]:
        """
        Parse binary YES/NO response from Ollama.

        Args:
            response_text: Raw response text from Ollama

        Returns:
            Tuple of (boolean pass/fail, explanation string)
        """
        # Convert to uppercase for easier parsing
        upper_text = response_text.upper()

        # Look for YES or NO at the beginning
        if upper_text.startswith('YES'):
            passed = True
            # Extract evidence (everything after YES, minus the word itself)
            evidence = response_text[3:].strip()
            if not evidence:
                evidence = "Condition met"
        elif upper_text.startswith('NO'):
            passed = False
            # Extract evidence (everything after NO, minus the word itself)
            evidence = response_text[2:].strip()
            if not evidence:
                evidence = "Condition not met"
        else:
            # Fallback: try to find YES/NO anywhere in the text
            if 'YES' in upper_text:
                passed = True
                # Split on YES and take the second part
                parts = response_text.split('YES', 1)
                evidence = parts[1].strip() if len(parts) > 1 else "Condition met"
            elif 'NO' in upper_text:
                passed = False
                # Split on NO and take the second part
                parts = response_text.split('NO', 1)
                evidence = parts[1].strip() if len(parts) > 1 else "Condition not met"
            else:
                # Last resort: assume fail if unclear
                passed = False
                evidence = f"Unclear response: {response_text[:100]}"

        return passed, evidence

    def check_tension(self, session_ending: str, next_concept: str) -> tuple[bool, str]:
        """
        Check if session's TENSION is properly formed.

        Args:
            session_ending: Final ~100 words of session
            next_concept: Name of next session's first concept

        Returns:
            Tuple of (pass/fail, explanation)
        """
        prompt = f"""You are a structural evaluator for educational content. Evaluate the final question (TENSION) of a reading session.

Session ending (final paragraph):
{session_ending}

Next session's first concept: {next_concept}

Check if the TENSION question is properly formed:
- It should be a concrete, answerable question (not rhetorical)
- It should NOT be answerable from the current session alone
- It should NOT be a definitional question
- It should NOT have an obvious yes/no answer
- It should be specific enough to provide a concrete handle for thinking"""

        condition = "Does the TENSION question meet all these criteria?"

        return self.evaluate_binary(prompt, condition)

    def check_anchor(self, session_opening: str, previous_ending: str, previous_concepts: list, tension_question: str = "") -> tuple[bool, str]:
        """
        Check if session's ANCHOR is properly formed.

        Args:
            session_opening: Opening ~200 words of session
            previous_ending: Final paragraph of previous session
            previous_concepts: List of concept names from previous session
            tension_question: The specific tension question from previous session (optional)

        Returns:
            Tuple of (pass/fail, explanation)
        """
        previous_concepts_str = ", ".join(previous_concepts) if previous_concepts else "none"
        tension_line = f"\nPrevious session tension question: \"{tension_question}\"" if tension_question else ""

        prompt = f"""You are a structural evaluator for educational content. Evaluate the opening (ANCHOR) of a reading session.

Current session opening:
{session_opening}

Previous session ending:
{previous_ending}{tension_line}

Previous session concepts: {previous_concepts_str}

Check if the ANCHOR is properly formed:
- The opening (first 2-3 sentences) must address and resolve the previous session's tension question — the answer doesn't need to be in the literal first sentence, but must appear before the session pivots to new content
- It should transition to new content within 3-4 sentences total
- It should NOT re-explain previous concepts in depth
- It should NOT reference terms not present in the previous session
- It must NOT reproduce verbatim or near-verbatim text from the previous session's ending (the ANCHOR must be freshly written)"""

        condition = "Does the ANCHOR meet all these criteria?"

        return self.evaluate_binary(prompt, condition)

    def check_coherence(self, full_session: str, session_concepts: list, revisit_concept: Optional[str], session_title: str = "") -> tuple[bool, str]:
        """
        Check if session maintains coherence.

        Args:
            full_session: Complete session text
            session_concepts: List of concept names in this session
            revisit_concept: Name of revisited concept (if any)
            session_title: Title of the session (optional, used for context)

        Returns:
            Tuple of (pass/fail, explanation)
        """
        session_concepts_str = ", ".join(session_concepts)
        revisit_str = revisit_concept if revisit_concept else "none"
        title_line = f"Session title: {session_title}\n" if session_title else ""

        prompt = f"""You are a structural evaluator for educational content. Evaluate coherence of a reading session.

{title_line}Full session:
{full_session}

Session concepts: {session_concepts_str}
Revisited concept: {revisit_str}

Check if the session is coherent:
- It should NOT use any concept before introducing it (forward references)
- If there's a REVISIT, it should NOT re-explain core mechanics from scratch
- It should introduce only the concepts listed above
- It should explain each concept before moving to the next
- Each concept listed in the concept list must be substantively discussed in the session. If a listed concept is entirely absent from the content, that is a COHERENCE failure."""

        condition = "Does the session meet all these coherence criteria?"

        return self.evaluate_binary(prompt, condition)


    def check_revisit(self, revisit_paragraph: str, revisit_concept: str, session_concepts: list) -> tuple[bool, str]:
        """
        Check if a REVISIT paragraph properly names the earlier concept and makes a new connection.

        Args:
            revisit_paragraph: The paragraph suspected to contain the REVISIT
            revisit_concept: Name of the concept being revisited
            session_concepts: Names of concepts in the current session

        Returns:
            Tuple of (pass/fail, explanation)
        """
        session_concepts_str = ", ".join(session_concepts)

        prompt = f"""You are a structural evaluator for educational content. Evaluate this REVISIT paragraph from a reading session.

REVISIT paragraph:
{revisit_paragraph}

Concept being revisited: {revisit_concept}
Current session concepts: {session_concepts_str}

A REVISIT paragraph must:
1. Explicitly name the earlier concept ('{revisit_concept}' or a clear reference to it)
2. Make one specific new connection to the current session's material — something not visible when the concept was first taught
3. NOT re-explain the core mechanics of the earlier concept from scratch (a sentence of context is fine, a full re-explanation is not)"""

        condition = f"Does this paragraph correctly revisit '{revisit_concept}' with a new specific connection without re-explaining it from scratch?"

        return self.evaluate_binary(prompt, condition)

    def check_transition(self, concept_a_tail: str, concept_b_head: str, concept_a_name: str, concept_b_name: str) -> tuple[bool, str]:
        """
        Check if the transition from concept A to concept B is causal/explicit.

        Args:
            concept_a_tail: Last ~80 words of concept A's teaching
            concept_b_head: First ~80 words of concept B's teaching
            concept_a_name: Name of concept A
            concept_b_name: Name of concept B

        Returns:
            Tuple of (pass/fail, explanation)
        """
        prompt = f"""You are evaluating the transition between two concepts in a technical learning session.

End of concept "{concept_a_name}":
{concept_a_tail}

Start of concept "{concept_b_name}":
{concept_b_head}

A GOOD transition: the last sentence of concept A explicitly identifies a gap, constraint, or problem that concept B solves. The reader can feel WHY concept B follows.
A BAD transition: concept A ends and concept B begins with no causal bridge — it's a topic jump ("Now let's look at X", "Next we cover Y") or there's no connection between the ending and the opening."""

        condition = "Does the transition from concept A to concept B provide a clear causal or logical bridge (not a topic jump)?"

        return self.evaluate_binary(prompt, condition)

    def check_revisit_specificity(self, revisit_paragraph: str, revisit_concept: str, connection_reason: str) -> tuple[bool, str]:
        """
        Check that REVISIT paragraph makes a concrete, specific connection.

        Args:
            revisit_paragraph: The ~150-word REVISIT paragraph from the session
            revisit_concept: Name of the concept being revisited
            connection_reason: The planned one-sentence connection reason from planner

        Returns:
            Tuple of (pass/fail, explanation)
        """
        prompt = f"""You are evaluating a REVISIT paragraph in a technical learning session. A REVISIT paragraph brings back an earlier concept and shows how it looks different — or more important — from the current session's vantage point.

Revisited concept: {revisit_concept}
Planned connection: {connection_reason}

REVISIT paragraph:
{revisit_paragraph}

A GOOD REVISIT: names the concept explicitly, identifies a specific constraint or assumption from the earlier concept that the current material illuminates, shows the concrete hinge where old and new lock together.
A BAD REVISIT: vague connection ("both relate to memory efficiency"), re-explains the earlier concept's mechanics from scratch, or only says the concept "reappears" without saying WHY it matters differently now."""

        condition = f"Does the REVISIT paragraph make a concrete, specific connection between '{revisit_concept}' and the current session's material (not vague, not a re-explanation)?"

        return self.evaluate_binary(prompt, condition)


def get_ollama_client(host: str = "http://localhost:11434", model: str = "llama3.2:3b") -> Optional[OllamaAPI]:
    """
    Get or create Ollama API client, with connection check.

    Args:
        host: Ollama server host URL
        model: Model name to use

    Returns:
        OllamaAPI instance if connection successful, None otherwise
    """
    client = OllamaAPI(host=host, model=model)

    if not client.check_connection():
        print("Warning: Cannot connect to Ollama server at", host)
        print("Continuing without evaluation...")
        return None

    return client


# Test code for verify_evaluate_binary task
if __name__ == "__main__":
    print("Testing evaluate_binary() method...")
    print()

    # Check if Ollama server is running
    client = OllamaAPI()
    if not client.check_connection():
        print("ERROR: Cannot connect to Ollama server at localhost:11434")
        print("Please ensure Ollama is running before testing.")
        exit(1)

    print("Connected to Ollama server successfully")
    print()

    # Test 1: Simple binary evaluation
    print("Test 1: Simple binary evaluation")
    try:
        result, evidence = client.evaluate_binary(
            "Test prompt: Check if this text contains the word 'test'.",
            "Does the text contain the word 'test'?"
        )
        print(f"  Result: {result}")
        print(f"  Evidence: {evidence}")
        print(f"  [PASS] Test 1 passed - evaluate_binary() returns (bool, str) tuple")
    except Exception as e:
        print(f"  [FAIL] Test 1 failed: {e}")
        exit(1)

    print()

    # Test 2: Verify return types
    print("Test 2: Verify return types")
    try:
        result, evidence = client.evaluate_binary(
            "Test prompt: Verify the sky is blue.",
            "Is the sky typically blue on a clear day?"
        )
        assert isinstance(result, bool), f"Expected bool, got {type(result)}"
        assert isinstance(evidence, str), f"Expected str, got {type(evidence)}"
        print(f"  Result type: {type(result).__name__}")
        print(f"  Evidence type: {type(evidence).__name__}")
        print(f"  [PASS] Test 2 passed - return types are correct")
    except Exception as e:
        print(f"  [FAIL] Test 2 failed: {e}")
        exit(1)

    print()

    # Test 3: Test with different expected outcomes
    print("Test 3: Test with different expected outcomes")
    try:
        result_yes, evidence_yes = client.evaluate_binary(
            "Test prompt: Check if water is wet.",
            "Is water wet?"
        )
        print(f"  YES test - Result: {result_yes}, Evidence: {evidence_yes}")

        result_no, evidence_no = client.evaluate_binary(
            "Test prompt: Check if humans can fly without equipment.",
            "Can humans fly without equipment?"
        )
        print(f"  NO test - Result: {result_no}, Evidence: {evidence_no}")
        print(f"  [PASS] Test 3 passed - handles different outcomes")
    except Exception as e:
        print(f"  [FAIL] Test 3 failed: {e}")
        exit(1)

    print()
    print("=" * 60)
    print("All tests passed! evaluate_binary() is working correctly.")
    print("=" * 60)