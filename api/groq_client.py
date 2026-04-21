"""
Groq API Wrapper - Fast LLM inference with free tier management.
Used as a supplementary provider for smaller tasks.
"""

import json
import re
import time
from typing import Dict, Any, Optional

from .rate_limiter import RateLimiter


class GroqAPI:
    """
    Wrapper for Groq API.
    Groq free tier: ~30 req/min, model-dependent TPM.
    Uses openai-compatible interface via groq SDK.
    """

    # Groq free tier limits (conservative)
    FREE_TIER_RPM = 30
    FREE_TIER_TPM = 14400  # tokens/min for llama-3.3-70b-versatile on free tier

    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        try:
            from groq import Groq
        except ImportError:
            raise ImportError("groq package not installed. Run: pip install groq")

        self.client = Groq(api_key=api_key, timeout=120.0)
        self.model = model
        self.rate_limiter = RateLimiter(rpm=self.FREE_TIER_RPM, tpm=self.FREE_TIER_TPM)

    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 max_tokens: int = 4096,
                 temperature: float = 0.7) -> str:
        """
        Generate text using Groq API with exponential backoff retry.

        Returns:
            Generated text response
        """
        max_retries = 3
        base_delay = 2.0

        estimated_tokens = self.rate_limiter.estimate_tokens(system_prompt + user_prompt)
        estimated_total = estimated_tokens + max_tokens

        for attempt in range(max_retries):
            try:
                self.rate_limiter.acquire(estimated_total)

                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                response_text = response.choices[0].message.content or ""

                usage = response.usage
                if usage:
                    print(f"  [Groq] tokens: {usage.prompt_tokens} in, {usage.completion_tokens} out")
                    self.rate_limiter.record_actual_tokens(usage.prompt_tokens + usage.completion_tokens)

                return response_text

            except Exception as e:
                error_str = str(e).lower()
                is_rate_limit = '429' in str(e) or 'rate limit' in error_str or 'rate_limit' in error_str

                if is_rate_limit:
                    wait = 60.0  # Wait a full minute on rate limit
                    print(f"  [Groq] Rate limit hit (attempt {attempt+1}/{max_retries}), waiting {wait:.0f}s")
                else:
                    wait = base_delay * (2 ** attempt)
                    print(f"  [Groq] Error (attempt {attempt+1}/{max_retries}), waiting {wait:.1f}s: {e}")

                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    raise

    def generate_json(self,
                      system_prompt: str,
                      user_prompt: str,
                      max_tokens: int = 4096) -> Dict[str, Any]:
        """Generate and parse a JSON response."""
        json_instruction = "\n\nIMPORTANT: Respond with valid JSON only. No markdown code blocks, no explanation."

        response_text = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt + json_instruction,
            max_tokens=max_tokens,
            temperature=0.3,
        )

        return self._parse_json(response_text)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        text = text.strip()
        text = re.sub(r'^```(?:json)?\s*', '', text)
        text = re.sub(r'\s*```$', '', text)
        text = text.strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', text)
            if match:
                try:
                    return json.loads(match.group(1))
                except json.JSONDecodeError:
                    pass
            raise Exception(f"Failed to parse JSON from Groq response: {text[:300]}")


def get_groq_client(config: Dict[str, Any]) -> Optional['GroqAPI']:
    """Create a Groq client if API key is configured, else return None."""
    api_key = config.get('groq_api_key', '')
    if not api_key:
        return None
    try:
        return GroqAPI(
            api_key=api_key,
            model=config.get('groq_model', 'llama-3.3-70b-versatile'),
        )
    except Exception as e:
        print(f"Warning: Could not initialize Groq client: {e}")
        return None
