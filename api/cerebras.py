"""
Cerebras API Wrapper - LLM calls with retry, token logging, rate limiting,
and dual-key rotation with daily quota tracking.
"""

import hashlib
import json
import os
import re
import tempfile
import time
from datetime import date
from pathlib import Path
from typing import Dict, Any, List, Optional

import openai

from .rate_limiter import RateLimiter


# --- Custom exception -----------------------------------------------------------

class DailyQuotaExhausted(Exception):
    """Raised when all configured Cerebras API keys have exhausted their daily token quota."""
    pass


def _salvage_truncated_json(text: str):
    """
    Attempt to recover a valid JSON object from a truncated response.
    Closes any open arrays/objects to make it parseable.
    Returns parsed dict/list or None if unrecoverable.
    """
    text = text.strip()
    if not text.startswith('{'):
        return None
    # Count open/close braces and brackets to determine what needs closing
    depth_brace = 0
    depth_bracket = 0
    in_string = False
    escape_next = False
    last_valid_pos = 0
    for i, ch in enumerate(text):
        if escape_next:
            escape_next = False
            continue
        if ch == '\\' and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
        if not in_string:
            if ch == '{':
                depth_brace += 1
            elif ch == '}':
                depth_brace -= 1
            elif ch == '[':
                depth_bracket += 1
            elif ch == ']':
                depth_bracket -= 1
            if depth_brace > 0 or depth_bracket > 0:
                last_valid_pos = i
    # Build closing suffix
    # If we're inside a string, close it
    if in_string:
        text += '"'
    # Close any open array elements or object values with null
    closing = ''
    if depth_bracket > 0:
        closing += ']' * depth_bracket
    if depth_brace > 0:
        closing += '}' * depth_brace
    try:
        return json.loads(text + closing)
    except json.JSONDecodeError:
        # Last resort: truncate to last complete top-level field
        # Find last complete "}" at depth 1
        try:
            idx = text.rfind('},\n')
            if idx > 0:
                truncated = text[:idx+1] + ']}'  # close array + object
                return json.loads('{"concepts":' + text[text.find('['):idx+1] + ']}')
        except Exception:
            pass
        return None


# --- Quota tracker --------------------------------------------------------------

class KeyQuotaTracker:
    """
    Tracks per-key daily token usage in state/key_quota.json.
    Uses the first 8 characters of the key (hashed) as the identifier so the
    full key is never stored on disk.
    Writes are atomic (write-to-temp then os.replace).
    """

    QUOTA_FILE = Path("state") / "key_quota.json"

    def __init__(self, daily_limit: int):
        self.daily_limit = daily_limit
        self.QUOTA_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def key_id(self, api_key: str) -> str:
        """Return a short opaque identifier for a key (never the key itself)."""
        prefix = api_key[:8] if len(api_key) >= 8 else api_key
        return hashlib.sha256(prefix.encode()).hexdigest()[:16]

    def would_exceed(self, api_key: str, estimated_tokens: int) -> bool:
        """Return True if using estimated_tokens more would exceed the daily limit."""
        record = self._get_record(api_key)
        return (record["tokens_used"] + estimated_tokens) > self.daily_limit

    def record_usage(self, api_key: str, tokens_used: int) -> None:
        """Add tokens_used to the running daily total for api_key."""
        data = self._load()
        kid = self.key_id(api_key)
        today = str(date.today())

        record = data.get(kid, {"date": today, "tokens_used": 0})
        if record["date"] != today:
            record = {"date": today, "tokens_used": 0}

        record["tokens_used"] += tokens_used
        data[kid] = record
        self._save(data)

    def tokens_used_today(self, api_key: str) -> int:
        """Return tokens already used today for api_key."""
        return self._get_record(api_key)["tokens_used"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_record(self, api_key: str) -> Dict[str, Any]:
        data = self._load()
        kid = self.key_id(api_key)
        today = str(date.today())
        record = data.get(kid, {"date": today, "tokens_used": 0})
        # Reset if date changed
        if record["date"] != today:
            record = {"date": today, "tokens_used": 0}
        return record

    def _load(self) -> Dict[str, Any]:
        if not self.QUOTA_FILE.exists():
            return {}
        try:
            with open(self.QUOTA_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}

    def _save(self, data: Dict[str, Any]) -> None:
        state_dir = self.QUOTA_FILE.parent
        temp_fd, temp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
        try:
            with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            os.replace(temp_path, self.QUOTA_FILE)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise


# --- Main client ----------------------------------------------------------------

class CerebrasAPI:
    """
    Wrapper for Cerebras API with retry logic, token tracking,
    and dual-key daily quota rotation.
    """

    # Free tier approximate limits per model
    FREE_TIER_RPM = 30      # requests per minute
    FREE_TIER_TPM = 60000   # tokens per minute (conservative estimate)

    def __init__(self,
                 api_keys: List[str],
                 base_url: str = "https://api.cerebras.ai/v1",
                 model: str = "llama-3.3-70b",
                 daily_token_limit: int = 900000):
        # Filter out blank keys
        self._keys = [k for k in api_keys if k and k.strip()]
        if not self._keys:
            raise ValueError("At least one non-empty Cerebras API key must be provided.")

        self.model = model
        self.rate_limiter = RateLimiter(rpm=self.FREE_TIER_RPM, tpm=self.FREE_TIER_TPM)
        self.quota_tracker = KeyQuotaTracker(daily_limit=daily_token_limit)
        self._base_url = base_url

        # Start with the first key; _select_key() will rotate if needed
        self._current_key_index = 0
        self._client = self._make_client(self._keys[0])

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(self,
                 system_prompt: str,
                 user_prompt: str,
                 max_tokens: int = 4096,
                 temperature: float = 0.7,
                 enable_web_search: bool = False,
                 model_override: str = None) -> str:
        """
        Generate text using Cerebras API with exponential backoff retry
        and automatic key rotation when the daily quota is reached.

        Returns:
            Generated text response

        Raises:
            DailyQuotaExhausted: If all keys have exhausted their daily quota.
            Exception: If all retries exhausted for non-quota reasons.
        """
        max_retries = 4
        base_delay = 60.0  # TPM window is 60s — first retry waits the full window

        # Estimate tokens for rate limiting and quota check
        estimated_input_tokens = self.rate_limiter.estimate_tokens(system_prompt + user_prompt)
        estimated_total = estimated_input_tokens + max_tokens

        # Select the best key before starting
        self._select_key(estimated_total)

        for attempt in range(max_retries):
            try:
                # Wait for both RPM and TPM budget
                self.rate_limiter.acquire(estimated_total)

                slot = self._current_key_index + 1
                print(f"  [Cerebras Key {slot}] sending request (attempt {attempt+1}/{max_retries})")

                response = self._client.chat.completions.create(
                    model=model_override or self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                response_text = response.choices[0].message.content or ""

                # Log token usage and update quota + rate limiter with actuals
                usage = response.usage
                if usage:
                    actual_tokens = usage.prompt_tokens + usage.completion_tokens
                    print(
                        f"  [Cerebras Key {slot}] tokens: "
                        f"{usage.prompt_tokens} in, {usage.completion_tokens} out "
                        f"(daily used: {self.quota_tracker.tokens_used_today(self._keys[self._current_key_index]) + actual_tokens:,})"
                    )
                    self.rate_limiter.record_actual_tokens(actual_tokens)
                    self.quota_tracker.record_usage(self._keys[self._current_key_index], actual_tokens)

                return response_text

            except openai.RateLimitError as e:
                slot = self._current_key_index + 1
                # On second attempt, try rotating to the other key
                if attempt == 1 and len(self._keys) > 1:
                    next_index = (self._current_key_index + 1) % len(self._keys)
                    if next_index != self._current_key_index:
                        self._current_key_index = next_index
                        self._client = openai.OpenAI(
                            base_url="https://api.cerebras.ai/v1",
                            api_key=self._keys[self._current_key_index],
                        )
                        print(f"  [Cerebras] Rotated to Key {self._current_key_index + 1} after rate limit")
                wait = base_delay * (attempt + 1)  # 60s, 120s, 180s, 240s
                slot = self._current_key_index + 1
                print(f"  [Cerebras Key {slot}] Rate limit hit (attempt {attempt+1}/{max_retries}), waiting {wait:.1f}s: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    raise

            except openai.APIStatusError as e:
                slot = self._current_key_index + 1
                if e.status_code == 429:
                    wait = base_delay * (attempt + 1)  # 60s, 120s, 180s, 240s
                    print(f"  [Cerebras Key {slot}] 429 rate limit (attempt {attempt+1}/{max_retries}), waiting {wait:.1f}s")
                    if attempt < max_retries - 1:
                        time.sleep(wait)
                        continue
                elif e.status_code >= 500:
                    wait = base_delay * (2 ** attempt)
                    print(f"  [Cerebras Key {slot}] Server error {e.status_code} (attempt {attempt+1}/{max_retries}), waiting {wait:.1f}s")
                    if attempt < max_retries - 1:
                        time.sleep(wait)
                        continue
                else:
                    print(f"  [Cerebras Key {slot}] API error {e.status_code}: {e}")
                    raise
                if attempt == max_retries - 1:
                    raise

            except openai.APIConnectionError as e:
                slot = self._current_key_index + 1
                wait = base_delay * (2 ** attempt)
                print(f"  [Cerebras Key {slot}] Connection error (attempt {attempt+1}/{max_retries}), waiting {wait:.1f}s: {e}")
                if attempt < max_retries - 1:
                    time.sleep(wait)
                else:
                    raise

            except Exception as e:
                slot = self._current_key_index + 1
                print(f"  [Cerebras Key {slot}] Unexpected error (attempt {attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait = base_delay * (2 ** attempt)
                    time.sleep(wait)
                else:
                    raise

    def generate_json(self,
                      system_prompt: str,
                      user_prompt: str,
                      max_tokens: int = 4096,
                      model_override: str = None) -> Dict[str, Any]:
        """Generate and parse a JSON response."""
        json_instruction = "\n\nIMPORTANT: Respond with valid JSON only. No markdown code blocks, no explanation, no preamble."

        response_text = self.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt + json_instruction,
            max_tokens=max_tokens,
            temperature=0.3,
            model_override=model_override,
        )

        return self._parse_json(response_text)

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Parse JSON from response, stripping markdown code blocks if present."""
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
            # Handle truncated JSON: response was cut off mid-object.
            # Try to salvage by closing any open structures.
            salvaged = _salvage_truncated_json(text)
            if salvaged is not None:
                print("  [WARN] JSON was truncated — salvaged partial response")
                return salvaged
            raise Exception(f"Failed to parse JSON from response: {text[:300]}")

    def count_tokens(self, text: str) -> int:
        """Estimate token count (~4 chars per token for English)."""
        return len(text) // 4

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_client(self, api_key: str) -> openai.OpenAI:
        return openai.OpenAI(
            api_key=api_key,
            base_url=self._base_url,
            timeout=300.0,  # 5 min timeout for large context requests
        )

    def _select_key(self, estimated_tokens: int) -> None:
        """
        Rotate to the first key that has sufficient daily quota remaining.
        Raises DailyQuotaExhausted if no key can handle the request today.
        """
        num_keys = len(self._keys)
        # Try all keys starting from the current one
        for offset in range(num_keys):
            idx = (self._current_key_index + offset) % num_keys
            key = self._keys[idx]
            if not self.quota_tracker.would_exceed(key, estimated_tokens):
                if idx != self._current_key_index:
                    self._current_key_index = idx
                    self._client = self._make_client(key)
                    print(f"  [Cerebras] Rotated to Key {idx + 1} (daily quota on previous key reached)")
                return

        # All keys exhausted
        raise DailyQuotaExhausted(
            f"All {num_keys} Cerebras API key(s) have exhausted their daily token quota "
            f"({self.quota_tracker.daily_limit:,} tokens/day). Resume tomorrow."
        )


# --- Factory --------------------------------------------------------------------

def get_cerebras_client(config: Dict[str, Any]) -> CerebrasAPI:
    """Create a CerebrasAPI client from a config dict.

    Reads cerebras_api_key_1 and cerebras_api_key_2; falls back to the legacy
    cerebras_api_key field so existing configs continue to work during migration.
    """
    keys: List[str] = []

    # New dual-key fields
    key1 = config.get("cerebras_api_key_1", "")
    key2 = config.get("cerebras_api_key_2", "")
    if key1:
        keys.append(key1)
    if key2:
        keys.append(key2)

    # Legacy single-key fallback
    if not keys:
        legacy = config.get("cerebras_api_key", "")
        if legacy:
            keys.append(legacy)

    if not keys:
        raise ValueError(
            "No Cerebras API key found in config. "
            "Set cerebras_api_key_1 (and optionally cerebras_api_key_2) in config.json."
        )

    return CerebrasAPI(
        api_keys=keys,
        model=config.get("cerebras_model", "llama-3.3-70b"),
        daily_token_limit=config.get("cerebras_daily_token_limit", 900000),
    )
