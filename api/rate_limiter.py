"""
Rate limiter supporting both RPM (requests per minute) and TPM (tokens per minute).
Used for Cerebras and Groq free tier compliance.
"""

import time
from collections import deque
from threading import Lock


class RateLimiter:
    """
    Sliding window rate limiter tracking both requests/min and tokens/min.
    Thread-safe for parallel pipeline stages.
    """

    def __init__(self, rpm: int = 30, tpm: int = 60000):
        self.rpm = rpm
        self.tpm = tpm
        self._lock = Lock()
        self._request_times: deque = deque()
        self._token_log: deque = deque()  # (timestamp, token_count) pairs

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 characters per token."""
        return max(1, len(text) // 4)

    def acquire(self, estimated_tokens: int = 100) -> None:
        """
        Block until both RPM and TPM budgets allow the request.
        Waits as long as needed, then records the request.
        """
        with self._lock:
            self._wait_for_rpm()
            self._wait_for_tpm(estimated_tokens)
            now = time.time()
            self._request_times.append(now)
            self._token_log.append((now, estimated_tokens))

    def record_actual_tokens(self, actual_tokens: int) -> None:
        """
        Update the most recent token log entry with actual usage.
        Call this after receiving the API response.
        """
        with self._lock:
            if self._token_log:
                ts, _ = self._token_log[-1]
                self._token_log[-1] = (ts, actual_tokens)

    def _wait_for_rpm(self) -> None:
        """Wait until the RPM window has space. Must hold lock."""
        while True:
            now = time.time()
            cutoff = now - 60.0
            while self._request_times and self._request_times[0] < cutoff:
                self._request_times.popleft()

            if len(self._request_times) < self.rpm:
                return

            oldest = self._request_times[0]
            wait_time = (oldest + 60.0) - now + 0.1
            if wait_time > 0:
                print(f"  [RateLimit] RPM limit ({self.rpm}/min) reached, waiting {wait_time:.1f}s...")
                self._lock.release()
                time.sleep(wait_time)
                self._lock.acquire()

    def _wait_for_tpm(self, tokens: int) -> None:
        """Wait until the TPM window has space. Must hold lock."""
        while True:
            now = time.time()
            cutoff = now - 60.0
            while self._token_log and self._token_log[0][0] < cutoff:
                self._token_log.popleft()

            current_tpm = sum(t for _, t in self._token_log)

            if current_tpm + tokens <= self.tpm:
                return

            # Wait for oldest token batch to expire
            if self._token_log:
                oldest_ts = self._token_log[0][0]
                wait_time = (oldest_ts + 60.0) - now + 0.1
                if wait_time > 0:
                    print(f"  [RateLimit] TPM limit ({self.tpm}/min) reached (used: {current_tpm}), waiting {wait_time:.1f}s...")
                    self._lock.release()
                    time.sleep(wait_time)
                    self._lock.acquire()
                    continue

            return
