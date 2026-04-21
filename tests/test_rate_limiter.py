"""
Tests for api/rate_limiter.py - RateLimiter class
"""

import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.rate_limiter import RateLimiter


class TestEstimateTokens:
    def test_basic_estimation(self):
        rl = RateLimiter()
        # 8 chars / 4 = 2 tokens
        assert rl.estimate_tokens("abcdefgh") == 2

    def test_empty_string_returns_one(self):
        rl = RateLimiter()
        assert rl.estimate_tokens("") == 1

    def test_single_char_returns_one(self):
        rl = RateLimiter()
        assert rl.estimate_tokens("x") == 1

    def test_longer_string(self):
        rl = RateLimiter()
        text = "a" * 400
        assert rl.estimate_tokens(text) == 100

    def test_unicode_text_does_not_crash(self):
        rl = RateLimiter()
        result = rl.estimate_tokens("café résumé naïve")
        assert result >= 1

    def test_returns_int(self):
        rl = RateLimiter()
        result = rl.estimate_tokens("hello world test")
        assert isinstance(result, int)


class TestAcquireBasic:
    def test_acquire_does_not_block_under_limits(self):
        rl = RateLimiter(rpm=30, tpm=60000)
        start = time.time()
        rl.acquire(100)
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_acquire_records_request(self):
        rl = RateLimiter(rpm=30, tpm=60000)
        rl.acquire(100)
        assert len(rl._request_times) == 1

    def test_acquire_records_token_log(self):
        rl = RateLimiter(rpm=30, tpm=60000)
        rl.acquire(250)
        assert len(rl._token_log) == 1
        assert rl._token_log[0][1] == 250

    def test_multiple_acquires_accumulate(self):
        rl = RateLimiter(rpm=30, tpm=60000)
        for _ in range(5):
            rl.acquire(100)
        assert len(rl._request_times) == 5
        assert len(rl._token_log) == 5

    def test_acquire_with_zero_tokens_does_not_crash(self):
        rl = RateLimiter(rpm=30, tpm=60000)
        rl.acquire(0)  # Should not raise

    def test_token_log_timestamp_is_recent(self):
        rl = RateLimiter(rpm=30, tpm=60000)
        before = time.time()
        rl.acquire(100)
        after = time.time()
        ts = rl._token_log[0][0]
        assert before <= ts <= after


class TestRecordActualTokens:
    def test_updates_most_recent_entry(self):
        rl = RateLimiter(rpm=30, tpm=60000)
        rl.acquire(100)
        rl.record_actual_tokens(350)
        assert rl._token_log[-1][1] == 350

    def test_does_not_crash_when_empty(self):
        rl = RateLimiter()
        rl.record_actual_tokens(500)  # Should not raise

    def test_only_updates_last_entry(self):
        rl = RateLimiter(rpm=30, tpm=60000)
        rl.acquire(100)
        rl.acquire(200)
        rl.record_actual_tokens(999)
        assert rl._token_log[0][1] == 100  # First entry unchanged
        assert rl._token_log[1][1] == 999  # Last entry updated

    def test_timestamp_is_preserved(self):
        rl = RateLimiter(rpm=30, tpm=60000)
        rl.acquire(100)
        original_ts = rl._token_log[0][0]
        rl.record_actual_tokens(500)
        assert rl._token_log[0][0] == original_ts  # Timestamp unchanged


class TestRPMWindowBehavior:
    def test_expired_requests_are_purged(self):
        rl = RateLimiter(rpm=5, tpm=999999)
        # Inject an old timestamp (> 60 s ago)
        rl._request_times.append(time.time() - 61)
        rl.acquire(10)
        # Old entry purged; only new one remains
        assert len(rl._request_times) == 1

    def test_within_window_requests_are_counted(self):
        rl = RateLimiter(rpm=10, tpm=999999)
        now = time.time()
        # Inject 3 recent requests (within window)
        rl._request_times.append(now - 10)
        rl._request_times.append(now - 5)
        rl._request_times.append(now - 1)
        rl.acquire(50)
        # 3 old + 1 new = 4
        assert len(rl._request_times) == 4

    def test_custom_rpm_is_respected_in_config(self):
        rl = RateLimiter(rpm=7, tpm=999999)
        assert rl.rpm == 7


class TestTPMWindowBehavior:
    def test_tpm_allows_request_when_under_budget(self):
        rl = RateLimiter(rpm=999, tpm=1000)
        start = time.time()
        rl.acquire(500)
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_expired_tokens_are_purged(self):
        rl = RateLimiter(rpm=999, tpm=500)
        old_time = time.time() - 61
        rl._token_log.append((old_time, 400))
        # Old 400 tokens expired; acquiring 400 should not block
        start = time.time()
        rl.acquire(400)
        elapsed = time.time() - start
        assert elapsed < 1.0

    def test_current_tpm_sum_reflects_log(self):
        rl = RateLimiter(rpm=999, tpm=9999)
        rl.acquire(300)
        rl.acquire(400)
        total = sum(t for _, t in rl._token_log)
        assert total == 700

    def test_custom_tpm_is_respected_in_config(self):
        rl = RateLimiter(rpm=30, tpm=14400)
        assert rl.tpm == 14400


class TestThreadSafety:
    def test_concurrent_acquires_do_not_raise(self):
        rl = RateLimiter(rpm=100, tpm=999999)
        errors = []

        def worker():
            try:
                rl.acquire(10)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

    def test_concurrent_record_actual_tokens_does_not_raise(self):
        rl = RateLimiter(rpm=100, tpm=999999)
        rl.acquire(100)
        errors = []

        def worker():
            try:
                rl.record_actual_tokens(50)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
