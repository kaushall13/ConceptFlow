"""
Tests for api/cerebras.py - CerebrasAPI class
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

import openai
from api.cerebras import CerebrasAPI, get_cerebras_client


def _make_mock_response(content="Hello world", prompt_tokens=100, completion_tokens=50):
    """Build a mock openai chat completion response."""
    response = MagicMock()
    response.choices[0].message.content = content
    response.usage.prompt_tokens = prompt_tokens
    response.usage.completion_tokens = completion_tokens
    return response


class TestCerebrasAPIInit:
    def test_creates_client_with_api_key(self):
        with patch("api.cerebras.openai.OpenAI") as mock_openai:
            api = CerebrasAPI(api_key="test-key-123")
            mock_openai.assert_called_once()
            call_kwargs = mock_openai.call_args[1]
            assert call_kwargs["api_key"] == "test-key-123"

    def test_default_model(self):
        with patch("api.cerebras.openai.OpenAI"):
            api = CerebrasAPI(api_key="key")
            assert api.model == "llama-3.3-70b"

    def test_custom_model(self):
        with patch("api.cerebras.openai.OpenAI"):
            api = CerebrasAPI(api_key="key", model="gpt-oss-120b")
            assert api.model == "gpt-oss-120b"

    def test_rate_limiter_created(self):
        with patch("api.cerebras.openai.OpenAI"):
            api = CerebrasAPI(api_key="key")
            assert api.rate_limiter is not None
            assert api.rate_limiter.rpm == CerebrasAPI.FREE_TIER_RPM


class TestGenerate:
    def setup_method(self):
        with patch("api.cerebras.openai.OpenAI"):
            self.api = CerebrasAPI(api_key="test-key")
        self.api.rate_limiter = MagicMock()

    def test_happy_path_returns_text(self):
        self.api.client.chat.completions.create.return_value = _make_mock_response("Great answer")
        result = self.api.generate("sys", "user")
        assert result == "Great answer"

    def test_passes_system_and_user_to_messages(self):
        self.api.client.chat.completions.create.return_value = _make_mock_response()
        self.api.generate("SYS_PROMPT", "USER_PROMPT")
        call_kwargs = self.api.client.chat.completions.create.call_args[1]
        messages = call_kwargs["messages"]
        assert messages[0] == {"role": "system", "content": "SYS_PROMPT"}
        assert messages[1] == {"role": "user", "content": "USER_PROMPT"}

    def test_logs_token_usage(self, capsys):
        self.api.client.chat.completions.create.return_value = _make_mock_response(
            "response", prompt_tokens=200, completion_tokens=80
        )
        self.api.generate("sys", "user")
        captured = capsys.readouterr()
        assert "200" in captured.out
        assert "80" in captured.out

    def test_records_actual_tokens_after_call(self):
        self.api.client.chat.completions.create.return_value = _make_mock_response(
            prompt_tokens=100, completion_tokens=50
        )
        self.api.generate("sys", "user")
        self.api.rate_limiter.record_actual_tokens.assert_called_once_with(150)

    def test_rate_limit_error_retries_and_succeeds(self):
        self.api.client.chat.completions.create.side_effect = [
            openai.RateLimitError("rate limit", response=MagicMock(), body={}),
            _make_mock_response("Retry success"),
        ]
        with patch("api.cerebras.time.sleep"):
            result = self.api.generate("sys", "user")
        assert result == "Retry success"
        assert self.api.client.chat.completions.create.call_count == 2

    def test_rate_limit_error_exhausts_retries_and_raises(self):
        self.api.client.chat.completions.create.side_effect = openai.RateLimitError(
            "rate limit", response=MagicMock(), body={}
        )
        with patch("api.cerebras.time.sleep"):
            with pytest.raises(openai.RateLimitError):
                self.api.generate("sys", "user")
        assert self.api.client.chat.completions.create.call_count == 3  # max_retries

    def test_server_error_500_retries(self):
        err = openai.APIStatusError("500 error", response=MagicMock(status_code=500), body={})
        err.status_code = 500
        self.api.client.chat.completions.create.side_effect = [
            err,
            _make_mock_response("Server recovered"),
        ]
        with patch("api.cerebras.time.sleep"):
            result = self.api.generate("sys", "user")
        assert result == "Server recovered"

    def test_client_error_400_does_not_retry(self):
        err = openai.APIStatusError("400 bad request", response=MagicMock(status_code=400), body={})
        err.status_code = 400
        self.api.client.chat.completions.create.side_effect = err
        with pytest.raises(openai.APIStatusError):
            self.api.generate("sys", "user")
        # Only one attempt — no retries on 4xx non-rate-limit
        assert self.api.client.chat.completions.create.call_count == 1

    def test_connection_error_retries(self):
        self.api.client.chat.completions.create.side_effect = [
            openai.APIConnectionError(request=MagicMock()),
            _make_mock_response("Reconnected"),
        ]
        with patch("api.cerebras.time.sleep"):
            result = self.api.generate("sys", "user")
        assert result == "Reconnected"

    def test_empty_content_returns_empty_string(self):
        resp = _make_mock_response()
        resp.choices[0].message.content = None
        self.api.client.chat.completions.create.return_value = resp
        result = self.api.generate("sys", "user")
        assert result == ""


class TestGenerateJson:
    def setup_method(self):
        with patch("api.cerebras.openai.OpenAI"):
            self.api = CerebrasAPI(api_key="test-key")
        self.api.rate_limiter = MagicMock()

    def test_returns_parsed_dict(self):
        self.api.client.chat.completions.create.return_value = _make_mock_response('{"key": "value"}')
        result = self.api.generate_json("sys", "user")
        assert result == {"key": "value"}

    def test_appends_json_instruction_to_user_prompt(self):
        self.api.client.chat.completions.create.return_value = _make_mock_response('{"a": 1}')
        self.api.generate_json("sys", "user prompt")
        call_kwargs = self.api.client.chat.completions.create.call_args[1]
        user_content = call_kwargs["messages"][1]["content"]
        assert "JSON" in user_content
        assert "user prompt" in user_content

    def test_uses_low_temperature(self):
        self.api.client.chat.completions.create.return_value = _make_mock_response('{"x": 1}')
        self.api.generate_json("sys", "user")
        call_kwargs = self.api.client.chat.completions.create.call_args[1]
        assert call_kwargs["temperature"] == 0.3


class TestParseJson:
    def setup_method(self):
        with patch("api.cerebras.openai.OpenAI"):
            self.api = CerebrasAPI(api_key="key")

    def test_plain_json_object(self):
        result = self.api._parse_json('{"concepts": []}')
        assert result == {"concepts": []}

    def test_plain_json_array(self):
        result = self.api._parse_json('[1, 2, 3]')
        assert result == [1, 2, 3]

    def test_strips_markdown_json_fence(self):
        text = "```json\n{\"key\": \"val\"}\n```"
        result = self.api._parse_json(text)
        assert result == {"key": "val"}

    def test_strips_plain_backtick_fence(self):
        text = "```\n{\"a\": 1}\n```"
        result = self.api._parse_json(text)
        assert result == {"a": 1}

    def test_extracts_json_from_prose(self):
        text = 'Here is the answer: {"found": true} and some text after.'
        result = self.api._parse_json(text)
        assert result == {"found": True}

    def test_raises_on_unparseable_text(self):
        with pytest.raises(Exception, match="Failed to parse JSON"):
            self.api._parse_json("This is not JSON at all.")

    def test_handles_nested_json(self):
        obj = {"outer": {"inner": [1, 2, {"deep": True}]}}
        result = self.api._parse_json(json.dumps(obj))
        assert result == obj

    def test_strips_surrounding_whitespace(self):
        result = self.api._parse_json('   {"spaced": true}   ')
        assert result == {"spaced": True}


class TestCountTokens:
    def setup_method(self):
        with patch("api.cerebras.openai.OpenAI"):
            self.api = CerebrasAPI(api_key="key")

    def test_token_count_estimation(self):
        assert self.api.count_tokens("abcdefgh") == 2  # 8 / 4

    def test_empty_string_returns_zero(self):
        assert self.api.count_tokens("") == 0


class TestGetCerebrasClient:
    def test_creates_client_from_config(self):
        config = {"cerebras_api_key": "mykey", "cerebras_model": "custom-model"}
        with patch("api.cerebras.openai.OpenAI"):
            client = get_cerebras_client(config)
        assert client.model == "custom-model"

    def test_uses_default_model_when_missing(self):
        config = {"cerebras_api_key": "mykey"}
        with patch("api.cerebras.openai.OpenAI"):
            client = get_cerebras_client(config)
        assert client.model == "llama-3.3-70b"
