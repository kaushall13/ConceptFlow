"""
Tests for api/ollama.py - OllamaAPI class
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.ollama import OllamaAPI, get_ollama_client


def _make_api(host="http://localhost:11434", model="llama3.2:3b"):
    return OllamaAPI(host=host, model=model)


def _make_mock_response(text="YES This is well-formed", status_code=200):
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {"response": text}
    resp.raise_for_status = MagicMock()
    return resp


class TestOllamaAPIInit:
    def test_strips_trailing_slash_from_host(self):
        api = _make_api(host="http://localhost:11434/")
        assert api.host == "http://localhost:11434"

    def test_sets_api_url(self):
        api = _make_api()
        assert api.api_url == "http://localhost:11434/api/generate"

    def test_custom_model_stored(self):
        api = _make_api(model="mistral:7b")
        assert api.model == "mistral:7b"


class TestCheckConnection:
    def test_returns_true_on_200(self):
        api = _make_api()
        with patch("api.ollama.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=200)
            assert api.check_connection() is True

    def test_returns_false_on_non_200(self):
        api = _make_api()
        with patch("api.ollama.requests.get") as mock_get:
            mock_get.return_value = MagicMock(status_code=404)
            assert api.check_connection() is False

    def test_returns_false_on_connection_error(self):
        api = _make_api()
        with patch("api.ollama.requests.get", side_effect=requests.RequestException("down")):
            assert api.check_connection() is False


class TestParseBinaryResponse:
    def setup_method(self):
        self.api = _make_api()

    def test_yes_at_start_returns_true(self):
        passed, evidence = self.api._parse_binary_response("YES The condition is met.")
        assert passed is True

    def test_no_at_start_returns_false(self):
        passed, evidence = self.api._parse_binary_response("NO The condition failed.")
        assert passed is False

    def test_yes_extracts_evidence(self):
        passed, evidence = self.api._parse_binary_response("YES Because it works.")
        assert "Because it works." in evidence

    def test_no_extracts_evidence(self):
        passed, evidence = self.api._parse_binary_response("NO It lacks specificity.")
        assert "It lacks specificity." in evidence

    def test_yes_no_evidence_returns_default(self):
        passed, evidence = self.api._parse_binary_response("YES")
        assert passed is True
        assert evidence == "Condition met"

    def test_no_no_evidence_returns_default(self):
        passed, evidence = self.api._parse_binary_response("NO")
        assert passed is False
        assert evidence == "Condition not met"

    def test_yes_buried_in_text(self):
        passed, _ = self.api._parse_binary_response("After some thinking: YES it meets the criteria")
        assert passed is True

    def test_no_buried_in_text(self):
        passed, _ = self.api._parse_binary_response("My answer: NO this does not meet criteria")
        assert passed is False

    def test_unclear_response_assumes_fail(self):
        passed, evidence = self.api._parse_binary_response("Maybe, it depends on the context.")
        assert passed is False
        assert "Unclear" in evidence

    def test_case_insensitive_yes(self):
        passed, _ = self.api._parse_binary_response("yes it passes")
        assert passed is True

    def test_case_insensitive_no(self):
        passed, _ = self.api._parse_binary_response("no it fails")
        assert passed is False


class TestEvaluateBinary:
    def setup_method(self):
        self.api = _make_api()

    def test_happy_path_returns_tuple(self):
        with patch("api.ollama.requests.post", return_value=_make_mock_response("YES Condition met")):
            result = self.api.evaluate_binary("prompt text", "check condition")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], str)

    def test_yes_response_returns_true(self):
        with patch("api.ollama.requests.post", return_value=_make_mock_response("YES Definitely passes")):
            passed, _ = self.api.evaluate_binary("p", "c")
        assert passed is True

    def test_no_response_returns_false(self):
        with patch("api.ollama.requests.post", return_value=_make_mock_response("NO It fails")):
            passed, _ = self.api.evaluate_binary("p", "c")
        assert passed is False

    def test_request_includes_condition(self):
        with patch("api.ollama.requests.post", return_value=_make_mock_response("YES ok")) as mock_post:
            self.api.evaluate_binary("prompt", "UNIQUE_CONDITION_STRING_XYZ")
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or (call_args.args[1] if len(call_args.args) > 1 else {})
        assert "UNIQUE_CONDITION_STRING_XYZ" in payload["prompt"]

    def test_request_includes_model(self):
        with patch("api.ollama.requests.post", return_value=_make_mock_response("YES ok")) as mock_post:
            self.api.evaluate_binary("prompt", "cond")
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or (call_args.args[1] if len(call_args.args) > 1 else {})
        assert payload["model"] == self.api.model

    def test_retries_on_request_exception(self):
        responses = [
            requests.RequestException("connection failed"),
            _make_mock_response("YES ok"),
        ]
        with patch("api.ollama.requests.post", side_effect=responses), \
             patch("api.ollama.time.sleep"):
            passed, _ = self.api.evaluate_binary("p", "c")
        assert passed is True

    def test_raises_after_max_retries(self):
        with patch("api.ollama.requests.post", side_effect=requests.RequestException("down")), \
             patch("api.ollama.time.sleep"):
            with pytest.raises(requests.RequestException):
                self.api.evaluate_binary("p", "c")

    def test_uses_low_temperature(self):
        with patch("api.ollama.requests.post", return_value=_make_mock_response("YES ok")) as mock_post:
            self.api.evaluate_binary("prompt", "cond")
        call_args = mock_post.call_args
        payload = call_args.kwargs.get("json") or (call_args.args[1] if len(call_args.args) > 1 else {})
        assert payload["options"]["temperature"] == 0.1


class TestCheckTension:
    def setup_method(self):
        self.api = _make_api()

    def test_delegates_to_evaluate_binary(self):
        with patch.object(self.api, "evaluate_binary", return_value=(True, "ok")) as mock_eval:
            result = self.api.check_tension("ending text", "next concept")
        mock_eval.assert_called_once()
        assert result == (True, "ok")

    def test_next_concept_in_prompt(self):
        with patch.object(self.api, "evaluate_binary", return_value=(True, "ok")) as mock_eval:
            self.api.check_tension("session ending", "UNIQUE_NEXT_CONCEPT_XYZ")
        call_args = mock_eval.call_args
        prompt = call_args[0][0]
        assert "UNIQUE_NEXT_CONCEPT_XYZ" in prompt

    def test_session_ending_in_prompt(self):
        with patch.object(self.api, "evaluate_binary", return_value=(True, "ok")) as mock_eval:
            self.api.check_tension("UNIQUE_SESSION_ENDING_XYZ", "next")
        call_args = mock_eval.call_args
        prompt = call_args[0][0]
        assert "UNIQUE_SESSION_ENDING_XYZ" in prompt


class TestCheckAnchor:
    def setup_method(self):
        self.api = _make_api()

    def test_delegates_to_evaluate_binary(self):
        with patch.object(self.api, "evaluate_binary", return_value=(True, "ok")) as mock_eval:
            result = self.api.check_anchor("opening", "prev ending", ["C1", "C2"])
        mock_eval.assert_called_once()
        assert result == (True, "ok")

    def test_concepts_joined_in_prompt(self):
        with patch.object(self.api, "evaluate_binary", return_value=(True, "ok")) as mock_eval:
            self.api.check_anchor("opening", "prev", ["ConceptAlpha", "ConceptBeta"])
        call_args = mock_eval.call_args
        prompt = call_args[0][0]
        assert "ConceptAlpha" in prompt
        assert "ConceptBeta" in prompt

    def test_empty_concepts_list(self):
        with patch.object(self.api, "evaluate_binary", return_value=(False, "fail")):
            passed, evidence = self.api.check_anchor("opening", "prev", [])
        assert passed is False


class TestCheckCoherence:
    def setup_method(self):
        self.api = _make_api()

    def test_delegates_to_evaluate_binary(self):
        with patch.object(self.api, "evaluate_binary", return_value=(True, "ok")) as mock_eval:
            self.api.check_coherence("full text", ["C1", "C2"], "RevisitC")
        mock_eval.assert_called_once()

    def test_none_revisit_handled(self):
        with patch.object(self.api, "evaluate_binary", return_value=(True, "ok")):
            passed, _ = self.api.check_coherence("text", ["C1"], None)
        assert passed is True

    def test_revisit_concept_in_prompt(self):
        with patch.object(self.api, "evaluate_binary", return_value=(True, "ok")) as mock_eval:
            self.api.check_coherence("text", ["C1"], "UNIQUE_REVISIT_NAME_XYZ")
        call_args = mock_eval.call_args
        prompt = call_args[0][0]
        assert "UNIQUE_REVISIT_NAME_XYZ" in prompt


class TestGetOllamaClient:
    def test_returns_client_when_connected(self):
        with patch.object(OllamaAPI, "check_connection", return_value=True):
            client = get_ollama_client("http://localhost:11434", "llama3.2:3b")
        assert client is not None

    def test_returns_none_when_not_connected(self, capsys):
        with patch.object(OllamaAPI, "check_connection", return_value=False):
            client = get_ollama_client("http://localhost:11434", "llama3.2:3b")
        assert client is None
        captured = capsys.readouterr()
        assert "Warning" in captured.out or "Cannot connect" in captured.out
