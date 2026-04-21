"""
Tests for main.py - Flask API endpoints
"""

import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def app_dirs(tmp_path):
    """Create temporary state and output directories and patch the module-level constants."""
    state_dir = tmp_path / "state"
    output_dir = tmp_path / "output"
    state_dir.mkdir()
    output_dir.mkdir()

    with patch("main.STATE_DIR", str(state_dir)), \
         patch("main.OUTPUT_DIR", str(output_dir)), \
         patch("main.CONFIG_FILE", str(tmp_path / "config.json")):
        yield {
            "state_dir": state_dir,
            "output_dir": output_dir,
            "config_file": tmp_path / "config.json",
            "tmp_path": tmp_path,
        }


@pytest.fixture
def client(app_dirs):
    """Flask test client with patched filesystem directories."""
    from main import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c, app_dirs


class TestGetState:
    def test_returns_default_when_no_file(self, client):
        c, dirs = client
        resp = c.get("/api/state")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["current_stage"] == "Not started"

    def test_returns_state_file_content(self, client):
        c, dirs = client
        state = {
            "current_stage": "Ingest",
            "outputs": {"Ingest": "text"},
            "session_results": {}
        }
        state_file = dirs["state_dir"] / "pipeline_state.json"
        state_file.write_text(json.dumps(state))
        resp = c.get("/api/state")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["current_stage"] == "Ingest"

    def test_response_has_required_keys(self, client):
        c, dirs = client
        resp = c.get("/api/state")
        data = json.loads(resp.data)
        assert "current_stage" in data


class TestGetIndex:
    def test_returns_404_when_no_state_file(self, client):
        c, dirs = client
        resp = c.get("/api/index")
        assert resp.status_code == 404

    def test_returns_404_when_index_not_generated(self, client):
        c, dirs = client
        state = {"current_stage": "Ingest", "outputs": {}, "session_results": {}}
        (dirs["state_dir"] / "pipeline_state.json").write_text(json.dumps(state))
        resp = c.get("/api/index")
        assert resp.status_code == 404

    def test_returns_index_when_available(self, client):
        c, dirs = client
        index_data = [
            {"session_number": "01", "title": "T1", "estimated_minutes": 15, "concepts": []}
        ]
        state = {
            "current_stage": "IndexGen",
            "outputs": {"IndexGen": {"index": index_data, "approved": False}},
            "session_results": {}
        }
        (dirs["state_dir"] / "pipeline_state.json").write_text(json.dumps(state))
        resp = c.get("/api/index")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert "index" in data
        assert len(data["index"]) == 1

    def test_returns_approved_status(self, client):
        c, dirs = client
        state = {
            "current_stage": "IndexGen",
            "outputs": {"IndexGen": {
                "index": [{"session_number": "01", "title": "T1", "estimated_minutes": 15, "concepts": []}],
                "approved": True
            }},
            "session_results": {}
        }
        (dirs["state_dir"] / "pipeline_state.json").write_text(json.dumps(state))
        resp = c.get("/api/index")
        data = json.loads(resp.data)
        assert data["approved"] is True


class TestApproveIndex:
    def test_approve_action_sets_approved_true(self, client):
        c, dirs = client
        state = {
            "current_stage": "IndexGen",
            "outputs": {"IndexGen": {
                "index": [{"session_number": "01", "title": "T1", "estimated_minutes": 15, "concepts": []}],
                "approved": False
            }},
            "session_results": {}
        }
        state_file = dirs["state_dir"] / "pipeline_state.json"
        state_file.write_text(json.dumps(state))
        resp = c.post("/api/index", json={"action": "approve"})
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "approved"
        # Verify file was updated
        updated = json.loads(state_file.read_text())
        assert updated["outputs"]["IndexGen"]["approved"] is True

    def test_approve_without_index_returns_400(self, client):
        c, dirs = client
        state = {"current_stage": "Pass1", "outputs": {}, "session_results": {}}
        (dirs["state_dir"] / "pipeline_state.json").write_text(json.dumps(state))
        resp = c.post("/api/index", json={"action": "approve"})
        assert resp.status_code == 400

    def test_edit_action_stores_instructions(self, client):
        c, dirs = client
        state = {
            "current_stage": "IndexGen",
            "outputs": {"IndexGen": {"index": [], "approved": False}},
            "session_results": {}
        }
        state_file = dirs["state_dir"] / "pipeline_state.json"
        state_file.write_text(json.dumps(state))
        resp = c.post("/api/index", json={"action": "edit", "instructions": "Make titles shorter"})
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "edit_submitted"

    def test_edit_without_instructions_returns_400(self, client):
        c, dirs = client
        state = {
            "current_stage": "IndexGen",
            "outputs": {"IndexGen": {"index": [], "approved": False}},
            "session_results": {}
        }
        (dirs["state_dir"] / "pipeline_state.json").write_text(json.dumps(state))
        resp = c.post("/api/index", json={"action": "edit", "instructions": ""})
        assert resp.status_code == 400

    def test_invalid_action_returns_400(self, client):
        c, dirs = client
        state = {"current_stage": "IndexGen", "outputs": {}, "session_results": {}}
        (dirs["state_dir"] / "pipeline_state.json").write_text(json.dumps(state))
        resp = c.post("/api/index", json={"action": "invalid_action"})
        assert resp.status_code == 400


class TestGetConfig:
    def test_returns_config_without_api_keys(self, client):
        c, dirs = client
        config = {
            "cerebras_api_key": "secret-key",
            "groq_api_key": "another-secret",
            "ollama_host": "http://localhost:11434",
            "ollama_model": "llama3.2:3b",
        }
        dirs["config_file"].write_text(json.dumps(config))
        resp = c.get("/api/config")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        # Keys should not be exposed
        assert "cerebras_api_key" not in data
        assert "groq_api_key" not in data
        # But presence flags should be there
        assert "has_cerebras_key" in data
        assert data["has_cerebras_key"] is True

    def test_returns_empty_config_when_no_file(self, client):
        c, dirs = client
        resp = c.get("/api/config")
        assert resp.status_code == 200

    def test_has_groq_key_false_when_missing(self, client):
        c, dirs = client
        config = {"cerebras_api_key": "key"}
        dirs["config_file"].write_text(json.dumps(config))
        resp = c.get("/api/config")
        data = json.loads(resp.data)
        assert data["has_groq_key"] is False


class TestUpdateConfig:
    def test_updates_ollama_host(self, client):
        c, dirs = client
        resp = c.post("/api/config", json={"ollama_host": "http://192.168.1.100:11434"})
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "updated"
        # Verify saved
        saved = json.loads(dirs["config_file"].read_text())
        assert saved["ollama_host"] == "http://192.168.1.100:11434"

    def test_invalid_ollama_host_returns_400(self, client):
        c, dirs = client
        resp = c.post("/api/config", json={"ollama_host": "not-a-url"})
        assert resp.status_code == 400

    def test_https_host_accepted(self, client):
        c, dirs = client
        resp = c.post("/api/config", json={"ollama_host": "https://remote.host:11434"})
        assert resp.status_code == 200


class TestGetSessions:
    def test_returns_empty_sessions_when_no_file(self, client):
        c, dirs = client
        resp = c.get("/api/sessions")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["sessions"] == {}

    def test_returns_sessions_from_file(self, client):
        c, dirs = client
        sessions = {
            "01": {"content": "Session 1 text", "title": "T1"},
            "02": {"content": "Session 2 text", "title": "T2"},
        }
        (dirs["output_dir"] / "sessions.json").write_text(json.dumps(sessions))
        resp = c.get("/api/sessions")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert len(data["sessions"]) == 2


class TestGetProgress:
    def test_returns_empty_when_no_file(self, client):
        c, dirs = client
        resp = c.get("/api/progress")
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data == {}

    def test_returns_progress_data(self, client):
        c, dirs = client
        progress = {"01": {"completed": True, "bookmark": 42}}
        (dirs["output_dir"] / "progress.json").write_text(json.dumps(progress))
        resp = c.get("/api/progress")
        data = json.loads(resp.data)
        assert data["01"]["completed"] is True


class TestUpdateProgress:
    def test_creates_new_session_progress(self, client):
        c, dirs = client
        resp = c.post("/api/progress", json={
            "session_id": "01",
            "updates": {"completed": True}
        })
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "updated"
        # Verify file written
        saved = json.loads((dirs["output_dir"] / "progress.json").read_text())
        assert saved["01"]["completed"] is True

    def test_updates_existing_session(self, client):
        c, dirs = client
        progress = {"01": {"bookmark": 10}}
        (dirs["output_dir"] / "progress.json").write_text(json.dumps(progress))
        c.post("/api/progress", json={"session_id": "01", "updates": {"completed": True}})
        saved = json.loads((dirs["output_dir"] / "progress.json").read_text())
        assert saved["01"]["bookmark"] == 10  # Preserved
        assert saved["01"]["completed"] is True  # Added

    def test_missing_session_id_returns_400(self, client):
        c, dirs = client
        resp = c.post("/api/progress", json={"updates": {"completed": True}})
        assert resp.status_code == 400


class TestSetBookmark:
    def test_sets_bookmark(self, client):
        c, dirs = client
        resp = c.post("/api/bookmark", json={"session_id": "01", "word_index": 150})
        assert resp.status_code == 200
        data = json.loads(resp.data)
        assert data["status"] == "bookmarked"
        saved = json.loads((dirs["output_dir"] / "progress.json").read_text())
        assert saved["01"]["bookmark"] == 150

    def test_updates_existing_bookmark(self, client):
        c, dirs = client
        progress = {"01": {"bookmark": 50}}
        (dirs["output_dir"] / "progress.json").write_text(json.dumps(progress))
        c.post("/api/bookmark", json={"session_id": "01", "word_index": 200})
        saved = json.loads((dirs["output_dir"] / "progress.json").read_text())
        assert saved["01"]["bookmark"] == 200

    def test_missing_session_id_returns_400(self, client):
        c, dirs = client
        resp = c.post("/api/bookmark", json={"word_index": 100})
        assert resp.status_code == 400

    def test_missing_word_index_returns_400(self, client):
        c, dirs = client
        resp = c.post("/api/bookmark", json={"session_id": "01"})
        assert resp.status_code == 400

    def test_word_index_zero_is_valid(self, client):
        c, dirs = client
        resp = c.post("/api/bookmark", json={"session_id": "01", "word_index": 0})
        assert resp.status_code == 200
        saved = json.loads((dirs["output_dir"] / "progress.json").read_text())
        assert saved["01"]["bookmark"] == 0
