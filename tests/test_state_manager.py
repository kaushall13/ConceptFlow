"""
Tests for state/manager.py - StateManager class
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from state.manager import StateManager


@pytest.fixture
def sm(tmp_path):
    return StateManager(state_dir=str(tmp_path / "state"))


class TestInitAndLoad:
    def test_creates_state_directory(self, tmp_path):
        state_dir = tmp_path / "new_state"
        assert not state_dir.exists()
        StateManager(state_dir=str(state_dir))
        assert state_dir.exists()

    def test_load_returns_defaults_when_no_file(self, sm):
        state = sm.load_state()
        assert state["current_stage"] == "Not started"
        assert state["outputs"] == {}
        assert state["session_results"] == {}

    def test_load_returns_persisted_data(self, sm):
        sm.save_stage("Ingest", {"text": "hello"})
        state = sm.load_state()
        assert state["current_stage"] == "Ingest"
        assert state["outputs"]["Ingest"] == {"text": "hello"}


class TestSaveStage:
    def test_saves_stage_output(self, sm):
        sm.save_stage("Pass1", {"concepts": [{"name": "C1"}]})
        output = sm.get_stage_output("Pass1")
        assert output == {"concepts": [{"name": "C1"}]}

    def test_updates_current_stage(self, sm):
        sm.save_stage("Cluster", {"clusters": []})
        assert sm.get_current_stage() == "Cluster"

    def test_multiple_stage_saves_accumulate(self, sm):
        sm.save_stage("Ingest", "ingested text")
        sm.save_stage("Pass1", {"concepts": []})
        state = sm.load_state()
        assert "Ingest" in state["outputs"]
        assert "Pass1" in state["outputs"]

    def test_overwrites_existing_stage_output(self, sm):
        sm.save_stage("Ingest", "original")
        sm.save_stage("Ingest", "updated")
        assert sm.get_stage_output("Ingest") == "updated"

    def test_saves_complex_nested_output(self, sm):
        data = {"concepts": [{"name": "C", "weight": "heavy", "deps": [1, 2, 3]}]}
        sm.save_stage("GraphBuild", data)
        assert sm.get_stage_output("GraphBuild") == data


class TestSaveSessionResult:
    def test_saves_session_result(self, sm):
        sm.save_session_result("01", {"content": "Session text", "passed": True})
        results = sm.get_session_results()
        assert "01" in results
        assert results["01"]["content"] == "Session text"

    def test_saves_multiple_sessions(self, sm):
        sm.save_session_result("01", {"content": "S1"})
        sm.save_session_result("02", {"content": "S2"})
        results = sm.get_session_results()
        assert len(results) == 2

    def test_session_id_stored_as_string(self, sm):
        sm.save_session_result(1, {"content": "data"})  # integer id
        results = sm.get_session_results()
        assert "1" in results  # stored as str

    def test_overwrites_existing_session(self, sm):
        sm.save_session_result("01", {"content": "old"})
        sm.save_session_result("01", {"content": "new"})
        assert sm.get_session_results()["01"]["content"] == "new"


class TestGetStageOutput:
    def test_returns_none_for_missing_stage(self, sm):
        assert sm.get_stage_output("NonExistent") is None

    def test_returns_correct_output(self, sm):
        sm.save_stage("Pass2", [1, 2, 3])
        assert sm.get_stage_output("Pass2") == [1, 2, 3]


class TestIsStageComplete:
    def test_returns_false_when_not_run(self, sm):
        assert sm.is_stage_complete("Ingest") is False

    def test_returns_true_after_save(self, sm):
        sm.save_stage("Ingest", "text")
        assert sm.is_stage_complete("Ingest") is True


class TestResetStage:
    def test_removes_stage_output(self, sm):
        sm.save_stage("Ingest", "text")
        sm.reset_stage("Ingest")
        assert sm.get_stage_output("Ingest") is None

    def test_resets_current_stage_when_it_matches(self, sm):
        sm.save_stage("Ingest", "text")
        sm.reset_stage("Ingest")
        assert sm.get_current_stage() == "Not started"

    def test_does_not_reset_current_stage_if_different(self, sm):
        sm.save_stage("Ingest", "text")
        sm.save_stage("Pass1", {})
        sm.reset_stage("Ingest")  # Reset an earlier stage
        assert sm.get_current_stage() == "Pass1"  # Current stage unchanged

    def test_no_error_on_nonexistent_stage(self, sm):
        sm.reset_stage("NonExistent")  # Should not raise


class TestResetAll:
    def test_clears_all_data(self, sm):
        sm.save_stage("Ingest", "text")
        sm.save_session_result("01", {"content": "S1"})
        sm.reset_all()
        state = sm.load_state()
        assert state["current_stage"] == "Not started"
        assert state["outputs"] == {}
        assert state["session_results"] == {}


class TestMarkComplete:
    def test_marks_pipeline_complete(self, sm):
        sm.mark_complete()
        assert sm.is_complete() is True
        assert sm.get_current_stage() == "Complete"

    def test_is_complete_returns_false_initially(self, sm):
        assert sm.is_complete() is False


class TestGetProgressInfo:
    def test_returns_progress_structure(self, sm):
        info = sm.get_progress_info()
        assert "current_stage" in info
        assert "completed_stages" in info
        assert "sessions_generated" in info
        assert "total_sessions" in info

    def test_sessions_generated_counts_session_results(self, sm):
        sm.save_session_result("01", {"content": "S1"})
        sm.save_session_result("02", {"content": "S2"})
        info = sm.get_progress_info()
        assert info["sessions_generated"] == 2

    def test_total_sessions_from_index_gen(self, sm):
        sm.save_stage("IndexGen", {
            "index": [
                {"session_number": "01", "title": "T1", "concepts": []},
                {"session_number": "02", "title": "T2", "concepts": []},
            ],
            "approved": False,
        })
        info = sm.get_progress_info()
        assert info["total_sessions"] == 2

    def test_total_sessions_zero_when_no_index(self, sm):
        info = sm.get_progress_info()
        assert info["total_sessions"] == 0

    def test_completed_stages_lists_saved_stages(self, sm):
        sm.save_stage("Ingest", "text")
        sm.save_stage("Pass1", {})
        info = sm.get_progress_info()
        assert "Ingest" in info["completed_stages"]
        assert "Pass1" in info["completed_stages"]


class TestAtomicWrite:
    def test_file_is_valid_json_after_write(self, sm, tmp_path):
        sm.save_stage("Ingest", {"key": "value"})
        state_file = sm.state_file
        with open(state_file, "r") as f:
            data = json.load(f)
        assert data["outputs"]["Ingest"] == {"key": "value"}

    def test_no_temp_files_left_after_write(self, sm, tmp_path):
        sm.save_stage("Ingest", "data")
        state_dir = sm.state_dir
        tmp_files = list(state_dir.glob("*.tmp"))
        assert tmp_files == []
