"""
Pipeline State Manager - JSON-based persistence with atomic writes
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional


class StateManager:
    """Manages pipeline state persistence with atomic writes."""

    def __init__(self, state_dir: str = None, state_file: str = "pipeline_state.json", book_slug: str = None):
        if state_dir is None:
            state_dir = f"data/{book_slug}/state" if book_slug else "state"
        self.state_dir = Path(state_dir)
        self.state_file = self.state_dir / state_file
        self._ensure_state_dir()

    def _ensure_state_dir(self):
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def load_state(self) -> Dict[str, Any]:
        if not self.state_file.exists():
            return {
                "current_stage": "Not started",
                "outputs": {},
                "session_results": {}
            }
        with open(self.state_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def save_stage(self, stage_name: str, output: Any) -> None:
        state = self.load_state()
        state["current_stage"] = stage_name
        state["outputs"][stage_name] = output
        self._atomic_write(state)

    def save_session_result(self, session_id: str, result: Dict[str, Any]) -> None:
        state = self.load_state()
        state["session_results"][str(session_id)] = result
        self._atomic_write(state)

    def get_stage_output(self, stage_name: str) -> Optional[Any]:
        state = self.load_state()
        return state.get("outputs", {}).get(stage_name)

    def get_current_stage(self) -> str:
        state = self.load_state()
        return state.get("current_stage", "Not started")

    def is_stage_complete(self, stage_name: str) -> bool:
        state = self.load_state()
        return stage_name in state.get("outputs", {})

    def reset_stage(self, stage_name: str) -> None:
        state = self.load_state()
        if stage_name in state.get("outputs", {}):
            del state["outputs"][stage_name]
        if state.get("current_stage") == stage_name:
            state["current_stage"] = "Not started"
        self._atomic_write(state)

    def reset_all(self) -> None:
        state = {
            "current_stage": "Not started",
            "outputs": {},
            "session_results": {}
        }
        self._atomic_write(state)

    def mark_complete(self) -> None:
        state = self.load_state()
        state["current_stage"] = "Complete"
        self._atomic_write(state)

    def is_complete(self) -> bool:
        return self.get_current_stage() == "Complete"

    def _atomic_write(self, state: Dict[str, Any]) -> None:
        temp_fd, temp_path = tempfile.mkstemp(dir=self.state_dir, suffix='.tmp')
        try:
            with os.fdopen(temp_fd, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, self.state_file)
        except Exception:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise

    def get_session_results(self) -> Dict[str, Any]:
        state = self.load_state()
        return state.get("session_results", {})

    def get_session_result(self, session_num: int) -> Dict[str, Any]:
        results = self.get_session_results()
        return results.get(str(session_num)) or results.get(session_num)

    def get_progress_info(self) -> Dict[str, Any]:
        state = self.load_state()
        outputs = state.get("outputs", {})
        session_results = state.get("session_results", {})

        return {
            "current_stage": state.get("current_stage", "Not started"),
            "completed_stages": list(outputs.keys()),
            "sessions_generated": len(session_results),
            "total_sessions": self._get_total_sessions(state)
        }

    def _get_total_sessions(self, state: Dict[str, Any]) -> int:
        # IndexGen is where we save the session index
        index_gen_output = state.get("outputs", {}).get("IndexGen", {})
        if isinstance(index_gen_output, dict) and "index" in index_gen_output:
            index = index_gen_output["index"]
            if isinstance(index, list):
                return len(index)
        return 0


_state_manager = None


def get_state_manager() -> StateManager:
    global _state_manager
    if _state_manager is None:
        _state_manager = StateManager()
    return _state_manager
