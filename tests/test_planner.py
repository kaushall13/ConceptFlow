"""
Tests for pipeline/planner.py - plan_sessions, _group_into_sessions, _assign_revisits_heuristic,
_generate_titles_batched, revise_index
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.planner import (
    plan_sessions,
    _group_into_sessions,
    _finalize_session,
    _assign_revisits_heuristic,
    _generate_titles_batched,
    _generate_index,
    _validate_index,
    revise_index,
)


def _make_concept_dict(name, weight="medium", cluster="ClusterA"):
    return {
        "canonical_name": name,
        "description": f"Description of {name}",
        "concept_weight": weight,
        "cluster": cluster,
        "cross_theme_deps": [],
        "enrichment_flag": False,
    }


def _make_all_concepts_dict(concepts):
    return {c["canonical_name"]: c for c in concepts}


def _make_graph_output(concept_names, weights=None, clusters=None, edges=None):
    if weights is None:
        weights = ["medium"] * len(concept_names)
    if clusters is None:
        clusters = ["ClusterA"] * len(concept_names)
    concepts = [
        _make_concept_dict(name, weight=w, cluster=cl)
        for name, w, cl in zip(concept_names, weights, clusters)
    ]
    return {
        "concepts": concepts,
        "edges": edges or [],
        "sorted_concepts": concept_names,
        "orphans": [],
        "metadata": {},
    }


class TestGroupIntoSessions:
    def test_all_medium_concepts_grouped_correctly(self):
        names = [f"C{i}" for i in range(12)]
        all_concepts = _make_all_concepts_dict([_make_concept_dict(n) for n in names])
        sessions = _group_into_sessions(names, all_concepts)
        # All concepts must appear
        covered = [c["name"] for s in sessions for c in s["concepts"]]
        assert set(covered) == set(names)

    def test_heavy_concept_gets_own_session(self):
        names = ["Heavy", "Light1", "Light2", "Light3"]
        all_concepts = _make_all_concepts_dict([
            _make_concept_dict("Heavy", weight="heavy"),
            _make_concept_dict("Light1", weight="light"),
            _make_concept_dict("Light2", weight="light"),
            _make_concept_dict("Light3", weight="light"),
        ])
        sessions = _group_into_sessions(names, all_concepts)
        # Heavy must be alone in its session
        heavy_sessions = [s for s in sessions if any(c["name"] == "Heavy" for c in s["concepts"])]
        assert len(heavy_sessions) == 1
        assert len(heavy_sessions[0]["concepts"]) == 1

    def test_no_session_exceeds_5_concepts(self):
        names = [f"C{i}" for i in range(20)]
        all_concepts = _make_all_concepts_dict([_make_concept_dict(n, weight="light") for n in names])
        sessions = _group_into_sessions(names, all_concepts)
        for s in sessions:
            assert len(s["concepts"]) <= 5

    def test_empty_input_returns_empty_sessions(self):
        sessions = _group_into_sessions([], {})
        assert sessions == []

    def test_single_concept_creates_one_session(self):
        all_concepts = {"Solo": _make_concept_dict("Solo")}
        sessions = _group_into_sessions(["Solo"], all_concepts)
        assert len(sessions) == 1
        assert sessions[0]["concepts"][0]["name"] == "Solo"

    def test_no_concept_split_across_sessions(self):
        names = [f"C{i}" for i in range(10)]
        all_concepts = _make_all_concepts_dict([_make_concept_dict(n) for n in names])
        sessions = _group_into_sessions(names, all_concepts)
        all_seen = []
        for s in sessions:
            for c in s["concepts"]:
                all_seen.append(c["name"])
        # No duplicates
        assert len(all_seen) == len(set(all_seen))


class TestFinalizeSession:
    def test_returns_required_fields(self):
        all_concepts = _make_all_concepts_dict([_make_concept_dict("A"), _make_concept_dict("B")])
        session = _finalize_session(["A", "B"], all_concepts)
        assert "concepts" in session
        assert "concept_count" in session
        assert "total_weight" in session
        assert "estimated_minutes" in session

    def test_concept_count_matches(self):
        all_concepts = _make_all_concepts_dict([_make_concept_dict("A"), _make_concept_dict("B")])
        session = _finalize_session(["A", "B"], all_concepts)
        assert session["concept_count"] == 2

    def test_estimated_minutes_in_valid_range(self):
        all_concepts = _make_all_concepts_dict([_make_concept_dict("A")])
        session = _finalize_session(["A"], all_concepts)
        assert 10 <= session["estimated_minutes"] <= 20

    def test_unknown_concept_uses_defaults(self):
        session = _finalize_session(["NonExistent"], {})
        assert session["concept_count"] == 1


class TestAssignRevisits:
    def _make_sessions(self, n, weight="medium"):
        all_concept_dicts = [_make_concept_dict(f"C{i}") for i in range(n)]
        all_concepts = _make_all_concepts_dict(all_concept_dicts)
        sessions = []
        for i in range(n):
            sessions.append({
                "concepts": [{"name": f"C{i}", "description": "desc", "weight": weight}],
                "concept_count": 1,
                "total_weight": 2,
                "estimated_minutes": 15,
            })
        return sessions, all_concepts

    def test_first_three_sessions_have_no_revisit(self):
        sessions, all_concepts = self._make_sessions(10)
        llm = MagicMock()
        result = _assign_revisits_heuristic(sessions, all_concepts, llm)
        for s in result[:3]:
            assert s.get("revisit") is None

    def test_sessions_beyond_3_may_have_revisit(self):
        sessions, all_concepts = self._make_sessions(15)
        llm = MagicMock()
        result = _assign_revisits_heuristic(sessions, all_concepts, llm)
        # Not all will have revisit (score threshold), but structure is present
        for s in result:
            assert "revisit" in s

    def test_concept_not_revisited_twice(self):
        sessions, all_concepts = self._make_sessions(15)
        # Return None from LLM so all revisits come from heuristics (no mock pollution)
        llm = MagicMock()
        llm.generate_json.return_value = {"choice": None, "reason": ""}
        result = _assign_revisits_heuristic(sessions, all_concepts, llm)
        revisited = [s["revisit"]["name"] for s in result if s.get("revisit")]
        assert len(revisited) == len(set(revisited))

    def test_revisit_structure_has_name_and_reason(self):
        sessions, all_concepts = self._make_sessions(15)
        # Force high cross_theme_dep score by adding cross deps
        for s in sessions[8:]:
            for c in s["concepts"]:
                all_concepts[c["name"]]["cross_theme_deps"] = [
                    {"concept": f"C{s['concepts'][0]['name'][-1]}", "relationship": "requires"}
                ]

        llm = MagicMock()
        result = _assign_revisits_heuristic(sessions, all_concepts, llm)
        for s in result:
            if s.get("revisit"):
                assert "name" in s["revisit"]
                assert "reason" in s["revisit"]


class TestGenerateTitlesBatched:
    def test_titles_assigned_to_all_sessions(self):
        sessions = [
            {"concepts": [{"name": "C1", "description": ""}], "concept_count": 1, "total_weight": 2, "estimated_minutes": 15},
            {"concepts": [{"name": "C2", "description": ""}], "concept_count": 1, "total_weight": 2, "estimated_minutes": 15},
        ]
        all_concepts = {}
        llm = MagicMock()
        llm.generate_json.return_value = {"titles": ["Why Batching Wins", "Memory Is Everything"]}
        result = _generate_titles_batched(sessions, all_concepts, llm)
        assert result[0]["title"] == "Why Batching Wins"
        assert result[1]["title"] == "Memory Is Everything"

    def test_session_numbers_assigned(self):
        sessions = [
            {"concepts": [{"name": "C1", "description": ""}], "concept_count": 1, "total_weight": 2, "estimated_minutes": 15},
            {"concepts": [{"name": "C2", "description": ""}], "concept_count": 1, "total_weight": 2, "estimated_minutes": 15},
        ]
        llm = MagicMock()
        llm.generate_json.return_value = {"titles": ["T1", "T2"]}
        result = _generate_titles_batched(sessions, {}, llm)
        assert result[0]["session_number"] == "01"
        assert result[1]["session_number"] == "02"

    def test_fallback_title_when_batch_fails(self):
        sessions = [
            {"concepts": [{"name": "ConceptA", "description": ""}], "concept_count": 1, "total_weight": 2, "estimated_minutes": 15},
        ]
        llm = MagicMock()
        llm.generate_json.side_effect = Exception("API error")
        result = _generate_titles_batched(sessions, {}, llm)
        # Fallback: single concept title = concept name
        assert result[0]["title"] == "ConceptA"

    def test_fallback_two_concepts_title(self):
        sessions = [{
            "concepts": [
                {"name": "Alpha", "description": ""},
                {"name": "Beta", "description": ""},
            ],
            "concept_count": 2, "total_weight": 4, "estimated_minutes": 15,
        }]
        llm = MagicMock()
        llm.generate_json.side_effect = Exception("fail")
        result = _generate_titles_batched(sessions, {}, llm)
        assert "Alpha" in result[0]["title"] and "Beta" in result[0]["title"]

    def test_invalid_title_from_batch_uses_fallback(self):
        sessions = [
            {"concepts": [{"name": "ConceptX", "description": ""}], "concept_count": 1, "total_weight": 2, "estimated_minutes": 15},
        ]
        llm = MagicMock()
        # Title with only 1 word fails the 2-10 word check
        llm.generate_json.return_value = {"titles": ["X"]}
        result = _generate_titles_batched(sessions, {}, llm)
        assert result[0]["title"] == "ConceptX"


class TestValidateIndex:
    def test_valid_index_returns_true(self):
        index = [
            {
                "session_number": "01", "title": "T1", "estimated_minutes": 15,
                "concepts": [{"name": "C1", "description": "Desc"}]
            }
        ]
        assert _validate_index(index) is True

    def test_empty_list_returns_false(self):
        assert _validate_index([]) is False

    def test_missing_session_number_returns_false(self):
        index = [{"title": "T1", "estimated_minutes": 15, "concepts": [{"name": "C1"}]}]
        assert _validate_index(index) is False

    def test_empty_concepts_returns_false(self):
        index = [{"session_number": "01", "title": "T1", "estimated_minutes": 15, "concepts": []}]
        assert _validate_index(index) is False

    def test_non_list_returns_false(self):
        assert _validate_index({"not": "a list"}) is False


class TestReviseIndex:
    def test_returns_revised_index_on_success(self):
        current = [
            {"session_number": "01", "title": "T1", "estimated_minutes": 15,
             "concepts": [{"name": "C1", "description": "D1"}]}
        ]
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"index": current}
        result = revise_index(current, "Make titles more evocative", cerebras)
        assert result == current

    def test_returns_original_on_api_failure(self):
        current = [
            {"session_number": "01", "title": "T1", "estimated_minutes": 15,
             "concepts": [{"name": "C1", "description": "D1"}]}
        ]
        cerebras = MagicMock()
        cerebras.generate_json.side_effect = Exception("API error")
        result = revise_index(current, "Edit", cerebras)
        assert result == current

    def test_returns_original_when_revised_invalid(self):
        current = [
            {"session_number": "01", "title": "T1", "estimated_minutes": 15,
             "concepts": [{"name": "C1", "description": "D1"}]}
        ]
        cerebras = MagicMock()
        # Return invalid structure
        cerebras.generate_json.return_value = {"index": []}  # Empty index fails validation
        result = revise_index(current, "Edit", cerebras)
        assert result == current

    def test_instructions_included_in_api_call(self):
        current = [
            {"session_number": "01", "title": "T1", "estimated_minutes": 15,
             "concepts": [{"name": "C1", "description": "D1"}]}
        ]
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"index": current}
        revise_index(current, "UNIQUE_INSTRUCTION_MARKER_XYZ", cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1]
        assert "UNIQUE_INSTRUCTION_MARKER_XYZ" in user_prompt
