"""
Tests for pipeline/generator.py - generate_session_content and helpers
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.generator import (
    generate_session_content,
    _get_previous_context,
    _prepare_concept_metadata,
    _prepare_dependency_context,
    _get_cluster_description,
    _extend_concepts_body,
    _trim_to_words,
)


def _make_concept(name, weight="medium", cluster="ClusterA", enrichment=False):
    return {
        "canonical_name": name,
        "description": f"Description of {name}",
        "primary_passage": f"Passage for {name}",
        "author_anchor": f"Anchor: {name}",
        "concept_weight": weight,
        "enrichment_flag": enrichment,
        "cluster": cluster,
        "cross_theme_deps": [],
    }


def _make_session_plan(session_number="01", concepts=None, revisit=None):
    if concepts is None:
        concepts = [
            {"name": "ConceptA", "description": "Desc A", "weight": "medium"},
            {"name": "ConceptB", "description": "Desc B", "weight": "light"},
        ]
    plan = {
        "session_number": session_number,
        "title": "Test Title",
        "concepts": concepts,
        "estimated_minutes": 15,
    }
    if revisit:
        plan["revisit"] = revisit
    return plan


def _make_graph_output(concepts=None, edges=None):
    if concepts is None:
        concepts = [_make_concept("ConceptA"), _make_concept("ConceptB")]
    return {
        "concepts": concepts,
        "edges": edges or [],
        "sorted_concepts": [c["canonical_name"] for c in concepts],
    }


def _make_state_manager(sessions=None):
    sm = MagicMock()
    sm.get_session_results.return_value = sessions or {}
    return sm


class TestGetPreviousContext:
    def test_first_session_returns_is_first_true(self):
        sm = _make_state_manager()
        ctx = _get_previous_context("01", sm)
        assert ctx["is_first"] is True
        assert ctx["text"] == ""

    def test_non_first_with_previous_result(self):
        prev_text = "word " * 300  # 300 words
        sm = _make_state_manager({
            "01": {"content": prev_text, "concepts": [{"name": "C1"}]}
        })
        ctx = _get_previous_context("02", sm)
        assert ctx["is_first"] is False
        # Last 200 words
        words = ctx["text"].split()
        assert len(words) <= 200

    def test_non_first_missing_previous_returns_empty(self):
        sm = _make_state_manager({})  # No session 01
        ctx = _get_previous_context("02", sm)
        assert ctx["is_first"] is False
        assert ctx["text"] == ""

    def test_previous_concepts_extracted(self):
        sm = _make_state_manager({
            "02": {"content": "some text", "concepts": [{"name": "C1"}, {"name": "C2"}]}
        })
        ctx = _get_previous_context("03", sm)
        assert "C1" in ctx["concepts"]
        assert "C2" in ctx["concepts"]


class TestPrepareConceptMetadata:
    def test_all_concepts_in_metadata(self):
        concepts = [
            {"name": "C1", "description": "D1", "weight": "medium"},
            {"name": "C2", "description": "D2", "weight": "light"},
        ]
        all_concepts = {
            "C1": _make_concept("C1"),
            "C2": _make_concept("C2"),
        }
        meta = _prepare_concept_metadata(concepts, None, all_concepts)
        assert "C1" in meta
        assert "C2" in meta

    def test_revisit_metadata_stored_under_dunder_key(self):
        concepts = [{"name": "C1", "description": "D1", "weight": "medium"}]
        all_concepts = {"C1": _make_concept("C1"), "RevisitConcept": _make_concept("RevisitConcept")}
        revisit = {"name": "RevisitConcept", "reason": "connects to C1"}
        meta = _prepare_concept_metadata(concepts, revisit, all_concepts)
        assert "__revisit__" in meta
        assert meta["__revisit__"]["name"] == "RevisitConcept"
        assert meta["__revisit__"]["reason"] == "connects to C1"

    def test_no_revisit_no_dunder_key(self):
        concepts = [{"name": "C1", "description": "D1", "weight": "medium"}]
        all_concepts = {"C1": _make_concept("C1")}
        meta = _prepare_concept_metadata(concepts, None, all_concepts)
        assert "__revisit__" not in meta

    def test_enrichment_flag_propagated(self):
        concepts = [{"name": "C1", "description": "D1", "weight": "medium"}]
        all_concepts = {"C1": _make_concept("C1", enrichment=True)}
        meta = _prepare_concept_metadata(concepts, None, all_concepts)
        assert meta["C1"]["enrichment_flag"] is True


class TestPrepareDependencyContext:
    def test_prerequisites_identified(self):
        concepts = [{"name": "ConceptB"}]
        all_concepts = {
            "ConceptA": _make_concept("ConceptA"),
            "ConceptB": _make_concept("ConceptB"),
        }
        graph = _make_graph_output(
            concepts=[_make_concept("ConceptA"), _make_concept("ConceptB")],
            edges=[{"from": "ConceptA", "to": "ConceptB", "reason": "dep", "source": "explicit"}],
        )
        ctx = _prepare_dependency_context(concepts, all_concepts, graph)
        assert "ConceptB" in ctx["prerequisites"]

    def test_dependents_identified(self):
        concepts = [{"name": "ConceptA"}]
        all_concepts = {
            "ConceptA": _make_concept("ConceptA"),
            "ConceptB": _make_concept("ConceptB"),
        }
        graph = _make_graph_output(
            concepts=[_make_concept("ConceptA"), _make_concept("ConceptB")],
            edges=[{"from": "ConceptA", "to": "ConceptB", "reason": "dep", "source": "explicit"}],
        )
        ctx = _prepare_dependency_context(concepts, all_concepts, graph)
        assert "ConceptA" in ctx["dependents"]

    def test_empty_edges_returns_empty_dicts(self):
        concepts = [{"name": "C1"}]
        ctx = _prepare_dependency_context(concepts, {}, _make_graph_output(edges=[]))
        assert ctx["prerequisites"] == {}
        assert ctx["dependents"] == {}


class TestGetClusterDescription:
    def test_returns_description_string(self):
        concepts = [{"name": "C1"}, {"name": "C2"}]
        all_concepts = {
            "C1": _make_concept("C1", cluster="Memory"),
            "C2": _make_concept("C2", cluster="Memory"),
        }
        desc = _get_cluster_description(concepts, all_concepts, _make_graph_output())
        assert "Memory" in desc

    def test_returns_dominant_cluster(self):
        concepts = [{"name": "C1"}, {"name": "C2"}, {"name": "C3"}]
        all_concepts = {
            "C1": _make_concept("C1", cluster="Memory"),
            "C2": _make_concept("C2", cluster="Memory"),
            "C3": _make_concept("C3", cluster="Inference"),
        }
        desc = _get_cluster_description(concepts, all_concepts, _make_graph_output())
        assert "Memory" in desc  # Memory has 2, Inference has 1

    def test_no_cluster_returns_empty_string(self):
        desc = _get_cluster_description([], {}, _make_graph_output())
        assert desc == ""


class TestTrimToWords:
    def test_short_text_returned_unchanged(self):
        text = "This is short text"
        result = _trim_to_words(text, 100)
        assert result == text

    def test_long_text_trimmed(self):
        text = ("word " * 3000).strip()
        result = _trim_to_words(text, 2400)
        words = result.split()
        assert len(words) <= 2400

    def test_trims_to_sentence_boundary(self):
        text = "First sentence here. Second sentence here. " * 1000
        result = _trim_to_words(text, 100)
        # Result should end with a period
        assert result.endswith(".")

    def test_empty_text_returns_empty(self):
        assert _trim_to_words("", 100) == ""


class TestGenerateSessionContent:
    def test_happy_path_returns_tuple(self):
        cerebras = MagicMock()
        cerebras.generate.return_value = "word " * 2000
        sm = _make_state_manager()
        plan = _make_session_plan("01")
        graph = _make_graph_output()
        result = generate_session_content(plan, graph, sm, cerebras)
        assert isinstance(result, tuple)
        session_text, tension_excerpt = result
        assert isinstance(session_text, str)
        assert len(session_text) > 0
        assert isinstance(tension_excerpt, str)

    def test_tension_excerpt_is_last_80_words(self):
        cerebras = MagicMock()
        # 200 distinct words so we can verify the excerpt
        words = [f"w{i}" for i in range(200)]
        cerebras.generate.return_value = " ".join(words)
        sm = _make_state_manager()
        plan = _make_session_plan("01")
        graph = _make_graph_output()
        _, tension_excerpt = generate_session_content(plan, graph, sm, cerebras)
        excerpt_words = tension_excerpt.split()
        assert len(excerpt_words) <= 80

    def test_calls_cerebras_generate(self):
        cerebras = MagicMock()
        cerebras.generate.return_value = "word " * 2000
        sm = _make_state_manager()
        plan = _make_session_plan("01")
        graph = _make_graph_output()
        generate_session_content(plan, graph, sm, cerebras)
        # 3-pass architecture: at least 3 calls (tension plan, body, anchor+tension)
        assert cerebras.generate.call_count >= 3

    def test_first_session_skips_anchor_in_prompt(self):
        cerebras = MagicMock()
        cerebras.generate.return_value = "word " * 2000
        sm = _make_state_manager()
        plan = _make_session_plan("01")
        graph = _make_graph_output()
        generate_session_content(plan, graph, sm, cerebras)
        # Check system prompt of the anchor+tension call (3rd call) mentions session 01 handling
        all_calls = cerebras.generate.call_args_list
        # The anchor+tension system prompt references "Session 01"
        found = any(
            "Session 01" in str(c) or "session 01" in str(c).lower() or "skip" in str(c).lower()
            for c in all_calls
        )
        assert found

    def test_enrichment_flag_enables_web_search(self):
        cerebras = MagicMock()
        cerebras.generate.return_value = "word " * 2000
        sm = _make_state_manager()
        enriched_concept = _make_concept("EnrichedConcept", enrichment=True)
        graph = _make_graph_output(concepts=[enriched_concept])
        plan = _make_session_plan("01", concepts=[{"name": "EnrichedConcept", "description": "D", "weight": "medium"}])
        generate_session_content(plan, graph, sm, cerebras)
        # The concepts body call (2nd call) should have enable_web_search=True
        all_calls = cerebras.generate.call_args_list
        web_search_calls = [c for c in all_calls if c[1].get("enable_web_search") is True]
        assert len(web_search_calls) >= 1

    def test_short_session_triggers_extension(self):
        cerebras = MagicMock()
        # 3-pass calls: tension(short), body(short), anchor+tension(short), then extend body
        cerebras.generate.side_effect = [
            "tension question here",            # pass 1: tension plan
            "too short " * 50,                  # pass 2: body too short
            "anchor text\n\ntension text",      # pass 3: anchor+tension
            "long enough " * 500,               # extend body call
        ]
        sm = _make_state_manager()
        plan = _make_session_plan("01")
        graph = _make_graph_output()
        generate_session_content(plan, graph, sm, cerebras)
        # Should have made 4 calls (3 passes + 1 extension)
        assert cerebras.generate.call_count == 4

    def test_long_session_trimmed(self):
        cerebras = MagicMock()
        # Return long text for all calls; body call produces the bulk
        cerebras.generate.return_value = "word " * 3000
        sm = _make_state_manager()
        plan = _make_session_plan("01")
        graph = _make_graph_output()
        session_text, _ = generate_session_content(plan, graph, sm, cerebras)
        word_count = len(session_text.split())
        assert word_count <= 2400

    def test_revisit_metadata_included_in_prompt(self):
        cerebras = MagicMock()
        cerebras.generate.return_value = "word " * 2000
        sm = _make_state_manager()
        revisit_concept = _make_concept("RevisitTarget")
        graph = _make_graph_output(concepts=[
            _make_concept("ConceptA"),
            revisit_concept,
        ])
        plan = _make_session_plan(
            "03",
            revisit={"name": "RevisitTarget", "reason": "Unique revisit reason XYZ"},
        )
        generate_session_content(plan, graph, sm, cerebras)
        # RevisitTarget must appear in the concepts body call (2nd call)
        all_calls = cerebras.generate.call_args_list
        found = any("RevisitTarget" in str(c) for c in all_calls)
        assert found
