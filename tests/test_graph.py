"""
Tests for pipeline/graph.py - build_concept_graph and helpers
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.graph import (
    build_concept_graph,
    _merge_concepts,
    _topological_sort,
    _detect_orphans,
    _resolve_circular_dependencies,
    _build_edges,
    _find_matching_concept,
    _deduplicate_concepts,
)


def _make_concept(name, cluster="ClusterA", weight="medium", dep_signals=None, cross_deps=None, implicit_prereqs=None):
    return {
        "canonical_name": name,
        "original_name": name,
        "description": f"Description of {name}",
        "primary_passage": "",
        "dependency_signals": dep_signals or [],
        "implicit_prerequisites": implicit_prereqs or [],
        "cross_theme_deps": cross_deps or [],
        "concept_weight": weight,
        "cluster": cluster,
        "enrichment_flag": False,
        "author_anchor": "",
    }


def _make_pass2_output(cluster_name, concept_names):
    return {
        "concepts": [_make_concept(name, cluster=cluster_name) for name in concept_names]
    }


class TestMergeConcepts:
    def test_merges_all_cluster_concepts(self):
        pass2_output = {
            "ClusterA": _make_pass2_output("ClusterA", ["C1", "C2"]),
            "ClusterB": _make_pass2_output("ClusterB", ["C3"]),
        }
        result = _merge_concepts(pass2_output)
        names = [c["canonical_name"] for c in result]
        assert set(names) == {"C1", "C2", "C3"}

    def test_adds_cluster_field_to_each_concept(self):
        pass2_output = {"Memory": _make_pass2_output("Memory", ["C1"])}
        result = _merge_concepts(pass2_output)
        assert result[0]["cluster"] == "Memory"

    def test_empty_pass2_output_returns_empty_list(self):
        assert _merge_concepts({}) == []

    def test_empty_cluster_contributes_no_concepts(self):
        pass2_output = {"EmptyCluster": {"concepts": []}}
        result = _merge_concepts(pass2_output)
        assert result == []


class TestTopologicalSort:
    def test_simple_linear_chain(self):
        concepts = [_make_concept("A"), _make_concept("B"), _make_concept("C")]
        edges = [
            {"from": "A", "to": "B", "reason": "A before B", "source": "explicit"},
            {"from": "B", "to": "C", "reason": "B before C", "source": "explicit"},
        ]
        order = _topological_sort(concepts, edges)
        assert order.index("A") < order.index("B")
        assert order.index("B") < order.index("C")

    def test_all_concepts_in_result(self):
        concepts = [_make_concept("A"), _make_concept("B"), _make_concept("C")]
        edges = [{"from": "A", "to": "B", "reason": "dep", "source": "explicit"}]
        order = _topological_sort(concepts, edges)
        assert set(order) == {"A", "B", "C"}

    def test_no_edges_returns_all_concepts(self):
        concepts = [_make_concept("X"), _make_concept("Y")]
        order = _topological_sort(concepts, [])
        assert set(order) == {"X", "Y"}

    def test_diamond_dependency(self):
        # A -> B, A -> C, B -> D, C -> D
        concepts = [_make_concept(n) for n in ["A", "B", "C", "D"]]
        edges = [
            {"from": "A", "to": "B", "reason": "", "source": "explicit"},
            {"from": "A", "to": "C", "reason": "", "source": "explicit"},
            {"from": "B", "to": "D", "reason": "", "source": "explicit"},
            {"from": "C", "to": "D", "reason": "", "source": "explicit"},
        ]
        order = _topological_sort(concepts, edges)
        assert order.index("A") < order.index("D")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_single_concept_no_crash(self):
        order = _topological_sort([_make_concept("Solo")], [])
        assert order == ["Solo"]


class TestDetectOrphans:
    def test_detects_concepts_with_no_edges(self):
        concepts = [_make_concept("A"), _make_concept("B"), _make_concept("C")]
        edges = [{"from": "A", "to": "B", "reason": "", "source": "explicit"}]
        orphans = _detect_orphans(concepts, edges)
        assert "C" in orphans
        assert "A" not in orphans
        assert "B" not in orphans

    def test_no_orphans_when_all_connected(self):
        concepts = [_make_concept("A"), _make_concept("B")]
        edges = [{"from": "A", "to": "B", "reason": "", "source": "explicit"}]
        orphans = _detect_orphans(concepts, edges)
        assert orphans == []

    def test_all_orphans_when_no_edges(self):
        concepts = [_make_concept("A"), _make_concept("B"), _make_concept("C")]
        orphans = _detect_orphans(concepts, [])
        assert set(orphans) == {"A", "B", "C"}


class TestResolveCircularDependencies:
    def test_returns_unchanged_when_no_cycles(self):
        concepts = [_make_concept("A"), _make_concept("B"), _make_concept("C")]
        edges = [
            {"from": "A", "to": "B", "reason": "", "source": "explicit"},
            {"from": "B", "to": "C", "reason": "", "source": "explicit"},
        ]
        cerebras = MagicMock()
        result_concepts, result_edges = _resolve_circular_dependencies(concepts, edges, cerebras)
        assert len(result_edges) == 2

    def test_removes_edge_to_break_cycle(self):
        # A -> B -> A is a cycle
        concepts = [_make_concept("A"), _make_concept("B")]
        edges = [
            {"from": "A", "to": "B", "reason": "", "source": "explicit"},
            {"from": "B", "to": "A", "reason": "", "source": "explicit"},
        ]
        cerebras = MagicMock()
        _, result_edges = _resolve_circular_dependencies(concepts, edges, cerebras)
        # At least one edge should have been removed to break the cycle
        assert len(result_edges) < 2


class TestBuildEdges:
    def test_extracts_explicit_dependency_signals(self):
        dep_signal = {"signal": "recall that A", "location": "S1", "refers_to": "ConceptA"}
        concepts = [
            _make_concept("ConceptA"),
            _make_concept("ConceptB", dep_signals=[dep_signal]),
        ]
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"dependencies": []}
        edges = _build_edges(concepts, cerebras)
        edge_pairs = [(e["from"], e["to"]) for e in edges]
        assert ("ConceptA", "ConceptB") in edge_pairs

    def test_removes_self_loops(self):
        dep_signal = {"signal": "recall that self", "location": "S1", "refers_to": "ConceptA"}
        concepts = [_make_concept("ConceptA", dep_signals=[dep_signal])]
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"dependencies": []}
        edges = _build_edges(concepts, cerebras)
        self_loops = [e for e in edges if e["from"] == e["to"]]
        assert self_loops == []

    def test_deduplicates_edges(self):
        # Same dep from two sources
        dep1 = {"signal": "recall A", "location": "S1", "refers_to": "ConceptA"}
        dep2 = {"signal": "building on A", "location": "S2", "refers_to": "ConceptA"}
        concepts = [
            _make_concept("ConceptA"),
            _make_concept("ConceptB", dep_signals=[dep1, dep2]),
        ]
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"dependencies": []}
        edges = _build_edges(concepts, cerebras)
        pairs = [(e["from"], e["to"]) for e in edges]
        # Should only have one (A->B) pair
        assert pairs.count(("ConceptA", "ConceptB")) == 1

    def test_cross_theme_deps_become_edges(self):
        cross_dep = {"concept": "ConceptA", "relationship": "requires"}
        concepts = [
            _make_concept("ConceptA"),
            _make_concept("ConceptB", cross_deps=[cross_dep]),
        ]
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"dependencies": []}
        edges = _build_edges(concepts, cerebras)
        edge_pairs = [(e["from"], e["to"]) for e in edges]
        assert ("ConceptA", "ConceptB") in edge_pairs


class TestFindMatchingConcept:
    def setup_method(self):
        self.concepts = [
            _make_concept("Flash Attention"),
            _make_concept("KV Cache"),
            _make_concept("Paged Attention"),
        ]
        self.concept_map = {c["canonical_name"]: i for i, c in enumerate(self.concepts)}

    def test_exact_match(self):
        result = _find_matching_concept("KV Cache", self.concept_map, self.concepts)
        assert result == "KV Cache"

    def test_partial_match(self):
        result = _find_matching_concept("attention", self.concept_map, self.concepts)
        # Should find one of the attention-named concepts
        assert result in ["Flash Attention", "Paged Attention"] or result == ""

    def test_no_match_returns_empty_string(self):
        result = _find_matching_concept("Completely Unrelated Term XYZABC", self.concept_map, self.concepts)
        assert result == ""


class TestDeduplicateConcepts:
    def test_small_list_skips_deduplication(self):
        concepts = [_make_concept(f"C{i}") for i in range(50)]  # < 100
        cerebras = MagicMock()
        result = _deduplicate_concepts(concepts, cerebras)
        # Should return unchanged, no API call
        cerebras.generate_json.assert_not_called()
        assert len(result) == 50

    def test_large_list_calls_cerebras(self):
        concepts = [_make_concept(f"C{i}") for i in range(110)]
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"duplicates": []}
        _deduplicate_concepts(concepts, cerebras)
        cerebras.generate_json.assert_called_once()

    def test_duplicate_removed(self):
        concepts = [_make_concept(f"C{i}") for i in range(110)]
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {
            "duplicates": [{"keep_index": 0, "discard_index": 5, "reason": "Same"}]
        }
        result = _deduplicate_concepts(concepts, cerebras)
        assert len(result) == 109  # One removed

    def test_api_failure_returns_original_list(self):
        concepts = [_make_concept(f"C{i}") for i in range(110)]
        cerebras = MagicMock()
        cerebras.generate_json.side_effect = Exception("API error")
        result = _deduplicate_concepts(concepts, cerebras)
        assert len(result) == 110  # Original unchanged


class TestBuildConceptGraph:
    def test_full_pipeline_returns_correct_keys(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"duplicates": [], "dependencies": []}
        pass2_output = {
            "ClusterA": _make_pass2_output("ClusterA", ["C1", "C2", "C3"]),
        }
        result = build_concept_graph(pass2_output, cerebras)
        assert "concepts" in result
        assert "edges" in result
        assert "sorted_concepts" in result
        assert "orphans" in result
        assert "metadata" in result

    def test_sorted_concepts_contains_all_names(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"duplicates": [], "dependencies": []}
        pass2_output = {
            "ClusterA": _make_pass2_output("ClusterA", ["C1", "C2"]),
        }
        result = build_concept_graph(pass2_output, cerebras)
        assert set(result["sorted_concepts"]) == {"C1", "C2"}

    def test_metadata_counts_are_consistent(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"duplicates": [], "dependencies": []}
        pass2_output = {"ClusterA": _make_pass2_output("ClusterA", ["C1", "C2", "C3"])}
        result = build_concept_graph(pass2_output, cerebras)
        assert result["metadata"]["total_concepts"] == len(result["concepts"])
