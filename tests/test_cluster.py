"""
Tests for pipeline/cluster.py - cluster_concepts and _adjust_cluster_count
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.cluster import cluster_concepts, _adjust_cluster_count


def _make_concepts(n=30):
    return [
        {"name": f"Concept{i}", "description": f"Description of concept {i}"}
        for i in range(n)
    ]


def _make_cluster_result(cluster_names=None, concepts_per_cluster=10):
    if cluster_names is None:
        cluster_names = ["Memory & Hardware", "Attention Mechanisms", "Optimization", "Scheduling", "Inference"]
    return {
        "clusters": [
            {
                "name": name,
                "concepts": [f"Concept{i}" for i in range(j * concepts_per_cluster, (j + 1) * concepts_per_cluster)],
                "description": f"Description of {name} cluster",
            }
            for j, name in enumerate(cluster_names)
        ]
    }


class TestClusterConcepts:
    def test_happy_path_returns_clusters(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_cluster_result()
        concepts = _make_concepts(50)
        result = cluster_concepts(concepts, cerebras)
        assert "clusters" in result
        assert len(result["clusters"]) == 5

    def test_calls_generate_json_once_for_valid_result(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_cluster_result()
        cluster_concepts(_make_concepts(50), cerebras)
        assert cerebras.generate_json.call_count == 1

    def test_concept_names_sent_to_api(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_cluster_result()
        concepts = [{"name": "FlashAttention", "description": "Tiling trick"}]
        cluster_concepts(concepts, cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1]
        assert "FlashAttention" in user_prompt

    def test_raises_when_clusters_key_missing(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = {"invalid": "structure"}
        with pytest.raises(ValueError, match="missing 'clusters' key"):
            cluster_concepts(_make_concepts(50), cerebras)

    def test_fewer_than_5_clusters_triggers_adjust(self):
        cerebras = MagicMock()
        few_clusters = _make_cluster_result(["C1", "C2", "C3"])  # Only 3 clusters
        adjusted = _make_cluster_result()  # 5 clusters
        cerebras.generate_json.side_effect = [few_clusters, adjusted]
        result = cluster_concepts(_make_concepts(30), cerebras)
        assert cerebras.generate_json.call_count == 2

    def test_more_than_8_clusters_triggers_adjust(self):
        cerebras = MagicMock()
        many_clusters = _make_cluster_result(
            ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10"],
            concepts_per_cluster=3
        )
        adjusted = _make_cluster_result()
        cerebras.generate_json.side_effect = [many_clusters, adjusted]
        result = cluster_concepts(_make_concepts(30), cerebras)
        assert cerebras.generate_json.call_count == 2

    def test_missing_concepts_assigned_to_first_cluster(self, capsys):
        cerebras = MagicMock()
        # Cluster only assigns some concepts
        partial_result = {
            "clusters": [
                {"name": "ClusterA", "concepts": ["Concept0", "Concept1", "Concept2", "Concept3", "Concept4"],
                 "description": "Desc"},
                {"name": "ClusterB", "concepts": ["Concept5", "Concept6", "Concept7", "Concept8", "Concept9"],
                 "description": "Desc"},
                {"name": "ClusterC", "concepts": ["Concept10", "Concept11", "Concept12", "Concept13", "Concept14"],
                 "description": "Desc"},
                {"name": "ClusterD", "concepts": ["Concept15", "Concept16", "Concept17", "Concept18", "Concept19"],
                 "description": "Desc"},
                {"name": "ClusterE", "concepts": ["Concept20", "Concept21", "Concept22", "Concept23", "Concept24"],
                 "description": "Desc"},
            ]
        }
        concepts = _make_concepts(30)  # 30 concepts, only 25 assigned above
        cerebras.generate_json.return_value = partial_result
        result = cluster_concepts(concepts, cerebras)
        # Missing Concept25-29 should be added to first cluster
        all_assigned = []
        for cluster in result["clusters"]:
            all_assigned.extend(cluster["concepts"])
        assert len(all_assigned) == 30

    def test_cluster_descriptions_preserved(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_cluster_result()
        result = cluster_concepts(_make_concepts(50), cerebras)
        for cluster in result["clusters"]:
            assert "description" in cluster

    def test_prints_cluster_count(self, capsys):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_cluster_result()
        cluster_concepts(_make_concepts(50), cerebras)
        captured = capsys.readouterr()
        assert "5" in captured.out


class TestAdjustClusterCount:
    def test_too_few_instruction_mentions_merge(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_cluster_result()
        _adjust_cluster_count(_make_cluster_result(["C1", "C2"]), cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1]
        assert "merge" in user_prompt.lower() or "5-8" in user_prompt

    def test_too_many_instruction_mentions_split(self):
        cerebras = MagicMock()
        cerebras.generate_json.return_value = _make_cluster_result()
        ten_clusters = _make_cluster_result([f"C{i}" for i in range(10)], concepts_per_cluster=2)
        _adjust_cluster_count(ten_clusters, cerebras)
        call_args = cerebras.generate_json.call_args
        user_prompt = call_args[0][1]
        assert "split" in user_prompt.lower() or "5-8" in user_prompt

    def test_returns_adjusted_result(self):
        cerebras = MagicMock()
        adjusted = _make_cluster_result()
        cerebras.generate_json.return_value = adjusted
        result = _adjust_cluster_count(_make_cluster_result(["C1", "C2"]), cerebras)
        assert len(result["clusters"]) == 5
