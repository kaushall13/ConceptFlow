"""
Shared pytest fixtures for the Micro-Learning Curriculum Builder test suite.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the project root is on the path so all imports work
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Lightweight concept / session helpers
# ---------------------------------------------------------------------------

def make_concept(name="ConceptA", weight="medium", cluster="ClusterX", enrichment=False):
    return {
        "canonical_name": name,
        "original_name": name,
        "description": f"Description of {name}",
        "primary_passage": f"Passage for {name}",
        "secondary_passages": [],
        "dependency_signals": [],
        "implicit_prerequisites": [],
        "author_anchor": f"Anchor for {name}",
        "enrichment_flag": enrichment,
        "concept_weight": weight,
        "cross_theme_deps": [],
        "cluster": cluster,
    }


def make_session_plan(session_number="01", concepts=None, revisit=None):
    if concepts is None:
        concepts = [
            {"name": "ConceptA", "description": "Desc A", "weight": "medium"},
            {"name": "ConceptB", "description": "Desc B", "weight": "light"},
        ]
    plan = {
        "session_number": session_number,
        "title": f"Session {session_number} Title",
        "concepts": concepts,
        "estimated_minutes": 15,
        "total_weight": 4,
        "concept_count": len(concepts),
    }
    if revisit:
        plan["revisit"] = revisit
    return plan


def make_graph_output(concepts=None, edges=None, sorted_concepts=None):
    if concepts is None:
        concepts = [make_concept("ConceptA"), make_concept("ConceptB")]
    if edges is None:
        edges = []
    if sorted_concepts is None:
        sorted_concepts = [c["canonical_name"] for c in concepts]
    return {
        "concepts": concepts,
        "edges": edges,
        "sorted_concepts": sorted_concepts,
        "orphans": [],
        "metadata": {
            "total_concepts": len(concepts),
            "total_edges": len(edges),
            "orphan_count": 0,
        },
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_cerebras():
    """Mock CerebrasAPI client that returns configurable JSON or text."""
    client = MagicMock()
    client.generate.return_value = "Generated text response"
    client.generate_json.return_value = {"concepts": [], "clusters": []}
    client.rate_limiter = MagicMock()
    return client


@pytest.fixture
def mock_ollama():
    """Mock OllamaAPI client with default passing evaluations."""
    client = MagicMock()
    client.check_tension.return_value = (True, "Tension is well-formed")
    client.check_anchor.return_value = (True, "Anchor is well-formed")
    client.check_coherence.return_value = (True, "Coherent session")
    client.evaluate_binary.return_value = (True, "Condition met")
    return client


@pytest.fixture
def state_manager(tmp_path):
    """StateManager backed by a temporary directory."""
    from state.manager import StateManager
    sm = StateManager(state_dir=str(tmp_path / "state"))
    return sm


@pytest.fixture
def sample_concepts():
    return [
        make_concept("FlashAttention", "heavy", "Attention"),
        make_concept("KV Cache", "medium", "Memory"),
        make_concept("Continuous Batching", "medium", "Scheduling"),
        make_concept("Paged Attention", "light", "Memory"),
        make_concept("Beam Search", "light", "Decoding"),
    ]


@pytest.fixture
def sample_session_plan():
    return make_session_plan()


@pytest.fixture
def sample_graph_output(sample_concepts):
    return make_graph_output(
        concepts=sample_concepts,
        edges=[
            {"from": "KV Cache", "to": "Paged Attention", "reason": "Paged Attention builds on KV Cache", "source": "explicit"},
        ],
        sorted_concepts=[c["canonical_name"] for c in sample_concepts],
    )


@pytest.fixture
def long_session_text():
    """Generate a session text in the 1800-2400 word range."""
    paragraph = ("The concept of flash attention fundamentally changes how transformers "
                 "handle the NxN attention matrix. Instead of materializing the full matrix "
                 "in high-bandwidth memory, it tiles computations to keep data in SRAM. "
                 "This is a concrete example of how memory bandwidth, not compute, is "
                 "often the true bottleneck in modern GPU workloads. ")
    # ~1900 words
    return paragraph * 38


@pytest.fixture
def short_session_text():
    """Session text that is too short (< 1500 words)."""
    return "This is a very short session. " * 50  # ~300 words


@pytest.fixture
def index_gen_output():
    return {
        "index": [
            {
                "session_number": "01",
                "title": "Why Your GPU Is Starving",
                "estimated_minutes": 15,
                "concepts": [
                    {"name": "FlashAttention", "description": "Tiling avoids HBM materialisation"},
                    {"name": "KV Cache", "description": "Stores KV pairs to avoid recomputation"},
                ],
            },
            {
                "session_number": "02",
                "title": "Memory Hierarchies Matter",
                "estimated_minutes": 14,
                "concepts": [
                    {"name": "Paged Attention", "description": "Block-based KV cache management"},
                ],
                "revisit": {"name": "FlashAttention", "reason": "Paged attention is flash attention's cousin"},
            },
        ],
        "approved": False,
    }
