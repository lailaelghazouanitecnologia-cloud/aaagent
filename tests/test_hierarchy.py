"""Tests for hierarchy optimization module."""

import torch
import pytest

from hierarchy.analyzer import ClusterAnalyzer, HierarchySnapshot, ClusterProfile
from hierarchy.llm_judge import LLMJudge, JudgmentResult
from hierarchy.actions import Action, ActionType, HierarchyModifier
from hierarchy.search import HierarchySearch, SearchResult


# ── Minimal model mock ──

class MockCentroids(torch.nn.Module):
    def __init__(self, n, d):
        super().__init__()
        self.n_clusters = n
        self.centroids = torch.nn.Parameter(torch.randn(n, d) * 0.02)

    def forward(self, weights):
        return torch.matmul(weights, self.centroids)

    def pairwise_cosine_similarity(self):
        normed = self.centroids / (self.centroids.norm(dim=-1, keepdim=True) + 1e-8)
        return torch.matmul(normed, normed.T)


class MockRouter(torch.nn.Module):
    def __init__(self, d, n):
        super().__init__()
        self.linear = torch.nn.Linear(d, n, bias=False)

    def forward(self, x, temperature=None):
        logits = self.linear(x)
        if temperature is not None:
            logits = logits / temperature
        return torch.softmax(logits, dim=-1)


class MockLocalEmbedding(torch.nn.Module):
    def __init__(self, vocab_size, d):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, d)

    def forward(self, x):
        return self.embedding(x)


class MockEmbedding(torch.nn.Module):
    def __init__(self, vocab_size=128, d=32, K=8, M=2):
        super().__init__()
        self.local_embedding = MockLocalEmbedding(vocab_size, d)
        self.fine_router = MockRouter(d, K)
        self.fine_centroids = MockCentroids(K, d)
        self.coarse_router = MockRouter(d, M)
        self.coarse_centroids = MockCentroids(M, d)


class MockModel(torch.nn.Module):
    def __init__(self, vocab_size=128, d=32, K=8, M=2):
        super().__init__()
        self.embedding = MockEmbedding(vocab_size, d, K, M)


# ── Tests ──

class TestClusterAnalyzer:
    def test_analyze_produces_snapshot(self):
        model = MockModel()
        analyzer = ClusterAnalyzer(model, tokenizer=None, device="cpu")
        snap = analyzer.analyze(step=100, top_k=5)

        assert isinstance(snap, HierarchySnapshot)
        assert snap.step == 100
        assert len(snap.fine_profiles) == 8  # K=8
        assert len(snap.coarse_groups) == 2  # M=2

    def test_profiles_have_tokens(self):
        model = MockModel()
        analyzer = ClusterAnalyzer(model, tokenizer=None, device="cpu")
        snap = analyzer.analyze(step=0, top_k=5)

        for p in snap.fine_profiles:
            assert isinstance(p, ClusterProfile)
            assert len(p.top_tokens) <= 5
            assert p.cluster_id >= 0

    def test_format_for_llm(self):
        model = MockModel()
        analyzer = ClusterAnalyzer(model, tokenizer=None, device="cpu")
        snap = analyzer.analyze(step=0)
        text = analyzer.format_for_llm(snap)

        assert "Cluster Hierarchy Analysis" in text
        assert "Fine clusters: 8" in text
        assert "Coarse groups: 2" in text
        assert "Group" in text
        assert "Cluster" in text

    def test_entropy_values(self):
        model = MockModel()
        analyzer = ClusterAnalyzer(model, tokenizer=None, device="cpu")
        snap = analyzer.analyze(step=0)

        assert snap.fine_entropy > 0  # Not collapsed
        assert snap.coarse_entropy >= 0


class TestLLMJudge:
    def test_mock_evaluate(self):
        judge = LLMJudge(backend="mock")
        result = judge.evaluate("test cluster description", n_fine=8, n_coarse=2)

        assert isinstance(result, JudgmentResult)
        assert 0 <= result.quality_score <= 10
        assert 0 <= result.coherence <= 10
        assert 0 <= result.balance <= 10

    def test_parse_valid_json(self):
        judge = LLMJudge(backend="mock")
        text = '{"quality_score": 7.5, "coherence": 8.0, "balance": 6.0, "separation": 7.0, "suggested_merges": [[1, 2]], "suggested_splits": [3], "reassign": [], "reasoning": "good"}'
        result = judge._parse_response(text)

        assert result.quality_score == 7.5
        assert result.coherence == 8.0
        assert result.suggested_merges == [(1, 2)]
        assert result.suggested_splits == [3]
        assert result.reasoning == "good"

    def test_parse_invalid_json(self):
        judge = LLMJudge(backend="mock")
        result = judge._parse_response("not json at all")
        assert result.quality_score == 5.0  # Default
        assert result.reasoning == "parse_failed"

    def test_parse_markdown_fenced(self):
        judge = LLMJudge(backend="mock")
        text = '```json\n{"quality_score": 9.0, "coherence": 8.5, "balance": 7.0, "separation": 8.0, "suggested_merges": [], "suggested_splits": [], "reassign": [], "reasoning": "excellent"}\n```'
        result = judge._parse_response(text)
        assert result.quality_score == 9.0


class TestActions:
    def test_merge_fine(self):
        model = MockModel(K=8, M=2)
        modifier = HierarchyModifier(model)
        c_before_a = model.embedding.fine_centroids.centroids[0].clone()
        c_before_b = model.embedding.fine_centroids.centroids[1].clone()

        action = Action(ActionType.MERGE_FINE, {"cluster_a": 0, "cluster_b": 1})
        success = modifier.apply(action)
        assert success

        # After merge, both centroids should be the average
        expected = (c_before_a + c_before_b) / 2
        assert torch.allclose(model.embedding.fine_centroids.centroids[0], expected)

    def test_split_fine(self):
        model = MockModel(K=8, M=2)
        modifier = HierarchyModifier(model)
        c_before = model.embedding.fine_centroids.centroids[0].clone()

        action = Action(ActionType.SPLIT_FINE, {"cluster_id": 0})
        success = modifier.apply(action)
        assert success

        # Original centroid should have changed (noise added)
        assert not torch.allclose(model.embedding.fine_centroids.centroids[0], c_before)

    def test_rebalance(self):
        model = MockModel(K=8, M=2)
        modifier = HierarchyModifier(model)
        c_before = model.embedding.coarse_centroids.centroids.clone()

        action = Action(ActionType.REBALANCE, {})
        success = modifier.apply(action)
        assert success

        # Coarse centroids should have changed
        assert not torch.allclose(model.embedding.coarse_centroids.centroids, c_before)

    def test_reassign(self):
        model = MockModel(K=8, M=2)
        modifier = HierarchyModifier(model)
        w_before = model.embedding.coarse_router.linear.weight.clone()

        action = Action(ActionType.REASSIGN, {"fine_id": 0, "to_coarse": 1})
        success = modifier.apply(action)
        assert success

        # Router weights should have changed
        assert not torch.allclose(model.embedding.coarse_router.linear.weight, w_before)

    def test_reset_empty(self):
        model = MockModel(K=8, M=2)
        # Zero out one cluster's router weight to simulate empty
        model.embedding.fine_router.linear.weight.data[3] *= 0

        modifier = HierarchyModifier(model)
        action = Action(ActionType.RESET_EMPTY, {"threshold": 0.01})
        success = modifier.apply(action)
        assert success

    def test_invalid_merge(self):
        model = MockModel(K=8, M=2)
        modifier = HierarchyModifier(model)
        action = Action(ActionType.MERGE_FINE, {"cluster_a": 0, "cluster_b": 999})
        success = modifier.apply(action)
        assert not success


class TestHierarchySearch:
    def test_search_runs(self):
        model = MockModel(K=8, M=2)
        judge = LLMJudge(backend="mock")
        search = HierarchySearch(
            model=model,
            judge=judge,
            max_nodes=5,
            max_depth=2,
            device="cpu",
        )

        result = search.search(step=100)
        assert isinstance(result, SearchResult)
        assert result.nodes_explored > 0
        assert result.initial_score >= 0
        assert result.time_seconds >= 0

    def test_search_with_zero_nodes(self):
        model = MockModel(K=8, M=2)
        search = HierarchySearch(model=model, max_nodes=0, device="cpu")
        result = search.search(step=0)
        assert result.nodes_explored == 0

    def test_state_save_restore(self):
        model = MockModel(K=8, M=2)
        search = HierarchySearch(model=model, max_nodes=1, device="cpu")

        state = search._save_state()
        # Modify model
        model.embedding.fine_centroids.centroids.data *= 2.0
        # Restore
        search._restore_state(state)

        # Should be back to original
        for name in state:
            param = dict(model.named_parameters())[name]
            assert torch.allclose(param.data, state[name])
