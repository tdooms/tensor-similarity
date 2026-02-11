"""Tests for bigram analysis."""
import pytest
import torch

from analysis.behaviour.bigram import BigramAnalyzer
from tests.analysis_tests.conftest import V, T, B


def test_bigram_analyzer_init():
    """Test BigramAnalyzer initialization."""
    analyzer = BigramAnalyzer(vocab_size=V)
    
    assert analyzer.vocab_size == V
    assert not analyzer._is_fitted


def test_bigram_analyzer_fit(dummy_dataloader):
    """Test fitting bigram distribution."""
    analyzer = BigramAnalyzer(vocab_size=V)
    analyzer.fit(dummy_dataloader, max_samples=20)
    
    assert analyzer._is_fitted
    assert analyzer.total_bigrams > 0
    assert len(analyzer.bigram_counts) > 0


def test_bigram_conditional_distribution(dummy_dataloader):
    """Test getting conditional distribution."""
    analyzer = BigramAnalyzer(vocab_size=V)
    analyzer.fit(dummy_dataloader, max_samples=20)
    
    # Get distribution for a token that appeared
    context_token = list(analyzer.bigram_counts.keys())[0]
    dist = analyzer.get_conditional_distribution(context_token)
    
    assert dist.shape == (V,)
    assert torch.isclose(dist.sum(), torch.tensor(1.0), atol=1e-5)
    assert (dist >= 0).all()


def test_bigram_entropy(dummy_dataloader):
    """Test computing bigram entropy."""
    analyzer = BigramAnalyzer(vocab_size=V)
    analyzer.fit(dummy_dataloader, max_samples=20)
    
    context_token = list(analyzer.bigram_counts.keys())[0]
    entropy = analyzer.compute_bigram_entropy(context_token)
    
    assert entropy >= 0
    assert entropy < float('inf')


def test_bigram_score_computation(model, dummy_dataloader):
    """Test computing bigram score."""
    analyzer = BigramAnalyzer(vocab_size=V)
    analyzer.fit(dummy_dataloader, max_samples=20)
    
    # Get a batch
    batch = next(iter(dummy_dataloader))
    input_ids = batch["input_ids"]
    
    score = analyzer.compute_bigram_score(model, input_ids, position=0)
    
    assert isinstance(score, float)
    assert score > 0
    assert score < float('inf')


def test_average_bigram_score(model, dummy_dataloader):
    """Test computing average bigram score."""
    analyzer = BigramAnalyzer(vocab_size=V)
    analyzer.fit(dummy_dataloader, max_samples=20)
    
    avg_score, avg_entropy = analyzer.compute_average_bigram_score(
        model, dummy_dataloader, n_samples=10, seed=42
    )
    
    assert isinstance(avg_score, float)
    assert isinstance(avg_entropy, float)
    assert avg_score > 0
    assert avg_entropy > 0


def test_bigram_stats(dummy_dataloader):
    """Test getting bigram statistics."""
    analyzer = BigramAnalyzer(vocab_size=V)
    
    stats_before = analyzer.get_stats()
    assert stats_before["fitted"] == False
    
    analyzer.fit(dummy_dataloader, max_samples=20)
    
    stats_after = analyzer.get_stats()
    assert stats_after["fitted"] == True
    assert stats_after["total_bigrams"] > 0
    assert stats_after["unique_contexts"] > 0


def test_bigram_not_fitted_raises():
    """Test that using unfitted analyzer raises error."""
    analyzer = BigramAnalyzer(vocab_size=V)
    
    with pytest.raises(RuntimeError):
        analyzer.get_conditional_distribution(0)
