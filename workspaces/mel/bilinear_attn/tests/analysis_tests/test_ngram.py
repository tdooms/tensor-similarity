"""Tests for n-gram analysis."""
import pytest
import torch

from analysis.behaviour.ngram import NgramAnalyzer
from tests.analysis_tests.conftest import V, T, B


def test_ngram_analyzer_init():
    """Test NgramAnalyzer initialization."""
    analyzer = NgramAnalyzer(vocab_size=V)
    
    assert analyzer.vocab_size == V
    assert not analyzer._is_fitted


def test_ngram_extract_from_data(dummy_dataloader):
    """Test extracting n-grams from data."""
    analyzer = NgramAnalyzer(vocab_size=V, max_common_ngrams=100)
    analyzer.extract_common_ngrams_from_data(
        dummy_dataloader, max_n=4, max_samples=30
    )
    
    assert analyzer._is_fitted
    assert 2 in analyzer.common_ngrams
    assert len(analyzer.common_ngrams[2]) > 0


def test_ngram_loss_computation(model, dummy_dataloader):
    """Test computing n-gram loss."""
    analyzer = NgramAnalyzer(vocab_size=V, max_common_ngrams=50)
    analyzer.extract_common_ngrams_from_data(
        dummy_dataloader, max_n=3, max_samples=30
    )
    
    loss = analyzer.compute_ngram_loss(model, n=2, batch_size=8)
    
    assert isinstance(loss, float)
    assert loss > 0
    assert loss < float('inf')


def test_position_loss_computation(model, dummy_dataloader):
    """Test computing position loss."""
    analyzer = NgramAnalyzer(vocab_size=V)
    
    loss = analyzer.compute_position_loss(
        model, dummy_dataloader, position=1, max_batches=5
    )
    
    assert isinstance(loss, float)
    assert loss > 0
    assert loss < float('inf')


def test_ngram_score_computation(model, dummy_dataloader):
    """Test computing n-gram score."""
    analyzer = NgramAnalyzer(vocab_size=V, max_common_ngrams=50)
    analyzer.extract_common_ngrams_from_data(
        dummy_dataloader, max_n=3, max_samples=30
    )
    
    scores = analyzer.compute_ngram_score(
        model, dummy_dataloader, n=2, max_val_batches=5
    )
    
    assert "2gram_loss" in scores
    assert "position_2_loss" in scores
    assert "2gram_score" in scores
    assert scores["2gram_score"] > 0


def test_all_ngram_scores(model, dummy_dataloader):
    """Test computing all n-gram scores."""
    analyzer = NgramAnalyzer(vocab_size=V, max_common_ngrams=50)
    analyzer.extract_common_ngrams_from_data(
        dummy_dataloader, max_n=3, max_samples=30
    )
    
    all_scores = analyzer.compute_all_ngram_scores(
        model, dummy_dataloader, max_val_batches=5
    )
    
    assert "2gram_score" in all_scores
    assert "3gram_score" in all_scores


def test_ngram_stats(dummy_dataloader):
    """Test getting n-gram statistics."""
    analyzer = NgramAnalyzer(vocab_size=V)
    
    stats_before = analyzer.get_stats()
    assert stats_before["fitted"] == False
    
    analyzer.extract_common_ngrams_from_data(
        dummy_dataloader, max_n=3, max_samples=30
    )
    
    stats_after = analyzer.get_stats()
    assert stats_after["fitted"] == True


def test_ngram_not_fitted_raises():
    """Test that using unfitted analyzer raises error."""
    analyzer = NgramAnalyzer(vocab_size=V)
    
    # Create a dummy model for testing
    import torch.nn as nn
    model = nn.Linear(V, V)
    
    with pytest.raises(RuntimeError):
        analyzer.compute_ngram_loss(model, n=2)
