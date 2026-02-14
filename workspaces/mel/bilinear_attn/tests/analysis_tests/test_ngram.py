"""Tests for n-gram analysis."""
import pytest
import tempfile
import torch
from pathlib import Path

from analysis.behaviour.ngram import NgramAnalyzer
from tests.analysis_tests.conftest import V, T, B, PAD_TOKEN


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
    
    loss = analyzer.compute_ngram_loss(model, n=2, bos_token_id=0, batch_size=8)
    
    assert isinstance(loss, float)
    assert loss > 0
    assert loss < float('inf')


def test_test_loss_computation(model, dummy_dataloader):
    """Test computing test loss (l_test)."""
    analyzer = NgramAnalyzer(vocab_size=V)
    
    loss = analyzer.compute_test_loss(
        model, dummy_dataloader, n=2, bos_token_id=0, max_batches=5
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
        model, dummy_dataloader, n=2, bos_token_id=0, max_val_batches=5
    )
    
    assert "2gram_loss" in scores
    assert "2gram_test_loss" in scores
    assert "2gram_score" in scores
    assert scores["2gram_score"] > 0


def test_all_ngram_scores(model, dummy_dataloader):
    """Test computing all n-gram scores."""
    analyzer = NgramAnalyzer(vocab_size=V, max_common_ngrams=50)
    analyzer.extract_common_ngrams_from_data(
        dummy_dataloader, max_n=3, max_samples=30
    )
    
    all_scores = analyzer.compute_all_ngram_scores(
        model, dummy_dataloader, bos_token_id=0, max_val_batches=5
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


def test_ngram_save_load(dummy_dataloader):
    """Test saving and loading n-gram data."""
    analyzer = NgramAnalyzer(vocab_size=V, max_common_ngrams=50)
    analyzer.extract_common_ngrams_from_data(
        dummy_dataloader, max_n=4, max_samples=30
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "ngram.pt")
        analyzer.save(path)
        
        loaded = NgramAnalyzer.load(path)
        
        assert loaded._is_fitted
        assert loaded.vocab_size == V
        assert loaded.max_common_ngrams == 50
        
        # Check common_ngrams match for each n
        for n in analyzer.common_ngrams:
            assert n in loaded.common_ngrams
            assert len(loaded.common_ngrams[n]) == len(analyzer.common_ngrams[n])


def test_ngram_ignores_padding(padded_dataloader):
    """Test that n-gram extraction ignores padding tokens."""
    analyzer = NgramAnalyzer(vocab_size=V, max_common_ngrams=100)
    analyzer.extract_common_ngrams_from_data(
        padded_dataloader, max_n=4, max_samples=50
    )
    
    assert analyzer._is_fitted
    
    # Real tokens are in [1, V); PAD_TOKEN positions have attention_mask=0.
    # No extracted n-gram should contain the padding token.
    for n, ngrams in analyzer.common_ngrams.items():
        for context, final in ngrams:
            assert PAD_TOKEN not in context, f"Padding token found in {n}-gram context: {context}"
            assert final != PAD_TOKEN, f"Padding token found as {n}-gram final token"


def test_ngram_not_fitted_raises():
    """Test that using unfitted analyzer raises error."""
    analyzer = NgramAnalyzer(vocab_size=V)
    
    # Create a dummy model for testing
    import torch.nn as nn
    model = nn.Linear(V, V)
    
    with pytest.raises(RuntimeError):
        analyzer.compute_ngram_loss(model, n=2, bos_token_id=0)
