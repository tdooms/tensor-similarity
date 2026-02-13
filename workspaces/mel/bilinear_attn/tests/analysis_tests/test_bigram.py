"""Tests for bigram analysis."""
import pytest
import tempfile
import torch
from pathlib import Path

from analysis.behaviour.bigram import BigramAnalyzer
from tests.analysis_tests.conftest import V, T, B, PAD_TOKEN


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
    assert analyzer.count_matrix is not None
    assert analyzer.count_matrix.shape == (V, V)


def test_bigram_conditional_distribution(dummy_dataloader):
    """Test getting conditional distribution (sparse)."""
    analyzer = BigramAnalyzer(vocab_size=V)
    analyzer.fit(dummy_dataloader, max_samples=20)
    
    # Get distribution for a token that appeared (find one with nonzero row)
    context_token = int((analyzer.count_matrix.sum(dim=1) > 0).nonzero()[0].item())
    indices, probs = analyzer.get_conditional_distribution(context_token)
    
    assert indices.numel() > 0
    assert indices.numel() == probs.numel()
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)
    assert (probs > 0).all()
    
    # Unseen context should return empty
    unseen = int((analyzer.count_matrix.sum(dim=1) == 0).nonzero()[0].item())
    idx_empty, p_empty = analyzer.get_conditional_distribution(unseen)
    assert idx_empty.numel() == 0
    assert p_empty.numel() == 0


def test_bigram_entropy(dummy_dataloader):
    """Test computing bigram entropy."""
    analyzer = BigramAnalyzer(vocab_size=V)
    analyzer.fit(dummy_dataloader, max_samples=20)
    
    context_token = int((analyzer.count_matrix.sum(dim=1) > 0).nonzero()[0].item())
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
        model, dummy_dataloader.dataset, n_samples=10, seed=42
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


def test_bigram_save_load(dummy_dataloader):
    """Test saving and loading bigram distribution."""
    analyzer = BigramAnalyzer(vocab_size=V)
    analyzer.fit(dummy_dataloader, max_samples=20)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        path = str(Path(tmpdir) / "bigram.pt")
        analyzer.save(path)
        
        loaded = BigramAnalyzer.load(path)
        
        assert loaded._is_fitted
        assert loaded.vocab_size == V
        assert loaded.total_bigrams == analyzer.total_bigrams
        
        # Check conditional distributions match
        context_token = int((analyzer.count_matrix.sum(dim=1) > 0).nonzero()[0].item())
        orig_idx, orig_probs = analyzer.get_conditional_distribution(context_token)
        loaded_idx, loaded_probs = loaded.get_conditional_distribution(context_token)
        assert torch.equal(orig_idx, loaded_idx)
        assert torch.allclose(orig_probs, loaded_probs)


def test_bigram_ignores_padding(padded_dataloader):
    """Test that bigram fitting ignores padding tokens."""
    analyzer = BigramAnalyzer(vocab_size=V)
    analyzer.fit(padded_dataloader, max_samples=50)
    
    assert analyzer._is_fitted
    assert analyzer.total_bigrams > 0
    
    # Real tokens are in [1, V); PAD_TOKEN positions have attention_mask=0.
    # The pad token should never appear as a context or next-token.
    pad_row = analyzer.count_matrix[PAD_TOKEN]
    assert pad_row.sum().item() == 0, "Padding token should not appear as bigram context"
    
    pad_col = analyzer.count_matrix[:, PAD_TOKEN]
    assert pad_col.sum().item() == 0, "Padding token should not appear as bigram next-token"


def test_bigram_not_fitted_raises():
    """Test that using unfitted analyzer raises error."""
    analyzer = BigramAnalyzer(vocab_size=V)
    
    with pytest.raises(RuntimeError):
        analyzer.get_conditional_distribution(0)
