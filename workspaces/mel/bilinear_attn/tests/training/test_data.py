"""Tests for data loading and tokenization."""
import pytest
import torch

from data.tokenization import get_tokenizer


def test_get_tokenizer():
    """Test that tokenizer loads correctly."""
    tokenizer = get_tokenizer()
    
    assert tokenizer is not None
    assert tokenizer.pad_token is not None


def test_tokenizer_encode_decode():
    """Test tokenizer encode/decode roundtrip."""
    tokenizer = get_tokenizer()
    
    text = "Hello, world!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    
    assert "Hello" in decoded or "hello" in decoded.lower()
    assert "world" in decoded or "world" in decoded.lower()
