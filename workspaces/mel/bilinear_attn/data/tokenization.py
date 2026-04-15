from transformers import AutoTokenizer

TOKENIZER_REPO = "SimpleStories/SimpleStories-5M"


def get_tokenizer(name: str = TOKENIZER_REPO):
    """Get a tokenizer by name.
    
    Args:
        name: Tokenizer name or HuggingFace repo (default: SimpleStories-5M)
        
    Returns:
        HuggingFace tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer
