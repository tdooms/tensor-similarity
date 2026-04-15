"""Analyze induction head behavior on specific sequence formats.

Tests 4 different sequence formats:
1. ABCABC - Simple repeat
2. ABCDAB - Partial repeat at end
3. ABABAB - Alternating pattern
4. ABCDBC - Middle subsequence repeat
"""

import torch
import yaml
from pathlib import Path
import argparse
import numpy as np

from models import AttentionLM
from models.attention_kernels.bilinear import BilinearAttention, QuadraticAttention


class AttentionCapture:
    """Capture attention patterns from attention layers using hooks."""
    
    def __init__(self, model, attn_type: str):
        self.model = model
        self.attn_type = attn_type
        self.patterns = {}
        self.hooks = []
        
    def register_hooks(self):
        """Register forward hooks on all attention layers."""
        for layer_idx, layer in enumerate(self.model.layers):
            if isinstance(layer, (BilinearAttention, QuadraticAttention)):
                hook = layer.register_forward_hook(
                    self._make_hook(layer_idx)
                )
                self.hooks.append(hook)
    
    def _make_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            x = input[0]
            _, debug = module.forward(x, return_debug=True)
            
            if isinstance(module, BilinearAttention):
                pattern = debug['pattern']
                for head_idx in range(module.n_head):
                    pattern_h = pattern[:, head_idx, :, :]
                    self.patterns[(layer_idx, head_idx, 'combined')] = pattern_h.detach().cpu()
            
            elif isinstance(module, QuadraticAttention):
                pattern = debug['pattern']
                for head_idx in range(module.n_head):
                    pattern_h = pattern[:, head_idx, :, :]
                    self.patterns[(layer_idx, head_idx, 'combined')] = pattern_h.detach().cpu()
        
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


def generate_format_sequences(format_type: str, vocab_size: int, n_ctx: int, n_samples: int = 100):
    """Generate sequences of a specific format.
    
    Args:
        format_type: One of 'ABCABC', 'ABCDAB', 'ABABAB', 'ABCDBC'
        vocab_size: Size of vocabulary
        n_ctx: Context length
        n_samples: Number of samples to generate
    
    Returns:
        tokens: (n_samples, n_ctx) tensor
        description: String describing the format
    """
    tokens = torch.zeros((n_samples, n_ctx), dtype=torch.long)
    
    if format_type == 'ABCABC':
        # Simple repeat: [ABC][ABC]
        seq_len = n_ctx // 2
        desc = f"Simple repeat [ABC][ABC] with seq_len={seq_len}"
        
        for i in range(n_samples):
            # Generate unique tokens for subsequence
            subseq = torch.randperm(vocab_size)[:seq_len]
            tokens[i, :seq_len] = subseq
            tokens[i, seq_len:2*seq_len] = subseq
            
            # Fill any remaining with unique tokens
            if 2*seq_len < n_ctx:
                remaining = n_ctx - 2*seq_len
                used = set(subseq.tolist())
                available = [t for t in range(vocab_size) if t not in used]
                tokens[i, 2*seq_len:] = torch.tensor(available[:remaining])
    
    elif format_type == 'ABCDAB':
        # Partial repeat at end: [ABCD][AB]
        full_len = n_ctx // 2
        partial_len = n_ctx // 4
        desc = f"Partial repeat [ABCD][AB] with full={full_len}, partial={partial_len}"
        
        for i in range(n_samples):
            # Generate unique tokens
            subseq = torch.randperm(vocab_size)[:full_len]
            tokens[i, :full_len] = subseq
            tokens[i, full_len:full_len+partial_len] = subseq[:partial_len]
            
            # Fill remaining with unique tokens
            if full_len + partial_len < n_ctx:
                remaining = n_ctx - full_len - partial_len
                used = set(subseq.tolist())
                available = [t for t in range(vocab_size) if t not in used]
                tokens[i, full_len+partial_len:] = torch.tensor(available[:remaining])
    
    elif format_type == 'ABABAB':
        # Alternating pattern: [AB][AB][AB]
        pair_len = 2
        n_repeats = n_ctx // pair_len
        desc = f"Alternating [AB][AB][AB] with {n_repeats} repeats"
        
        for i in range(n_samples):
            # Generate unique pair
            pair = torch.randperm(vocab_size)[:pair_len]
            for j in range(n_repeats):
                tokens[i, j*pair_len:(j+1)*pair_len] = pair
            
            # Fill any remaining
            if n_repeats * pair_len < n_ctx:
                remaining = n_ctx - n_repeats * pair_len
                used = set(pair.tolist())
                available = [t for t in range(vocab_size) if t not in used]
                tokens[i, n_repeats*pair_len:] = torch.tensor(available[:remaining])
    
    elif format_type == 'ABCDBC':
        # Middle subsequence repeat: [ABCD][BC]
        full_len = n_ctx // 2
        middle_start = full_len // 4
        middle_len = full_len // 2
        desc = f"Middle repeat [ABCD][BC] with full={full_len}, middle={middle_len}"
        
        for i in range(n_samples):
            # Generate unique tokens for full sequence
            subseq = torch.randperm(vocab_size)[:full_len]
            tokens[i, :full_len] = subseq
            
            # Repeat middle portion
            tokens[i, full_len:full_len+middle_len] = subseq[middle_start:middle_start+middle_len]
            
            # Fill remaining with unique tokens
            if full_len + middle_len < n_ctx:
                remaining = n_ctx - full_len - middle_len
                used = set(subseq.tolist())
                available = [t for t in range(vocab_size) if t not in used]
                tokens[i, full_len+middle_len:] = torch.tensor(available[:remaining])
    
    else:
        raise ValueError(f"Unknown format type: {format_type}")
    
    return tokens, desc


def compute_induction_score_for_format(pattern: torch.Tensor, format_type: str, n_ctx: int) -> float:
    """Compute induction score for a specific format.
    
    Args:
        pattern: (batch, n_ctx, n_ctx) attention pattern
        format_type: Type of sequence format
        n_ctx: Context length
    
    Returns:
        Average induction score
    """
    batch = pattern.shape[0]
    score = 0.0
    count = 0
    
    if format_type == 'ABCABC':
        # For [ABC][ABC], check attention from second half to token after match in first half
        seq_len = n_ctx // 2
        for b in range(batch):
            for q in range(seq_len, min(2*seq_len, n_ctx) - 1):
                k_match = q - seq_len
                k_induct = k_match + 1
                if k_induct < seq_len:
                    score += pattern[b, q, k_induct].item()
                    count += 1
    
    elif format_type == 'ABCDAB':
        # For [ABCD][AB], check attention from partial repeat to token after match
        full_len = n_ctx // 2
        partial_len = n_ctx // 4
        for b in range(batch):
            for q in range(full_len, min(full_len + partial_len, n_ctx) - 1):
                k_match = q - full_len
                k_induct = k_match + 1
                if k_induct < full_len:
                    score += pattern[b, q, k_induct].item()
                    count += 1
    
    elif format_type == 'ABABAB':
        # For [AB][AB][AB], check attention from each repeat to next token after previous
        pair_len = 2
        n_repeats = n_ctx // pair_len
        for b in range(batch):
            for repeat_idx in range(1, n_repeats):
                for offset in range(pair_len):
                    q = repeat_idx * pair_len + offset
                    if q >= n_ctx - 1:
                        continue
                    # Previous occurrence
                    k_match = (repeat_idx - 1) * pair_len + offset
                    k_induct = k_match + 1
                    if k_induct < n_ctx:
                        score += pattern[b, q, k_induct].item()
                        count += 1
    
    elif format_type == 'ABCDBC':
        # For [ABCD][BC], check attention from middle repeat to token after match
        full_len = n_ctx // 2
        middle_start = full_len // 4
        middle_len = full_len // 2
        for b in range(batch):
            for q in range(full_len, min(full_len + middle_len, n_ctx) - 1):
                offset = q - full_len
                k_match = middle_start + offset
                k_induct = k_match + 1
                if k_induct < full_len:
                    score += pattern[b, q, k_induct].item()
                    count += 1
    
    return score / count if count > 0 else 0.0


def analyze_format(model, attn_type: str, format_type: str, vocab_size: int, n_ctx: int, 
                   n_samples: int = 100):
    """Analyze induction behavior on a specific sequence format.
    
    Args:
        model: The attention model
        attn_type: 'bilinear' or 'quadratic'
        format_type: Sequence format to test
        vocab_size: Vocabulary size
        n_ctx: Context length
        n_samples: Number of samples
    
    Returns:
        Dictionary of metrics per head
    """
    # Generate sequences
    tokens, desc = generate_format_sequences(format_type, vocab_size, n_ctx, n_samples)
    
    print(f"\n{'=' * 80}")
    print(f"FORMAT: {format_type}")
    print(f"Description: {desc}")
    print(f"Samples: {n_samples}")
    print("=" * 80)
    
    # Show examples
    print("\nExample sequences:")
    for i in range(min(3, n_samples)):
        print(f"  Sample {i}: {tokens[i].tolist()}")
    
    # Capture attention patterns
    capture = AttentionCapture(model, attn_type)
    capture.register_hooks()
    
    with torch.no_grad():
        _ = model(tokens)
    
    capture.remove_hooks()
    
    # Compute induction scores for each head
    results = {}
    for (layer_idx, head_idx, circuit), pattern in capture.patterns.items():
        if circuit != 'combined':
            continue
        
        induction_score = compute_induction_score_for_format(pattern, format_type, n_ctx)
        results[(layer_idx, head_idx)] = induction_score
    
    # Print results
    print(f"\nInduction scores by head:")
    print(f"{'Layer':<8} {'Head':<6} {'Induction Score':<15}")
    print("-" * 40)
    
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for (layer_idx, head_idx), score in sorted_results:
        print(f"{layer_idx:<8} {head_idx:<6} {score:>12.4f}")
    
    # Identify top heads
    if sorted_results:
        top_head = sorted_results[0]
        print(f"\nStrongest induction head: Layer {top_head[0][0]}, Head {top_head[0][1]}")
        print(f"  Score: {top_head[1]:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Analyze induction heads on specific sequence formats')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint and config')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of samples per format')
    
    args = parser.parse_args()
    
    # Find checkpoint and config
    checkpoint_dir = Path(__file__).parent / args.checkpoint_dir
    
    # Check if this is a run directory or checkpoints subdirectory
    if checkpoint_dir.name == 'checkpoints':
        run_dir = checkpoint_dir.parent
    else:
        run_dir = checkpoint_dir
        checkpoint_dir = checkpoint_dir / 'checkpoints'
    
    # Look for final.pt or latest checkpoint
    if (run_dir / 'final.pt').exists():
        checkpoint_path = run_dir / 'final.pt'
    elif (checkpoint_dir / 'final.pt').exists():
        checkpoint_path = checkpoint_dir / 'final.pt'
    else:
        checkpoints = list(checkpoint_dir.glob('step_*.pt'))
        if not checkpoints:
            print(f"Error: No checkpoints found in {checkpoint_dir}")
            return
        checkpoint_path = max(checkpoints, key=lambda p: int(p.stem.split('_')[1]))
    
    # Find config
    config_path = run_dir / 'config.yaml'
    if not config_path.exists():
        print(f"Error: Config not found at {config_path}")
        return
    
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("=" * 80)
    print("INDUCTION HEAD ANALYSIS ON SEQUENCE FORMATS")
    print("=" * 80)
    print(f"\nConfig: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load model
    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel loaded successfully")
    print(f"  Layers: {cfg['model']['n_layers']}")
    print(f"  Heads per layer: {cfg['model']['n_head']}")
    print(f"  Total heads: {cfg['model']['n_layers'] * cfg['model']['n_head']}")
    print(f"  Context: {cfg['model']['n_ctx']}")
    print(f"  Attention type: {cfg['model']['attn_type']}")
    
    vocab_size = cfg['model']['vocab_size']
    n_ctx = cfg['model']['n_ctx']
    attn_type = cfg['model']['attn_type']
    
    # Test all formats
    formats = ['ABCABC', 'ABCDAB', 'ABABAB', 'ABCDBC']
    all_results = {}
    
    for format_type in formats:
        results = analyze_format(model, attn_type, format_type, vocab_size, n_ctx, args.n_samples)
        all_results[format_type] = results
    
    # Summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY: Top Induction Heads Across Formats")
    print("=" * 80)
    
    for format_type in formats:
        results = all_results[format_type]
        if results:
            top_head = max(results.items(), key=lambda x: x[1])
            print(f"\n{format_type}:")
            print(f"  Best head: Layer {top_head[0][0]}, Head {top_head[0][1]}")
            print(f"  Score: {top_head[1]:.4f}")


if __name__ == "__main__":
    main()
