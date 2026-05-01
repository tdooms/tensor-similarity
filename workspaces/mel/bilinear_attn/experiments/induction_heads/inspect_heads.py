#!/usr/bin/env python3
"""Inspect attention heads using PyTorch hooks on repeated sequences.

Architecture-agnostic version that works with:
- BilinearAttention or QuadraticAttention
- Any number of layers
- With or without bias
- Variable-length repeated sequences (to prevent RoPE shortcuts)

Based on Anthropic's "In-context Learning and Induction Heads" paper:
https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
"""

import sys
from pathlib import Path
import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple, Optional
import argparse

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models import AttentionLM
from models.attention_kernels.bilinear import BilinearAttention, QuadraticAttention
from experiments.induction_heads.analyze_sequence_formats import (
    compute_induction_score_for_format,
    generate_format_sequences,
)


FORMAT_CHOICES = ("ABCABC", "ABCDAB", "ABABAB", "ABCDBC")


class AttentionCapture:
    """Capture attention patterns from attention layers using hooks."""
    
    def __init__(self, model: AttentionLM, attn_type: str):
        self.model = model
        self.attn_type = attn_type  # 'bilinear' or 'quadratic'
        self.patterns = {}  # {(layer_idx, head_idx, circuit): pattern}
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
            # Use return_debug to get all intermediate values
            x = input[0]
            _, debug = module.forward(x, return_debug=True)
            
            if isinstance(module, BilinearAttention):
                # BilinearAttention: q1k1, q2k2, combined
                scores1 = debug['scores1']  # (batch, n_head, n_ctx, n_ctx)
                scores2 = debug['scores2']
                pattern = debug['pattern']  # combined with mask
                
                for head_idx in range(module.n_head):
                    pattern1_h = scores1[:, head_idx, :, :]
                    pattern2_h = scores2[:, head_idx, :, :]
                    pattern_combined_h = pattern[:, head_idx, :, :]
                    
                    self.patterns[(layer_idx, head_idx, 'q1k1')] = pattern1_h.detach().cpu()
                    self.patterns[(layer_idx, head_idx, 'q2k2')] = pattern2_h.detach().cpu()
                    self.patterns[(layer_idx, head_idx, 'combined')] = pattern_combined_h.detach().cpu()
            
            elif isinstance(module, QuadraticAttention):
                # QuadraticAttention: single scores, pattern
                scores = debug['scores']  # (batch, n_head, n_ctx, n_ctx)
                pattern = debug['pattern']  # squared and masked
                
                for head_idx in range(module.n_head):
                    scores_h = scores[:, head_idx, :, :]
                    pattern_h = pattern[:, head_idx, :, :]
                    
                    self.patterns[(layer_idx, head_idx, 'scores')] = scores_h.detach().cpu()
                    self.patterns[(layer_idx, head_idx, 'combined')] = pattern_h.detach().cpu()
        
        return hook
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def get_pattern(self, layer_idx: int, head_idx: int, circuit: str = 'combined') -> torch.Tensor:
        """Get attention pattern for a specific head and circuit."""
        return self.patterns.get((layer_idx, head_idx, circuit))


def generate_fixed_length_sequences(
    vocab_size: int,
    n_ctx: int,
    n_samples: int = 16,
    bos_token_id: int | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate fixed-length repeated sequences for inspection.

    Produces a ``[BOS?][subseq][subseq][pad?]`` layout where ``subseq`` has
    length ``(n_ctx - bos_offset) // 2``. The first ``bos_offset`` positions
    are reserved for a BOS token (id=``bos_token_id``) when provided. The
    subseq tokens are drawn from the content vocab (excluding BOS) so that
    the pattern visible to the model is identical to training.

    Args:
        vocab_size: Size of vocabulary (includes BOS if used).
        n_ctx: Full context length.
        n_samples: Number of sequences to generate.
        bos_token_id: If set, reserve position 0 for this id.

    Returns:
        tokens: (n_samples, n_ctx) sequences.
        repeat_masks: (n_samples, n_ctx) bool masks for second-half repeats
            (excluding the first token of the repeat).
    """
    bos_offset = 1 if bos_token_id is not None else 0
    content_len = n_ctx - bos_offset
    seq_len = content_len // 2
    tokens = torch.zeros((n_samples, n_ctx), dtype=torch.long)
    repeat_masks = torch.zeros((n_samples, n_ctx), dtype=torch.bool)

    content_vocab = vocab_size - bos_offset
    assert seq_len <= content_vocab, (
        f"seq_len={seq_len} exceeds content vocab={content_vocab} "
        f"(vocab_size={vocab_size}, bos={bos_token_id})"
    )

    for i in range(n_samples):
        # Sample subsequence from content vocab (excluding BOS if reserved).
        subseq = torch.randint(0, content_vocab, (seq_len,))
        if bos_token_id is not None:
            subseq = torch.where(subseq >= bos_token_id, subseq + 1, subseq)
        if bos_offset:
            tokens[i, 0] = bos_token_id
        tokens[i, bos_offset:bos_offset + seq_len] = subseq
        tokens[i, bos_offset + seq_len:bos_offset + 2 * seq_len] = subseq

        # Mark second-half repeat (excluding its first token).
        start = bos_offset + seq_len + 1
        end = bos_offset + 2 * seq_len
        repeat_masks[i, start:end] = True

    return tokens, repeat_masks


def compute_prefix_matching_score(pattern: torch.Tensor, repeat_masks: torch.Tensor,
                                 tokens: torch.Tensor, bos_offset: int = 0) -> float:
    """Compute prefix matching score for fixed-length [BOS?][seq][seq] sequences.

    For each position in the second half, measure attention to the matching
    position in the first half. ``bos_offset`` shifts both halves.
    """
    batch, n_ctx, _ = pattern.shape
    content_len = n_ctx - bos_offset
    seq_len = content_len // 2
    score = 0.0
    count = 0

    for b in range(batch):
        for q in range(bos_offset + seq_len, bos_offset + 2 * seq_len):
            k = q - seq_len  # matching position in first half
            score += pattern[b, q, k].item()
            count += 1

    return score / count if count > 0 else 0.0


def compute_induction_score(pattern: torch.Tensor, repeat_masks: torch.Tensor,
                           tokens: torch.Tensor, bos_offset: int = 0) -> float:
    """Compute induction score: attention to token AFTER previous occurrence.

    For fixed ``[BOS?][seq][seq]`` format, this is attention to position
    k+1 where k is the matching position in the first half.
    """
    batch, n_ctx, _ = pattern.shape
    content_len = n_ctx - bos_offset
    seq_len = content_len // 2
    score = 0.0
    count = 0

    for b in range(batch):
        for q in range(bos_offset + seq_len, bos_offset + 2 * seq_len - 1):
            k_match = q - seq_len
            k_induct = k_match + 1
            if k_induct < bos_offset + seq_len:
                score += pattern[b, q, k_induct].item()
                count += 1

    return score / count if count > 0 else 0.0


def compute_copying_score(pattern: torch.Tensor) -> float:
    """Compute copying score: attention to previous token."""
    batch, n_ctx, _ = pattern.shape
    score = 0.0
    count = 0
    
    for q in range(1, n_ctx):
        score += pattern[:, q, q - 1].mean().item()
        count += 1
    
    return score / count if count > 0 else 0.0


def analyze_all_heads(patterns: Dict, repeat_masks: torch.Tensor, tokens: torch.Tensor,
                      bos_offset: int = 0) -> Dict:
    """Compute all metrics for all heads."""
    results = {}

    for (layer_idx, head_idx, circuit), pattern in patterns.items():
        if circuit != 'combined':
            continue

        key = (layer_idx, head_idx)

        results[key] = {
            'prefix_matching': compute_prefix_matching_score(pattern, repeat_masks, tokens, bos_offset),
            'induction': compute_induction_score(pattern, repeat_masks, tokens, bos_offset),
            'copying': compute_copying_score(pattern),
        }

    return results


def analyze_format_heads(patterns: Dict, format_type: str, n_ctx: int,
                         bos_offset: int = 0) -> Dict:
    """Compute format-specific induction scores for all heads."""
    results = {}

    for (layer_idx, head_idx, circuit), pattern in patterns.items():
        if circuit != 'combined':
            continue

        results[(layer_idx, head_idx)] = {
            'induction': compute_induction_score_for_format(
                pattern, format_type, n_ctx, bos_offset=bos_offset,
            ),
            'copying': compute_copying_score(pattern),
        }

    return results


def visualize_attention_heads(patterns: Dict, n_layers: int, n_heads: int,
                              save_path: Path = None, step: Optional[int] = None,
                              format_type: Optional[str] = None):
    """Visualize attention patterns for all heads."""
    fig, axes = plt.subplots(n_layers, n_heads, figsize=(6 * n_heads, 6 * n_layers))
    if n_layers == 1 and n_heads == 1:
        axes = np.array([[axes]])
    elif n_layers == 1:
        axes = axes.reshape(1, -1)
    elif n_heads == 1:
        axes = axes.reshape(-1, 1)
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            pattern = patterns.get((layer_idx, head_idx, 'combined'))
            
            if pattern is None:
                continue
            
            # Average over batch
            pattern_avg = pattern.mean(dim=0).numpy()
            
            ax = axes[layer_idx, head_idx]
            im = ax.imshow(pattern_avg, cmap='viridis', aspect='auto', interpolation='nearest')
            ax.set_title(f'Layer {layer_idx}, Head {head_idx}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.grid(False)
    
    title = 'Attention Patterns'
    if format_type is not None:
        title += f' - {format_type}'
    if step is not None:
        title += f' - Step {step}'
    fig.suptitle(title, fontsize=16, y=0.995)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.close()


def inspect_tokens(model, cfg: Dict, tokens: torch.Tensor) -> Dict:
    """Run model once on tokens and return captured attention patterns."""
    capture = AttentionCapture(model, cfg['model']['attn_type'])
    capture.register_hooks()

    with torch.no_grad():
        _ = model(tokens)

    capture.remove_hooks()
    return capture.patterns


def print_metric_table(metrics: Dict, include_prefix: bool = True) -> None:
    if include_prefix:
        print(f"{'Layer':<8} {'Head':<6} {'Prefix':<10} {'Induction':<12} {'Copying':<10}")
        print("-" * 70)
        for (layer_idx, head_idx), scores in sorted(metrics.items()):
            print(f"{layer_idx:<8} {head_idx:<6} {scores['prefix_matching']:>8.4f}   "
                  f"{scores['induction']:>10.4f}   {scores['copying']:>8.4f}")
    else:
        print(f"{'Layer':<8} {'Head':<6} {'Induction':<12} {'Copying':<10}")
        print("-" * 55)
        for (layer_idx, head_idx), scores in sorted(metrics.items()):
            print(f"{layer_idx:<8} {head_idx:<6} {scores['induction']:>10.4f}   "
                  f"{scores['copying']:>8.4f}")

    if metrics:
        best_head = max(metrics.items(), key=lambda x: x[1]['induction'])
        print(f"\nStrongest induction head: Layer {best_head[0][0]}, Head {best_head[0][1]}")
        print(f"  Induction score: {best_head[1]['induction']:.4f}")


def inspect_one_format(model, cfg: Dict, format_type: str, n_samples: int,
                       output_dir: Path, step_label, bos_token_id: int | None,
                       bos_offset: int) -> None:
    """Inspect attention heads on one named sequence format."""
    n_ctx = cfg['model']['n_ctx']
    tokens, desc = generate_format_sequences(
        format_type,
        cfg['model']['vocab_size'],
        n_ctx,
        n_samples,
        bos_token_id=bos_token_id,
    )

    print(f"\n{'=' * 70}")
    print(f"INPUT FORMAT: {format_type}")
    print(f"{'=' * 70}")
    print(f"Description: {desc}")
    print(f"Samples: {n_samples}")

    print(f"\nExample sequences:")
    for i in range(min(3, n_samples)):
        print(f"  Sample {i}: {tokens[i].tolist()}")

    print(f"\nCapturing attention patterns...")
    patterns = inspect_tokens(model, cfg, tokens)
    n_heads = cfg['model']['n_layers'] * cfg['model']['n_head']
    print(f"Captured patterns for {n_heads} heads")

    print(f"\n{'=' * 70}")
    print("FORMAT-SPECIFIC INDUCTION METRICS")
    print("=" * 70)
    print("  - Induction: attention to token AFTER previous matching occurrence")
    print("  - Copying: attention to immediately previous token")
    print()

    metrics = analyze_format_heads(patterns, format_type, n_ctx, bos_offset)
    print_metric_table(metrics, include_prefix=False)

    format_dir = output_dir / format_type
    format_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nVisualizing attention patterns...")
    save_path = format_dir / f"attention_heads_{step_label}.png"
    visualize_attention_heads(
        patterns,
        n_layers=cfg['model']['n_layers'],
        n_heads=cfg['model']['n_head'],
        save_path=save_path,
        step=None if step_label == "final" else int(step_label),
        format_type=format_type,
    )


def main():
    parser = argparse.ArgumentParser(description='Inspect attention heads on repeated sequences')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoints (will auto-detect config)')
    parser.add_argument('--step', type=int, default=None,
                       help='Checkpoint step to load (default: use final.pt)')
    parser.add_argument('--n_samples', type=int, default=32,
                       help='Number of sequences to generate')
    parser.add_argument('--output_dir', type=str, default='attention_visualizations',
                       help='Directory to save visualizations')
    parser.add_argument('--format', type=str, default='all',
                       choices=('all', *FORMAT_CHOICES),
                       help='Input sequence format to inspect, or all formats')
    
    args = parser.parse_args()
    
    # Find checkpoint directory and config
    checkpoint_dir = Path(__file__).parent / args.checkpoint_dir
    
    # Look for config.yaml in checkpoint_dir or parent
    config_path = checkpoint_dir.parent / "config.yaml"
    if not config_path.exists():
        config_path = checkpoint_dir / "config.yaml"
    
    if not config_path.exists():
        print(f"Error: Could not find config.yaml in {checkpoint_dir} or {checkpoint_dir.parent}")
        return
    
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    print("=" * 70)
    print("ATTENTION HEAD INSPECTION (Architecture-Agnostic)")
    print("=" * 70)
    print(f"\nConfig: {config_path}")
    print(f"\nModel architecture:")
    print(f"  Layers: {cfg['model']['n_layers']}")
    print(f"  Heads per layer: {cfg['model']['n_head']}")
    print(f"  Total heads: {cfg['model']['n_layers'] * cfg['model']['n_head']}")
    print(f"  Context length: {cfg['model']['n_ctx']}")
    print(f"  Attention type: {cfg['model']['attn_type']}")
    
    if cfg['model']['attn_type'] == 'bilinear':
        print(f"  Circuits per head: 2 (Q1-K1, Q2-K2)")
        print(f"  Total circuits: {cfg['model']['n_layers'] * cfg['model']['n_head'] * 2}")
    else:
        print(f"  Circuits per head: 1 (Q-K)")
    
    # Load checkpoint
    if args.step is not None:
        checkpoint_path = checkpoint_dir / f"step_{args.step}.pt"
        step_label = args.step
    else:
        checkpoint_path = checkpoint_dir.parent / "final.pt"
        step_label = "final"
    
    if not checkpoint_path.exists():
        print(f"\nError: Checkpoint not found at {checkpoint_path}")
        print(f"Available checkpoints:")
        for ckpt in sorted(checkpoint_dir.glob("*.pt")):
            print(f"  {ckpt.name}")
        return
    
    print(f"\nLoading checkpoint: {checkpoint_path}")
    
    # Create model
    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded successfully")
    
    # Resolve BOS from config.
    data_cfg = cfg.get('data', {})
    bos_token_id = None
    if data_cfg.get('use_bos', False):
        bos_token_id = data_cfg.get('bos_token_id', cfg['model']['vocab_size'] - 1)
    bos_offset = 1 if bos_token_id is not None else 0

    n_ctx = cfg['model']['n_ctx']
    output_dir = Path(__file__).parent / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    formats = FORMAT_CHOICES if args.format == 'all' else (args.format,)
    print(f"\nInspecting input formats: {', '.join(formats)}")
    if bos_token_id is not None:
        print(f"  BOS token id={bos_token_id} (position 0 in every sequence)")

    for format_type in formats:
        inspect_one_format(
            model=model,
            cfg=cfg,
            format_type=format_type,
            n_samples=args.n_samples,
            output_dir=output_dir,
            step_label=step_label,
            bos_token_id=bos_token_id,
            bos_offset=bos_offset,
        )
    
    print(f"\n{'=' * 70}")
    print("DONE")
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
