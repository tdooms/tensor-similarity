"""Evaluate model predictions on validation data.

Shows model predictions vs ground truth for repeated sequences,
with repeated portions clearly marked in brackets.
"""

import torch
import yaml
from pathlib import Path
import argparse

from models import AttentionLM
from experiments.induction_heads.data import create_repeated_token_dataloaders


def format_sequence_with_brackets(tokens, repeat_mask):
    """Format sequence with repeated portions in brackets.
    
    Args:
        tokens: List of token ids
        repeat_mask: Boolean mask indicating repeated positions
    
    Returns:
        Formatted string with repeated tokens in brackets
    """
    result = []
    for i, token in enumerate(tokens):
        if repeat_mask[i]:
            result.append(f"[{token}]")
        else:
            result.append(str(token))
    return " ".join(result)


def evaluate_predictions(checkpoint_path: str, config_path: str, n_samples: int = 100):
    """Evaluate model predictions on validation data.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Path to config file
        n_samples: Number of samples to evaluate
    """
    # Load config
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    
    print("=" * 80)
    print("MODEL PREDICTION EVALUATION")
    print("=" * 80)
    print(f"\nConfig: {config_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Evaluating on {n_samples} validation samples")
    
    # Load model
    model = AttentionLM.from_config(cfg)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"\nModel loaded successfully")
    print(f"  Layers: {cfg['model']['n_layers']}")
    print(f"  Heads: {cfg['model']['n_head']}")
    print(f"  Context: {cfg['model']['n_ctx']}")
    print(f"  Vocab: {cfg['model']['vocab_size']}")
    
    # Resolve BOS from config so data matches training distribution.
    data_cfg = cfg.get('data', {})
    bos_token_id = None
    if data_cfg.get('use_bos', False):
        bos_token_id = data_cfg.get('bos_token_id', cfg['model']['vocab_size'] - 1)
    if bos_token_id is not None:
        print(f"  BOS token id={bos_token_id} (position 0 in every sequence)")

    # Create validation dataloader
    _, val_dl = create_repeated_token_dataloaders(
        vocab_size=cfg['model']['vocab_size'],
        n_ctx=cfg['model']['n_ctx'],
        batch_size=1,  # Process one at a time for clarity
        n_train=100,
        n_val=n_samples,
        seed=42,
        bos_token_id=bos_token_id,
    )
    
    print(f"\n{'=' * 80}")
    print("PREDICTIONS vs GROUND TRUTH")
    print("=" * 80)
    print("\nFormat: Repeated tokens shown in [brackets]")
    print("Ground Truth: Original sequence with repeated portions marked")
    print("Predictions:  Model's predictions for repeated positions\n")
    
    total_correct = 0
    total_tokens = 0
    
    with torch.no_grad():
        for idx, batch in enumerate(val_dl):
            if idx >= n_samples:
                break
            
            input_ids = batch["input_ids"][0]  # (T,)
            repeat_mask = batch["repeat_mask"][0]  # (T,)
            
            # Get model predictions
            logits = model(input_ids.unsqueeze(0))  # (1, T, V)
            preds = logits[0].argmax(dim=-1)  # (T,)
            
            # Prepare prediction sequence (only for repeated positions)
            pred_tokens = input_ids.clone()
            for i in range(len(repeat_mask)):
                if repeat_mask[i] and i > 0:
                    pred_tokens[i] = preds[i - 1]  # Prediction at i-1 predicts token at i
            
            # Count accuracy on repeated positions
            eval_mask = repeat_mask.clone()
            eval_mask[0] = False  # Can't predict position 0
            
            if eval_mask.any():
                # Shift for prediction
                shifted_preds = preds[:-1]
                shifted_targets = input_ids[1:]
                shifted_mask = eval_mask[1:]
                
                correct = (shifted_preds == shifted_targets) & shifted_mask
                total_correct += correct.sum().item()
                total_tokens += shifted_mask.sum().item()
                
                sample_acc = correct.sum().item() / max(1, shifted_mask.sum().item())
            else:
                sample_acc = 0.0
            
            # Format output
            ground_truth = format_sequence_with_brackets(input_ids.tolist(), repeat_mask.tolist())
            predictions = format_sequence_with_brackets(pred_tokens.tolist(), repeat_mask.tolist())
            
            print(f"Sample {idx + 1} (Accuracy: {sample_acc:.2%}):")
            print(f"  Ground Truth: {ground_truth}")
            print(f"  Predictions:  {predictions}")
            
            # Show mismatches
            mismatches = []
            for i in range(len(repeat_mask)):
                if repeat_mask[i] and i > 0:
                    if pred_tokens[i] != input_ids[i]:
                        mismatches.append(f"pos {i}: predicted {pred_tokens[i].item()}, expected {input_ids[i].item()}")
            
            if mismatches:
                print(f"  Mismatches: {', '.join(mismatches)}")
            print()
    
    # Overall statistics
    overall_acc = total_correct / max(1, total_tokens)
    print("=" * 80)
    print(f"OVERALL ACCURACY: {overall_acc:.2%} ({total_correct}/{total_tokens} tokens correct)")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Evaluate model predictions on validation data')
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                       help='Directory containing checkpoint and config')
    parser.add_argument('--n_samples', type=int, default=100,
                       help='Number of validation samples to evaluate')
    
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
        # Find latest checkpoint
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
    
    evaluate_predictions(str(checkpoint_path), str(config_path), args.n_samples)


if __name__ == "__main__":
    main()
