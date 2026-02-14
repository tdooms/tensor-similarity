"""Logan's local copy of model/training infrastructure.

Copied from workspaces/mel/bilinear_attn/ to decouple from mel's directory.
"""
from .model import AttentionLM, ATTN_REGISTRY
from .data import create_dataloaders, get_tokenizer
from .trainer import Trainer
from .losses import compute_loss, next_token_ce, per_position_ce
from .optim import create_optimizer, create_scheduler, Optimizers, _is_muon_param
from .eval import evaluate
