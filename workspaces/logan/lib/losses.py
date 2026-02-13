import torch
import torch.nn.functional as F


def next_token_ce(logits, input_ids, label_smoothing=0.0):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    return F.cross_entropy(shift_logits.view(B * T, V), shift_labels.view(B * T),
                           label_smoothing=label_smoothing)


def per_position_ce(logits, input_ids):
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = input_ids[:, 1:].contiguous()
    B, T, V = shift_logits.shape
    loss = F.cross_entropy(shift_logits.view(B * T, V), shift_labels.view(B * T), reduction="none")
    return loss.view(B, T).mean(dim=0)


def compute_loss(logits, input_ids, loss_type="next_token_ce", label_smoothing=0.0, **kwargs):
    if loss_type == "next_token_ce":
        return next_token_ce(logits, input_ids, label_smoothing)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
