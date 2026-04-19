"""AttentionLM as a Model for TN similarity.

This module provides a Component-compatible wrapper around AttentionLM
that implements the Model.components() interface required by the main
codebase's TN similarity algorithm.
"""

import torch
from torch import nn

from src.models.base import Model
from .embedding import EmbeddingComponent, UnembeddingComponent
from .attention import BilinearAttentionComponent, QuadraticAttentionComponent


class AttentionLMComponent(Model):
    """Component-compatible version of AttentionLM for TN similarity.
    
    This wraps the mel workspace's AttentionLM to provide the Model interface
    required by the main codebase's TN similarity algorithm.
    
    Architecture:
        tokens → Embed → [Attention + Residual] × n_layers → Unembed → logits
    
    Limitations:
        - Only supports norm_type='none' and norm_places=[] (no normalization)
        - Only supports 'bilinear' and 'quadratic' attention types
        - Does not support use_rmsnorm_qk=True
    """
    
    def __init__(
        self,
        vocab_size: int,
        n_ctx: int,
        d_model: int,
        n_head: int,
        n_layers: int,
        attn_scale: float = 0.2,
        attn_type: str = "bilinear",
        use_bias_qk: bool = True,
        rope_base: int = 10000,
    ) -> None:
        super().__init__(None)
        self.vocab_size = vocab_size
        self.n_ctx = n_ctx
        self.d_model = d_model
        self.n_head = n_head
        self.n_layers = n_layers
        self.attn_type = attn_type
        
        # Embedding layer
        self.embed = EmbeddingComponent(vocab_size, d_model)
        
        # Attention layers
        if attn_type == "bilinear":
            self.layers = nn.ModuleList([
                BilinearAttentionComponent(
                    d_model=d_model,
                    n_head=n_head,
                    n_ctx=n_ctx,
                    scale=attn_scale,
                    bias=use_bias_qk,
                    rope_base=rope_base,
                )
                for _ in range(n_layers)
            ])
        elif attn_type == "quadratic":
            self.layers = nn.ModuleList([
                QuadraticAttentionComponent(
                    d_model=d_model,
                    n_head=n_head,
                    n_ctx=n_ctx,
                    scale=attn_scale,
                    bias=use_bias_qk,
                    rope_base=rope_base,
                )
                for _ in range(n_layers)
            ])
        else:
            raise ValueError(
                f"Unsupported attention type for TN similarity: {attn_type!r}. "
                f"Only 'bilinear' and 'quadratic' are supported."
            )
        
        # Unembedding layer
        self.unembed = UnembeddingComponent(d_model, vocab_size)
    
    def components(self):
        """Returns list of components for TN similarity computation.
        
        Order: [embed, layer0, layer1, ..., unembed]
        """
        return [self.embed] + list(self.layers) + [self.unembed]
    
    def forward(self, x):
        """Forward pass (not used for TN similarity, but provided for completeness)."""
        raise NotImplementedError(
            "AttentionLMComponent.forward() is not implemented. "
            "Use the original AttentionLM for inference."
        )
    
    @classmethod
    def from_trained_model(
        cls,
        model,
        rope_base: int = None,
        *,
        ignore_norms: bool = False,
    ) -> "AttentionLMComponent":
        """Create from a trained AttentionLM model.
        
        Args:
            model: Trained AttentionLM from mel workspace
            rope_base: RoPE base frequency (default: use model's value if available)
            ignore_norms: If True, skip validation of norm-related fields
                (``norm_type``, ``norm_places``, ``embed_norm``, ``final_norm``,
                ``layer_norms``, ``use_rmsnorm_qk``) and do not copy any norm
                state into the component. Intended for the "trained-with-norm,
                measure-ignoring-norm" workflow: the resulting component
                behaves as if every norm were the identity. The caller is
                responsible for understanding that the TN-similarity value
                reflects the linear sub-network only.
            
        Returns:
            AttentionLMComponent with weights copied from the model
            
        Raises:
            ValueError: If model has unsupported configuration for TN similarity
        """
        # Validate model configuration
        _validate_model_for_tn_similarity(model, ignore_norms=ignore_norms)
        
        # Get rope_base from model if not provided
        if rope_base is None:
            if hasattr(model.layers[0], 'rotary') and hasattr(model.layers[0].rotary, 'base'):
                rope_base = model.layers[0].rotary.base
            else:
                rope_base = 10000  # Default
        
        # Create component model
        component = cls(
            vocab_size=model.vocab_size,
            n_ctx=model.n_ctx,
            d_model=model.d_model,
            n_head=model.n_head,
            n_layers=model.n_layers,
            attn_scale=model.layers[0].scale,
            attn_type=model.attn_type,
            use_bias_qk=model.layers[0].q1.bias is not None if hasattr(model.layers[0], 'q1') else model.layers[0].q.bias is not None,
            rope_base=rope_base,
        )
        
        # Copy embedding weights
        component.embed.weight.data.copy_(model.embed.weight.data)
        
        # Copy attention layer weights
        for comp_layer, model_layer in zip(component.layers, model.layers):
            if model.attn_type == "bilinear":
                comp_layer.q1.weight.data.copy_(model_layer.q1.weight.data)
                comp_layer.k1.weight.data.copy_(model_layer.k1.weight.data)
                comp_layer.q2.weight.data.copy_(model_layer.q2.weight.data)
                comp_layer.k2.weight.data.copy_(model_layer.k2.weight.data)
                if model_layer.q1.bias is not None:
                    comp_layer.q1.bias.data.copy_(model_layer.q1.bias.data)
                    comp_layer.k1.bias.data.copy_(model_layer.k1.bias.data)
                    comp_layer.q2.bias.data.copy_(model_layer.q2.bias.data)
                    comp_layer.k2.bias.data.copy_(model_layer.k2.bias.data)
            else:  # quadratic
                comp_layer.q.weight.data.copy_(model_layer.q.weight.data)
                comp_layer.k.weight.data.copy_(model_layer.k.weight.data)
                if model_layer.q.bias is not None:
                    comp_layer.q.bias.data.copy_(model_layer.q.bias.data)
                    comp_layer.k.bias.data.copy_(model_layer.k.bias.data)
            
            comp_layer.v.weight.data.copy_(model_layer.v.weight.data)
            comp_layer.o.weight.data.copy_(model_layer.o.weight.data)
        
        # Copy unembedding weights
        component.unembed.weight.data.copy_(model.unembed.weight.data)
        
        return component


def _validate_model_for_tn_similarity(model, *, ignore_norms: bool = False):
    """Validate that a model is compatible with TN similarity computation.
    
    Args:
        model: AttentionLM model to validate
        ignore_norms: If True, skip all norm-related checks (``norm_type``,
            ``norm_places``, ``embed_norm``, ``final_norm``, ``layer_norms``,
            ``use_rmsnorm_qk``). The model's norm modules will be treated as
            identities by the caller. Non-norm checks (e.g. ``attn_type``)
            still run.
        
    Raises:
        ValueError: If model has unsupported configuration
    """
    errors = []
    
    if not ignore_norms:
        # Check normalization
        if hasattr(model, 'norm_type') and model.norm_type != 'none':
            errors.append(
                f"norm_type must be 'none' for TN similarity, got {model.norm_type!r}"
            )
        
        if hasattr(model, 'norm_places') and model.norm_places:
            errors.append(
                f"norm_places must be empty for TN similarity, got {model.norm_places}"
            )
        
        # Check for embed_norm
        if hasattr(model, 'embed_norm') and model.embed_norm is not None:
            errors.append(
                "embed_norm must be None for TN similarity"
            )
        
        # Check for final_norm (must be Identity)
        if hasattr(model, 'final_norm') and not isinstance(model.final_norm, nn.Identity):
            errors.append(
                f"final_norm must be Identity for TN similarity, got {type(model.final_norm).__name__}"
            )
        
        # Check for layer_norms
        if hasattr(model, 'layer_norms') and model.layer_norms is not None:
            errors.append(
                "layer_norms must be None for TN similarity"
            )
    
    # Check attention type
    if hasattr(model, 'attn_type') and model.attn_type not in ('bilinear', 'quadratic'):
        errors.append(
            f"attn_type must be 'bilinear' or 'quadratic' for TN similarity, got {model.attn_type!r}"
        )
    
    # Check for RMSNorm on Q/K
    if not ignore_norms:
        for i, layer in enumerate(model.layers):
            if hasattr(layer, 'norm_qk') and not isinstance(layer.norm_qk, nn.Identity):
                errors.append(
                    f"Layer {i} has use_rmsnorm_qk=True, which is not supported for TN similarity"
                )
    
    if errors:
        raise ValueError(
            "Model configuration is not compatible with TN similarity:\n" +
            "\n".join(f"  - {e}" for e in errors) +
            "\n\nTo use TN similarity, create a model with:\n"
            "  norm_type='none'\n"
            "  norm_places=[]\n"
            "  use_rmsnorm_qk=False\n"
            "  attn_type='bilinear' or 'quadratic'"
        )
