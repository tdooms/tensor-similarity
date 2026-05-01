# Head Ablation

This directory contains reusable interventions for the trained induction-head
`AttentionLM` runs.

The pytest suite in `experiments/path_decomp` decomposes a two-layer reference
Transformer into residual, layer-1, and layer-2 path families. For the trained
induction model we do not reuse that reference implementation directly. Instead,
we use the same logical idea of isolating active attention contributions and
intervene on the model's own attention heads.

`head_ablation.py` zeros a selected head's active value stream
`z = pattern @ v` before the attention output projection. The residual part of
the layer is left intact. For the current quadratic attention run this means:

```text
baseline:  out = x + scale * (O([z_h0, z_h1]) - x)
ablated:   out = x + scale * (O([z_h0, 0]) - x)
```

Use this as a reusable first-pass path intervention. Finer path decompositions
can build on the same forward wrapper.
