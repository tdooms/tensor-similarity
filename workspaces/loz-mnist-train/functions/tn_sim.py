"""
Tensor similarity computation for tracking training dynamics.
"""
import torch
from einops import einsum

from src.components.similarity import similarity as _thomas_similarity
from src.paper.shared import cosine as _thomas_cosine


def get_interaction_matrix(model, include_embedding=True, symmetrize=True):
    """Build M[o,i,j] = Σ_h u_eff[o,h] · el[i,h] · er[j,h].

    Architecture: embed(d_input→d_embed) → MLP(L,R: d_embed→d_hidden, D: d_hidden→d_embed) → head(d_embed→d_output)
    u_eff = head.weight @ D absorbs the D projection into the effective unembed.
    """
    l, r = model.w_lr[0].unbind()  # [d_hidden, d_embed]
    d = model.w_d                   # [d_embed, d_hidden]
    u = model.w_u                   # [d_output, d_embed]
    u_eff = u @ d                   # [d_output, d_hidden]

    if include_embedding:
        e = model.embed.weight      # [d_embed, d_input]
        el = einsum(e, l, "m i, h m -> i h")  # [d_input, d_hidden]
        er = einsum(e, r, "m i, h m -> i h")  # [d_input, d_hidden]
        M = einsum(u_eff, el, er, "o h, i h, j h -> o i j")
    else:
        M = einsum(u_eff, l, r, "o h, h i, h j -> o i j")

    if symmetrize:
        M = 0.5 * (M + M.transpose(1, 2))

    return M


def tensor_similarity(T1, T2):
    dot_product = einsum(T1, T2, "o i j, o i j ->")
    norm1 = torch.sqrt(einsum(T1, T1, "o i j, o i j ->"))
    norm2 = torch.sqrt(einsum(T2, T2, "o i j, o i j ->"))
    return (dot_product / (norm1 * norm2)).item()


def model_similarity(model1, model2, include_embedding=True, symmetrize=True):
    M1 = get_interaction_matrix(model1, include_embedding=include_embedding, symmetrize=symmetrize)
    M2 = get_interaction_matrix(model2, include_embedding=include_embedding, symmetrize=symmetrize)
    return tensor_similarity(M1, M2)


def tn_sim_thomas(model1, model2):
    """Thomas's exact Gaussian functional similarity (requires model.components())."""
    state = _thomas_similarity(model1, model2)
    return _thomas_cosine(state)


def get_covariant_matrix(model1, model2, include_embedding=True, symmetrize=True, absoluteBool=False):
    M1 = get_interaction_matrix(model1, include_embedding, symmetrize)
    M2 = get_interaction_matrix(model2, include_embedding, symmetrize)

    if absoluteBool:
        M1 = M1.abs()
        M2 = M2.abs()
    C = einsum(M1, M2, "o1 i j, o2 i j -> o1 o2")
    return C


def covariance_similarity(model1A, model1B, model2A, model2B, include_embedding=True, symmetrize=True):
    M1 = get_covariant_matrix(model1A, model1B, include_embedding)
    M2 = get_covariant_matrix(model2A, model2B, include_embedding)

    dot_product = einsum(M1, M2, "o1 o2, o1 o2 ->")
    norm1 = torch.sqrt(einsum(M1, M1, "o1 o2, o1 o2 ->"))
    norm2 = torch.sqrt(einsum(M2, M2, "o1 o2, o1 o2 ->"))
    return (dot_product / (norm1 * norm2)).item()


def symmetric_inner_martin(model1, model2):
    ll = einsum(model1.w_l.squeeze(), model2.w_l.squeeze(), "hid1 i, hid2 i -> hid1 hid2")
    rr = einsum(model1.w_r.squeeze(), model2.w_r.squeeze(), "hid1 i, hid2 i -> hid1 hid2")
    lr = einsum(model1.w_l.squeeze(), model2.w_r.squeeze(), "hid1 i, hid2 i -> hid1 hid2")
    rl = einsum(model1.w_r.squeeze(), model2.w_l.squeeze(), "hid1 i, hid2 i -> hid1 hid2")

    core = 0.5 * ((ll * rr) + (lr * rl))  # [d_hidden, d_hidden]

    # Use effective unembed: u_eff = head.weight @ D
    u_eff1 = model1.w_u @ model1.w_d  # [d_output, d_hidden]
    u_eff2 = model2.w_u @ model2.w_d  # [d_output, d_hidden]
    dd = einsum(u_eff1, u_eff2, "o hid1, o hid2 -> hid1 hid2")  # [d_hidden, d_hidden]

    hid = einsum(core, dd, "hid1 hid2, hid1 hid3 -> hid2 hid3")
    return torch.trace(hid)


def tn_sim_martin(model1, model2):
    inner = symmetric_inner_martin(model1, model2)
    norm1 = torch.sqrt(symmetric_inner_martin(model1, model1))
    norm2 = torch.sqrt(symmetric_inner_martin(model2, model2))
    return inner / (norm1 * norm2)
