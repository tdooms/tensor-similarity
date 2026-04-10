"""
Tensor similarity computation for tracking training dynamics.
"""
import torch
from einops import einsum

def get_interaction_matrix(model, include_embedding=True, symmetrize=True):
    l, r = model.w_lr[0].unbind()  # [d_hidden, d_embed]
    u = model.w_u  # [d_output, d_hidden]
    
    if include_embedding:
        e = model.embed.weight  # [d_embed, d_input]
        
        # Contract embedding with L and R
        el = einsum(e, l, "e i, h e -> i h")  # [d_input, d_hidden]
        er = einsum(e, r, "e i, h e -> i h")  # [d_input, d_hidden]
        
        # Form interaction tensor in input space
        M = einsum(u, el, er, "o h, i h, j h -> o i j")
    else:
        # Form interaction matrix in hidden space
        M = einsum(u, l, r, "o h, h i, h j -> o i j")
    
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

def get_covariant_matrix(model1, model2, include_embedding=True, symmetrize=True, absoluteBool = False):
    

    M1 = get_interaction_matrix(model1, include_embedding, symmetrize)
    M2 = get_interaction_matrix(model2, include_embedding, symmetrize)

    if absoluteBool:
        M1 = M1.abs()
        M2 = M2.abs()
    C = einsum(M1, M2, "o1 i j, o2 i j -> o1 o2")

    return C

def covariance_similarity(model1A, model1B, model2A, model2B, include_embedding=True, symmetrize=True):
    M1 = get_covariant_matrix(model1A, model1B,include_embedding)
    M2 = get_covariant_matrix(model2A, model2B,include_embedding)

    dot_product = einsum(M1, M2, "o1 o2, o1 o2 ->")
    norm1 = torch.sqrt(einsum(M1, M1, "o1 o2, o1 o2 ->"))
    norm2 = torch.sqrt(einsum(M2, M2, "o1 o2, o1 o2 ->"))
    
    return (dot_product / (norm1 * norm2)).item()

def symmetric_inner_martin(model1, model2):
    
    ll = einsum(model1.w_l.squeeze(), model2.w_l.squeeze(), "hid1 i, hid2 i -> hid1 hid2")
    rr = einsum(model1.w_r.squeeze(), model2.w_r.squeeze(), "hid1 i, hid2 i -> hid1 hid2")

    lr = einsum(model1.w_l.squeeze(), model2.w_r.squeeze(), "hid1 i, hid2 i -> hid1 hid2")
    rl = einsum(model1.w_r.squeeze(), model2.w_l.squeeze(), "hid1 i, hid2 i -> hid1 hid2")

    core = 0.5 * ((ll * rr) + (lr * rl))

    dd = einsum(model1.w_u, model2.w_u, "o hid1, o hid2 -> hid1 hid2")
    hid = einsum(core, dd, "hid1 hid2, hid1 hid3 -> hid2 hid3")
    return torch.trace(hid)

def tn_sim_martin(model1, model2):
    inner = symmetric_inner_martin(model1, model2)
    norm1 = torch.sqrt(symmetric_inner_martin(model1, model1))
    norm2 = torch.sqrt(symmetric_inner_martin(model2, model2))
    return inner / (norm1 * norm2)