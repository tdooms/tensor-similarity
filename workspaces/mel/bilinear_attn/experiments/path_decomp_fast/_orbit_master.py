from __future__ import annotations

from functools import cache
from math import prod

import torch

from src.components.base import Term
from src.components.similarity import _OUT, _join
from src.components.utils import bridged_contract, orbits


@cache
def _isserlis_plan_no_sym(legs_basic, device, dtype):
    n = len(legs_basic)
    from src.components.utils import matchings

    all_m = matchings(tuple(range(n)))
    configs = tuple(all_m) + (tuple((i, i) for i in range(n)),)
    weights = (1.0,) * len(all_m) + (-(prod(range(1, n, 2)) - 1.0),)
    return configs, torch.tensor(weights, device=device, dtype=dtype)


def _joint_swap_perm(n_legs_side: int, offset: int) -> dict[int, int]:
    if n_legs_side != 5:
        return {}
    return {
        offset + 1: offset + 2,
        offset + 2: offset + 1,
        offset + 3: offset + 4,
        offset + 4: offset + 3,
    }


def _freeze_perm(perm: dict[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(sorted(perm.items()))


def _thaw_perm(perm):
    return dict(perm)


def _apply_perm_to_cfg(cfg, perm: dict[int, int]):
    out = []
    for i, j in cfg:
        ii = perm.get(i, i)
        jj = perm.get(j, j)
        out.append(tuple(sorted((ii, jj))))
    return tuple(sorted(out))


def _canon_cfg(cfg, frozen_perm):
    return _apply_perm_to_cfg(cfg, _thaw_perm(frozen_perm))


def _src_axis_permute(frozen_perm, n_legs: int) -> list[int]:
    perm = _thaw_perm(frozen_perm)
    inv = {perm.get(i, i): i for i in range(n_legs)}
    return [inv[i] for i in range(n_legs)]


def _orbit_group(n_left: int, n_right: int):
    left = (_freeze_perm(_joint_swap_perm(n_left, 0)),) if n_left == 5 else ()
    right = (_freeze_perm(_joint_swap_perm(n_right, n_left)),) if n_right == 5 else ()
    group = [()]
    if left:
        group.append(left[0])
    if right:
        group.append(right[0])
    if left and right:
        group.append(left[0] + right[0])
    return tuple(group)


def _bridges_for_cfg(cfg, legs_basic, src_names, S, bridges_to_f32: bool):
    bridges = []
    for i, j in cfg:
        a = legs_basic[i]
        b = legs_basic[j]
        if i == j:
            m = a[2]
            data = torch.stack([S[m, m, s, s, :, :, 0, 0] for s in range(2)])
            inds = (src_names[i],) + a[:2]
        else:
            data = S[a[2], b[2]]
            inds = (src_names[i], src_names[j]) + a[:2] + b[:2]
        if bridges_to_f32 and data.dtype in (torch.float16, torch.bfloat16):
            data = data.float()
        bridges.append((data, inds))
    return bridges


def _master_moment(tl, tr, ml, mr, S, bridges_to_f32: bool = False):
    tl = Term(tl.tn, tl.legs, symmetries=())
    tr = Term(tr.tn, tr.legs, symmetries=())
    tn, legs_basic, _ = _join(tl, tr, ml, mr)
    n_legs = len(legs_basic)
    n_left = len(tl.legs)
    n_right = len(tr.legs)
    src_names = tuple(f"src:leg{i}" for i in range(n_legs))
    out_inds = _OUT + src_names

    configs, weights = _isserlis_plan_no_sym(legs_basic, S.device, S.dtype)
    wick_configs = configs[:-1]
    correction_cfg = configs[-1]
    correction_w = float(weights[-1].item())

    group = _orbit_group(n_left, n_right)
    orbit_reps = orbits(wick_configs, group, _canon_cfg)

    master = None
    base_axes = list(range(len(_OUT)))

    for rep_cfg, _orbit_size in orbit_reps:
        rep_bridges = _bridges_for_cfg(rep_cfg, legs_basic, src_names, S, bridges_to_f32)
        contrib_rep = bridged_contract(tn, rep_bridges, out_inds)

        seen_members = {}
        for g in group:
            member = _apply_perm_to_cfg(rep_cfg, _thaw_perm(g))
            if member not in seen_members:
                seen_members[member] = g

        orbit_sum = None
        for g in seen_members.values():
            perm = _src_axis_permute(g, n_legs)
            contrib_g = contrib_rep.permute(*(base_axes + [len(_OUT) + p for p in perm]))
            orbit_sum = contrib_g if orbit_sum is None else orbit_sum + contrib_g

        master = orbit_sum if master is None else master + orbit_sum

    corr_bridges = _bridges_for_cfg(correction_cfg, legs_basic, src_names, S, bridges_to_f32)
    corr = bridged_contract(tn, corr_bridges, out_inds)
    master = master + correction_w * corr
    return master
