import torch
from torch import nn
from typing import NamedTuple
from quimb.tensor import Tensor, TensorNetwork
from abc import abstractmethod


class Term(NamedTuple):
    """A TN term with sequence wires, leg-to-position map, and optional symmetries.

    `symmetries` is a tuple of dict permutations on input-leg data names under
    which the term value is invariant. Used by Isserlis to dedupe matchings —
    each permutation maps leg-data names (e.g. 'in:d1') to leg-data names.
    """
    tn: TensorNetwork
    legs: dict
    symmetries: tuple = ()


class Component(nn.Module):
    """A wrapper class for all compositional modules.

    We use the following naming conventions:
    - Module indices have three prefixes: in(put), h(idden), and out(put).
    - The input indices are named 'in:d0', 'in:d1', ..., 'in:d{N-1}' for N inputs.
    - The output index is named 'out:d'.
    - Other indices are for instance 'in:s0' (first sequence input) or 'in:n' (normalisation input).
    
    We use the following additional conventions:
    - The first dimension of all '*:d' indices is a constant/bias dimension.
    """

    @abstractmethod
    def network(self):
        """Returns a list of tensor networks representing the component's layer."""
        return NotImplemented
    
    def make_inputs(self, input, modes=('b',)):
        n = len(self.input_inds())
        return [Tensor(input, inds=[*modes, f'in:d{i}']) for i in range(n)]
        
    def evaluate(self, input, extra=(), modes=('b',)):
        """Contracts a given input into the component tensor network and evaluates it.
        
        extra: any additional (Quimb) Tensors to be contracted (e.g. normalisation vectors).
        modes: the additional batch modes to keep track of (e.g. sequences).
        """
        
        inputs = self.make_inputs(input, modes=modes)
        return TensorNetwork([self.network(), *inputs, *extra]).contract(all, [*modes, 'out:d']).data
    
    def input_inds(self, only_dims=True):
        """Returns the input wire names of the component."""
        return [idx for idx in self.network().ind_map.keys() if idx.startswith('in:d' if only_dims else 'in:')]

    def output_inds(self, only_dims=True):
        """Returns the output wire names of the component."""
        return [idx for idx in self.network().ind_map.keys() if idx.startswith('out:d' if only_dims else 'out:')]
    
    def hidden_inds(self, only_dims=True):
        """Returns the hidden wire names of the component."""
        return [idx for idx in self.network().ind_map.keys() if idx.startswith('h:d' if only_dims else 'h:')]
    
    def terms(self, n_ctx, **like):
        """TN terms with sequence wires. f(x) = sum of terms.

        Default: all input legs and the output share the same sequence position
        (handled implicitly by all legs pointing to 'out:s').
        """
        tn = self.network()
        legs = sorted(idx for idx in tn.outer_inds() if idx.startswith('in:d'))
        return [Term(tn, {leg: 'out:s' for leg in legs})]

    def permutations(self):
        """Returns all Wick matchings (perfect matchings) of the input legs.

        Each matching is a list of (idx_a, idx_b) pairs specifying which indices
        to identify. By default, generates all (2N)!! matchings over the N input
        legs and their starred counterparts. Subclasses may override to provide
        a sparse subset.
        """
        legs = self.input_inds()
        all_legs = legs + [f'{l}*' for l in legs]

        def match(l):
            if not l: return [[]]
            return [[(l[0], l[i])] + r for i in range(1, len(l)) for r in match(l[1:i] + l[i+1:])]

        return match(all_legs)

    def contract(self, other=None, inner=None, out='out:d'):
        """Contract this component with another, summing over permutations.

        When inner is None: uses permutation-based contraction, starring other's
        outer indices and summing over self.permutations().
        When inner is provided: gauge-propagation mode, folding the inner matrix
        between traced indices (used by Model.norm/similarity).
        """
        other = other or self

        if inner is not None:
            bra, ket = self.network(), other.network()
            n = len(self.input_inds())
            inputs = [Tensor(inner, inds=(f'in:d{i}', f'in:d{i}*'), tags=('F',)) for i in range(n)]
            bra = bra.reindex({idx: idx + '*' for idx in bra.all_inds()})
            cross = TensorNetwork([bra, ket, *inputs])
            gram, exp = cross.contract_tags(all, output_inds=(out, f'{out}*'), equalize_norms=True, strip_exponent=True)
            return gram.data, exp

        net_a = self.network()
        net_b = other.network().reindex({leg: f'{leg}*' for leg in other.network().outer_inds()})

        return sum(
            (net_a.copy() & net_b.copy())
            .reindex(dict(perm))
            .contract(output_inds=(out, f'{out}*'))
            .data
            for perm in self.permutations()
        )