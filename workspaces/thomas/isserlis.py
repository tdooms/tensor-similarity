from networkx import sigma
from torch import nn
from quimb.tensor import Tensor, TensorNetwork

import torch

def parallel(tensor: Tensor | TensorNetwork, n: int = 2, tag: str = 'out'):
    """Reindexes a tensor (network) with $n$ parallel copies."""
    k = sum(ind.startswith('in:d') for ind in tensor.ind_map)
    
    # TODO: it'd be nicer if all indices were given a number automatically
    tensors = [tensor.reindex({'out:d': f'{tag}:d{j}'} | {f'in:d{i}': f'in:d{j*k + i}' for i in range(k)}) for j in range(n)]
    return TensorNetwork(tensors)

def sequential2(first, second, tag='l'):
    """Helper function to sequentially stack two tensor networks."""
    n = sum(ind.startswith('in:d') for ind in second.ind_map)
    return parallel(first, n=n, tag=tag) | second.reindex({f'in:d{i}': f'{tag}:d{i}' for i in range(n)})

def sequential(*nets, tag='l'):
    """Sequentially stack tensor networks, accounting for input cloning."""
    current = nets[0]
    for i, net in enumerate(nets[1:]):
        current = sequential2(current, net, tag=f'{tag}{i+1}')
    return current

def transpose(tn: TensorNetwork):
    return tn.reindex({idx: idx + '*' for idx in tn.all_inds()})

class State:
    def __init__(self, mu_a, mu_b, sigma_aa, sigma_bb, sigma_ab, delta_exp=0.0):
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.sigma_aa = sigma_aa
        self.sigma_bb = sigma_bb
        self.sigma_ab = sigma_ab
        self.delta_exp = delta_exp
    
    @staticmethod
    def default(d, device="cuda:0", dtype=torch.float32):
        return State(
            mu_a=torch.tensor([1] + [0]*d, device=device, dtype=dtype),
            mu_b=torch.tensor([1] + [0]*d, device=device, dtype=dtype),
            sigma_aa=torch.diag(torch.tensor([0] + [1]*d, device=device, dtype=dtype)),
            sigma_bb=torch.diag(torch.tensor([0] + [1]*d, device=device, dtype=dtype)),
            sigma_ab=torch.diag(torch.tensor([0] + [1]*d, device=device, dtype=dtype)),
            delta_exp=0.0
        )

class Component(nn.Module):
    """A wrapper for all tensor tree layers with utility functions.

    We use the following naming conventions:
    - The indices have three prefixes: in(put), h(idden), and out(put).
    - The input indices are named 'in:d0', 'in:d1', ..., 'in:d{N-1}' for N inputs.
    - The output index is named 'out:d'.
    - Other indices are for instance 'in:s0' (first sequence input) or 'in:n' (normalisation input).
    
    We use the following additional conventions:
    - The first dimension of all '*:d' indices is a constant/bias dimension.
    """
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        
    def _like(self):
        # t = list(self.inner.)[0]
        # return dict(device=t.device, dtype=t.dtype)
        return dict(device="cuda:0", dtype=torch.float32)
        
    def to(self, device):
        self.inner.apply_to_arrays(lambda x: x.to(device))
        return self

    def type(self, dtype):
        self.inner.apply_to_arrays(lambda x: x.type(dtype))
        return self
    
    def make_inputs(self, input, modes=['b']):
        n = len(self.input_inds())
        return [Tensor(input, inds=[*modes, f'in:d{i}']) for i in range(n)]
    
    def evaluate(self, input, extra=[], modes=['b']):
        """Contracts a given input into the component tensor network and evaluates it.
        
        extra: any additional (Quimb) Tensors to be contracted (e.g. normalisation vectors).
        modes: the additional batch modes to keep track of (e.g. sequences).
        """
        
        inputs = self.make_inputs(input, modes=modes)
        return TensorNetwork([self.inner, *inputs, *extra]).contract(all, [*modes, 'out:d']).data
    
    def input_inds(self, only_dims=True):
        """Returns the input wire names of the component."""
        return [idx for idx in self.inner.ind_map.keys() if idx.startswith('in:d' if only_dims else 'in:')]
    
    def d_input(self):
        """Returns the total input dimension of the component."""
        return self.inner.ind_size(self.input_inds()[0])

    def output_inds(self, only_dims=True):
        """Returns the output wire names of the component."""
        return [idx for idx in self.inner.ind_map.keys() if idx.startswith('out:d' if only_dims else 'out:')]
    
    def hidden_inds(self, only_dims=True):
        """Returns the hidden wire names of the component."""
        return [idx for idx in self.inner.ind_map.keys() if idx.startswith('h:d' if only_dims else 'h:')]
    
    def contract(self, other=None, metric=None, out: str = 'out:d'):
        """Self-contract the inputs of the component leaving only one specified wire.

        bra | ket: The two tensor networks (following module conventions) to be contracted.
        metric: the optional matrix to fold between the traced indices.
        out: the name of the output wire to keep on both sides.
        
        This function does not currently support including a matrix in the output index.
        Returns a (PyTorch) matrix.
        """
        
        bra, ket = self.inner, self.inner if other is None else other.inner
        # assert bra.input_inds() == ket.input_inds(), "Both tensor networks must have the same input indices."
        n = len(self.input_inds())
        
        # Make a list of matrices to fold between the trace OR fold in no matrices.
        if metric is not None:
            inputs = [Tensor(metric, inds=[f'in:d{i}', f'in:d{i}*'], tags=['F']) for i in range(n)]
            bra = transpose(bra)
        else:
            inputs = []
            bra = bra.reindex({out: out + '*'})
        
        # Construct and contract the traced network and return the gram matrix.
        cross = TensorNetwork([bra, ket, *inputs])
        gram, exp = cross.contract_tags(all, output_inds=[out, out + '*'], equalize_norms=True, strip_exponent=True)
        return gram.data, exp
    
    def isserlis(self, other=None, state=None):
        assert len(self.input_inds()) in [1, 2], "Currently only supports 1 or 2 input indices."
        
        state = state if (state is not None and state != 0) else State.default(self.d_input() - 1, **self._like())
        bra, ket = self.inner, self.inner if other is None else other.inner
        
        def sigma_metric(sigma, mu, exp=0.0):
            if not isinstance(exp, torch.Tensor):
                exp = torch.tensor(exp, device=sigma.device, dtype=sigma.dtype)
        
            lmetric = Tensor(torch.stack([mu, sigma]), inds=['s', 'in:d0', 'in:d0*'], tags=['M'])
            rmetric = Tensor(torch.stack([sigma, sigma]), inds=['s', 'in:d1', 'in:d1*'], tags=['M'])
            weight = Tensor(torch.stack([torch.tensor(4.0, **self._like()), 2.0*(10**exp)]), inds=['s'], tags=['W'])
            return lmetric & rmetric & weight
        
        def mu_metric(sigma, mu):
            return Tensor(sigma + mu.outer(mu), inds=['in:d0', 'in:d1'], tags=['M'])
        
        if len(self.input_inds()) == 1:
            m = lambda mu, i: Tensor(mu, inds=[f'in:d{i}'], tags=['M'])
            mu_a = (m(state.mu_a, 0) & bra).contract(all, ['out:d']).data
            mu_b = (m(state.mu_b, 0) & ket).contract(all, ['out:d']).data
            
            s_aa = Tensor(state.sigma_aa, inds=[f'in:d0', f'in:d0*'], tags=['S'])
            s_bb = Tensor(state.sigma_bb, inds=[f'in:d0', f'in:d0*'], tags=['S'])
            s_ab = Tensor(state.sigma_ab, inds=[f'in:d0', f'in:d0*'], tags=['S'])
            
            sigma_aa = (bra & s_aa & bra.reindex({'in:d0': 'in:d0*', 'out:d': 'out:d*'})).contract(all, ['out:d', 'out:d*']).data
            sigma_bb = (ket & s_bb & ket.reindex({'in:d0': 'in:d0*', 'out:d': 'out:d*'})).contract(all, ['out:d', 'out:d*']).data
            sigma_ab = (bra & s_ab & ket.reindex({'in:d0': 'in:d0*', 'out:d': 'out:d*'})).contract(all, ['out:d', 'out:d*']).data
            
            return State(mu_a, mu_b, sigma_aa, sigma_bb, sigma_ab, delta_exp=state.delta_exp)
        
        sigma_aa = bra & sigma_metric(state.sigma_aa, state.mu_a.outer(state.mu_a)) & transpose(bra)
        sigma_bb = ket & sigma_metric(state.sigma_bb, state.mu_b.outer(state.mu_b)) & transpose(ket)
        sigma_ab = bra & sigma_metric(state.sigma_ab, state.mu_a.outer(state.mu_b), exp=state.delta_exp) & transpose(ket)
        
        mu_a = bra & mu_metric(state.sigma_aa, state.mu_a)
        mu_b = ket & mu_metric(state.sigma_bb, state.mu_b)

        sigma_aa, aa_exp = sigma_aa.contract_tags(all, output_inds=['out:d', 'out:d*'], equalize_norms=True, strip_exponent=True)
        sigma_bb, bb_exp = sigma_bb.contract_tags(all, output_inds=['out:d', 'out:d*'], equalize_norms=True, strip_exponent=True)
        sigma_ab, ab_exp = sigma_ab.contract_tags(all, output_inds=['out:d', 'out:d*'], equalize_norms=True, strip_exponent=True)
        
        mu_a = mu_a.contract(all, output_inds=['out:d']) / (10 ** (aa_exp/2))
        mu_b = mu_b.contract(all, output_inds=['out:d']) / (10 ** (bb_exp/2))
        
        return State(
            mu_a=mu_a.data,
            mu_b=mu_b.data,
            sigma_aa=sigma_aa.data,
            sigma_bb=sigma_bb.data,
            sigma_ab=sigma_ab.data,
            delta_exp=state.delta_exp + (ab_exp - (aa_exp + bb_exp) / 2)
        )

    def trace(self, metric=None):
        metric = metric if metric is not None else torch.eye(self.d_input())
        metric = Tensor(metric, inds=['in:d0', 'in:d1'], tags=['M'])
        
        tn = TensorNetwork([self.inner, metric])
        gram, exp = tn.contract_tags(all, output_inds=['out:d'], equalize_norms=True, strip_exponent=True)
        return gram.data, exp