from torch import nn
from quimb.tensor import Tensor, TensorNetwork
from abc import abstractmethod
   
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
        """Returns a list of tensor networks representing the component."""
        return NotImplemented
    
    def make_inputs(self, input, modes=['b']):
        n = len(self.input_inds())
        return [Tensor(input, inds=[*modes, f'in:d{i}']) for i in range(n)]
        
    def evaluate(self, input, extra=[], modes=['b']):
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
    
    def contract(self, other=None, inner=None, out: str = 'out:d'):
        """Self-contract the inputs of the component leaving only one specified wire.

        bra | ket: The two tensor networks (following module conventions) to be contracted.
        inner: the optional matrix to fold between the traced indices.
        out: the name of the output wire to keep on both sides.
        
        This function does not currently support including a matrix in the output index.
        Returns a (PyTorch) matrix.
        """
        
        bra, ket = self.network(), self.network() if other is None else other.network()
        # assert bra.input_inds() == ket.input_inds(), "Both tensor networks must have the same input indices."
        n = len(self.input_inds())
        
        # Make a list of matrices to fold between the trace OR fold in no matrices.
        if inner is not None:
            inputs = [Tensor(inner, inds=[f'in:d{i}', f'in:d{i}*'], tags=['F']) for i in range(n)]
            bra = bra.reindex({idx: idx + '*' for idx in bra.all_inds()})
        else:
            inputs = []
            bra = bra.reindex({out: out + '*'})
        
        # Construct and contract the traced network and return the gram matrix.
        # The contraction strips tensor exponents (e.g. normalizes them) to avoid numerical issues.
        cross = TensorNetwork([bra, ket, *inputs])
        gram, exp = cross.contract_tags(all, output_inds=[out, out + '*'], equalize_norms=True, strip_exponent=True)
        return gram.data, exp