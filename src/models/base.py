
import torch

from torch import nn
from itertools import accumulate
from abc import abstractmethod
from operator import mul

from src.components import Component

class Model(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
    
    @abstractmethod
    def components(self):
        raise NotImplementedError()
    
    # def gauges(self):
    #     """compute the gauges that diagonalise the full model."""
    #     networks = [component.network() for component in self.components()]
        
    #     # Make all (but the last) layers orthogonal through recursive input self contractions.
    #     # data = (grammian, exponent)
    #     orth = list(accumulate(networks[:-1], lambda data, net: Component.isc(net, data[0]), initial=(None, 0)))[1:]

    #     # Make all layers diagonal through parallel output self contractions.
    #     diag = [Gauge.from_gram(*Component.osc(net, gauge), absorb='mid') for net, gauge in zip(networks[1:], orth)]

    #     # Combine both gauges.
    #     return [o.inject(d) for o, d in zip(orth, diag)]
    
    def norm(self):
        cp = self.n_copies_per_layer()
        gauges = list(accumulate(self.components(), lambda data, c: c.contract(inner=data[0]), initial=(None, 0)))[1:]
        return sum([e * c for c, (_, e) in zip(cp, gauges)]) + gauges[-1][0].trace().log10()
    
    def similarity(self, other):
        zipped = zip(self.components(), other.components())
        gauges = list(accumulate(zipped, lambda data, c: c[0].contract(c[1], data[0]), initial=(None, 0)))[1:]
        
        cp = self.n_copies_per_layer()
        trace = sum([e * c for c, (_, e) in zip(cp, gauges)]) + gauges[-1][0].trace().abs().log10()
        
        return 2 * trace - (self.norm() + other.norm())
    
    def n_copies_per_layer(self):
        """Returns the number of copies per layer in the tensor model."""
        n_inds = [len(c.input_inds()) for c in self.components()]
        return list(reversed(list(accumulate(reversed(n_inds[1:]), mul, initial=1))))
    
    # def evaluate(self, x, gauges=[], **kwargs):
    #     """Evaluate the tensor model on an input given optional gauges."""
    #     # Pad the input to account for the tensor model bias.
    #     x = torch.nn.functional.pad(x, (1, 0), value=1)
    #     info = []
        
    #     for component, gauge in zip_longest(self.components(), gauges, fillvalue=Fake()):
    #         # Normalise between each component for numerical stability.
    #         x = x * x[..., 1:].pow(2).mean(dim=-1, keepdim=True).rsqrt()
    #         # Forward pass through a the component's tensor network.
    #         x = component.evaluate(x)
    #         # Forward pass through a (possibly truncated) gauge if provided.
    #         x, metrics = gauge(x, **kwargs)
    #         info.append(metrics)
        
    #     return x[..., 1:], DataFrame.from_dict(info[:-1])