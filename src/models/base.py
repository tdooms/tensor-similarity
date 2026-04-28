from abc import abstractmethod

from torch import nn


class Model(nn.Module):
    n_ctx: int = 1

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

    @abstractmethod
    def components(self):
        raise NotImplementedError()
