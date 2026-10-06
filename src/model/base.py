import abc

import torch.nn as nn


class BaseModel(abc.ABC, nn.Module):
    """Base class for models.

    Subclass this and implement forward(). Register your subclass in
    configs/model/<variant>.yaml via _target_: model.<module>.<Class>.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        pass
