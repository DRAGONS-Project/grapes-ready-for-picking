import abc

from torch.utils.data import Dataset


class BaseDataset(abc.ABC, Dataset):
    """Base class for datasets.

    Subclass this and implement __len__ and __getitem__. Register your
    subclass in configs/dataset/<variant>.yaml via _target_: dataset.<module>.<Class>.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def __len__(self) -> int:
        pass

    @abc.abstractmethod
    def __getitem__(self, idx):
        pass
