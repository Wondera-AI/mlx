import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self,
        batch_size: int = 32,
        unused_data_arg: int = 0,
    ):
        self.batch_size = batch_size
        self.data = torch.randn(1000, 180)
        self.labels = torch.randint(0, 10, (1000,))
        self.unused_data_arg = unused_data_arg

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ValDataset(Dataset):
    def __init__(
        self,
        batch_size: int = 32,
    ):
        self.batch_size = batch_size
        self.data = torch.randn(1000, 180)
        self.labels = torch.randint(0, 10, (1000,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class TestDataset(Dataset):
    def __init__(
        self,
        batch_size: int = 32,
    ):
        self.batch_size = batch_size
        self.data = torch.randn(1000, 180)
        self.labels = torch.randint(0, 10, (1000,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
