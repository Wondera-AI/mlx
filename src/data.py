import ray
import ray.data as ray_data
import torch
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(
        self,
        batch_size: int = 32,
        unused_data_arg: int = 0,
    ):
        # add whatever args you need to use around the codebase
        self.batch_size = batch_size
        self.unused_data_arg = unused_data_arg

        # do what ever to load your data here - os walks etc...
        self.data = torch.randn(1000, 180)
        self.labels = torch.randint(0, 10, (1000,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @staticmethod
    def ray_map_batches(batch):
        # NOTE please read this all !! as Ray handles data in many many ways
        # sadly ray data is ugly and requires either a custom collate function or a custom map function
        # the solution here is to declare it as a normal pytorch dataset
        # and enforce declaration of a custom map function
        # NOTE this actually enables us to interchangeably use this method to map during or prior to training
        # NOTE prior to training will require new features to map, save and store the mapped ray dataset
        # for now we will use this to collate during training
        # latter reducing this iteration should speed things up for each batch loaded
        # NOTE in the interim there is a way to pre-load the iter_torch_batches
        # https://docs.ray.io/en/latest/data/api/input_output.html

        batch = batch["item"]  # and yes unfornutaley you must retrieve from item
        # but then you get access to the __getitem__ return signature
        data = torch.stack([torch.tensor(row[0]) for row in batch])
        labels = torch.stack([torch.tensor(row[1]) for row in batch])

        return {"data": data, "label": labels}


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

    @staticmethod
    def ray_map_batches(batch):
        batch = batch["item"]
        data = torch.stack([torch.tensor(row[0]) for row in batch])
        labels = torch.stack([torch.tensor(row[1]) for row in batch])

        return {"data": data, "label": labels}


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

    @staticmethod
    def ray_map_batches(batch):
        batch = batch["item"]
        data = torch.stack([torch.tensor(row[0]) for row in batch])
        labels = torch.stack([torch.tensor(row[1]) for row in batch])

        return {"data": data, "label": labels}


# ds = TrainDataset()
# ray_ds = ray_data.from_torch(ds)
# print(ray_ds.schema())
# batches = 0
# for _, batch in enumerate(
#     ray_ds.iter_torch_batches(
#         batch_size=32,
#         collate_fn=ds.ray_map_batches,
#     ),
# ):
#     print(batch["data"].shape, batch["label"].shape)
#     batches += 1

# print("Batches:", batches)
