from typing import Any

import torch
import torch.distributed as dist
import torchmetrics as tm
from ray import get, train
from ray.train.torch import get_device, prepare_data_loader, prepare_model
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

import src.losses as custom_losses
import src.metrics as custom_metrics
from src.data import TestDataset, TrainDataset, ValDataset
from src.lib.base_classes import BaseModule
from src.lib.configs import TrainConfig
from src.model import DNN
from src.tools import EarlyStopping, LRScheduler

DUMMY_PARAMS = [torch.nn.Parameter(torch.tensor(1.0))]


# define as many datasets as you want
class Datasets:
    def __init__(
        self,
        train: TrainDataset = TrainDataset(),
        val: ValDataset = ValDataset(),
        test: TestDataset = TestDataset(),
    ) -> None:
        self.train = train
        self.val = val
        self.test = test


# define as many models as you want
class Models:
    def __init__(
        self,
        dnn: DNN = DNN(),
    ) -> None:
        # assing you model to class
        self.dnn = dnn
        # need to inform ray of model to optimizer mapping for instantiation
        self.map_params_to_optimizers = {"dnn": "adam"}


# define as many optimizers as you want
class Optimizers:
    def __init__(
        self,
        adam: Adam = Adam(DUMMY_PARAMS),
    ) -> None:
        self.adam = adam


# define as many losses as you want
class Losses:
    def __init__(
        self,
        loss1: tm.MeanSquaredError = tm.MeanSquaredError(),
        loss2: custom_losses.HuberLossMetric = custom_losses.HuberLossMetric(),
    ) -> None:
        # ensure you assign each to self
        self.loss1 = loss1
        self.loss2 = loss2


# etc..
class Metrics:
    def __init__(
        self,
        mae: tm.MeanAbsoluteError = tm.MeanAbsoluteError(),
        mbd: custom_metrics.MeanBiasDeviation = custom_metrics.MeanBiasDeviation(),
    ) -> None:
        self.mae = mae
        self.mbd = mbd


# any tool you may need, or none at all
class Tools:
    def __init__(
        self,
        early_stoppage: EarlyStopping = EarlyStopping(),  # callable class
        lr_scheduler: LRScheduler = LRScheduler(),  # method driven class
        # NOTE add as many as you want ...
    ):
        self.early_stoppage = early_stoppage
        self.lr_scheduler = lr_scheduler


class Experiment(BaseModule):
    def __init__(
        self,
        cfg: TrainConfig,
        models: Models,
        datasets: Datasets,
        losses: Losses,
        optimizers: Optimizers,
        metrics: Metrics,
        tools: Tools,
        # paths: dict,
    ):
        super().__init__(
            cfg=cfg,
            models=models,
            datasets=datasets,
            optimizers=optimizers,
            losses=losses,
            metrics=metrics,
            tools=tools,
            # paths=paths,
        )
        # TODO move into worker - as per earlier you can override the default values here if you want runtime specific values
        # in this case we want to override our LRScheduler to leverage the dataset runtime values
        self.tools.lr_scheduler.total_steps = (
            len(self.datasets.train)
            // self.datasets.train.batch_size
            * self.cfg.num_epochs
        )

    # NOTE dataset_shards: not strictly typed but you can access all "Datasets" attributes
    def worker_loop(self, dataset_shards):
        # prepar trainable model for train synchronizations - wrapper for DDP
        self.models.dnn = prepare_model(self.models.dnn)

        # NOTE manually put non trainable model on device
        # - e.g. self.models.resnet = self.models.resnet.to(get_device())

        train_loader, _, _ = self.generate_loaders(dataset_shards=dataset_shards)

        for epoch in range(self.cfg.start_epoch, self.cfg.num_epochs):
            print(
                f"Epoch START: {epoch}, world rank: {train.get_context().get_world_rank()}"
            )

            self.train_steps(dataloader=train_loader, epoch=epoch)

            dummy_val_loss = 69
            if self.tools.early_stoppage(dummy_val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        # self.save_checkpoint(
        #     epoch=epoch,
        #     metrics={},
        # )

    def train_steps(self, dataloader: DataLoader, epoch: int):
        self.reset_all_metrics_and_losses()
        for batch_idx, batch in enumerate(dataloader):
            x, y = batch

            output = self.models.dnn(x)
            self.losses.loss1.update(output, y)

            # self.save_metrics_and_losses(
            #     batch_idx=batch_idx,
            #     epoch=epoch,
            # )

            self.losses.loss1.backward()

            self.optimizers.adam.step()
            self.optimizers.adam.zero_grad()

            self.tools.lr_scheduler.step(self.optimizers.adam)
