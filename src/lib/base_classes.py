import os
import tempfile
from abc import ABC, abstractmethod
from typing import Any

import torch
import torchmetrics as tm
from hydra_zen import instantiate
from omegaconf import DictConfig, ListConfig
from ray import train
from torch.utils.data import Dataset


class BaseModule(ABC):
    def __init__(
        self,
        cfg,
        models,
        datasets,
        optimizers,
        losses,
        metrics,
        tools,
        # paths,
    ):
        super().__init__()
        self.cfg = cfg
        self.models = models
        self.datasets = datasets
        self.optimizers = optimizers
        self.losses = losses
        self.metrics = metrics
        self.tools = tools
        # self.paths = paths

    # @staticmethod
    @abstractmethod
    def worker_loop(self, dataset_shards: Any):
        raise NotImplementedError

    @classmethod
    def init_module(
        cls,
        cfg_dict: DictConfig | ListConfig,
        cfg_name: str,
        models_name: str,
        datasets_name: str,
        losses_name: str,
        optimizers_name: str,
        metrics_name: str,
        tools_name: str,
    ):
        # builds config and instantiates model and optimizer
        train_cfg = instantiate(getattr(cfg_dict, cfg_name))
        models = instantiate(getattr(cfg_dict, models_name))
        datasets = instantiate(getattr(cfg_dict, datasets_name))
        losses = instantiate(getattr(cfg_dict, losses_name))
        metrics = instantiate(getattr(cfg_dict, metrics_name))
        tools = instantiate(getattr(cfg_dict, tools_name))

        optimizers = {}
        for model_name, model in models.__dict__.items():
            if isinstance(model, torch.nn.Module):
                assert hasattr(
                    models,
                    "map_params_to_optimizers",
                ), "map_params_to_optimizers signature not found in models"
                assert (
                    model_name in models.map_params_to_optimizers.keys()
                ), f"{model_name} not in model_names"

                optimizer_name = models.map_params_to_optimizers[model_name]
                # annotations = inspect.get_annotations(optimizer.__init__)
                # print(annotations)
                assert (
                    optimizer_name in getattr(cfg_dict, optimizers_name).keys()
                ), f"{model_name} not in optimizer config"
                # NOTE@dev incase other optimizers are used
                # and the params kwarg is named something else change here
                optimizer = instantiate(
                    getattr(cfg_dict, optimizers_name)[optimizer_name],
                    params=model.parameters(),
                )

                # Get the optimizer type name
                optimizer_type = type(optimizer).__name__.lower()
                optimizers[optimizer_type] = optimizer

        optimizers = instantiate(getattr(cfg_dict, optimizers_name), **optimizers)

        for name, ds in datasets.__dict__.items():
            if isinstance(ds, Dataset):
                assert hasattr(
                    ds,
                    "ray_map_batches",
                ), f"ray_map_batches signature not found in dataset {name}"

        return cls(
            train_cfg,
            models,
            datasets,
            losses,
            optimizers,
            metrics,
            tools,
        )

    def generate_loaders(self, dataset_shards):
        loaders = []
        for name, ds_conf in self.datasets.__dict__.items():
            ds = getattr(dataset_shards, name)

            loader = ds.iter_torch_batches(
                batch_size=ds_conf.batch_size,
                # prefetch_batches=self.cfg.prefetch_batches,
                collate_fn=ds_conf.ray_map_batches,
            )

            loaders.append(loader)

        return loaders

    def save_checkpoint(
        self,
        epoch: int,
        metrics: dict[str, Any],
    ):
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dicts": {
                name: model.state_dict()
                for name, model in self.models.__dict__.items()
                if isinstance(model, torch.nn.Module)
            },
            "optimizer_state_dicts": {
                name: optimizer.state_dict()
                for name, optimizer in self.optimizers.__dict__.items()
                if isinstance(optimizer, torch.optim.Optimizer)
            },
        }

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            checkpoint = None
            if train.get_context().get_world_rank() == 0:
                checkpoint_path = os.path.join(temp_checkpoint_dir, "states.pt")
                torch.save(checkpoint_data, checkpoint_path)
                checkpoint = train.Checkpoint.from_directory(temp_checkpoint_dir)

            train.report(
                self.save_metrics_and_losses(epoch=epoch, report=False),
                checkpoint=checkpoint,
            )

    def restore_checkpoint(self):
        assert self.cfg.restore_checkpoint_path, "No checkpoint to restore."
        assert os.path.exists(
            self.cfg.restore_checkpoint_path,
        ), f"Checkpoint path {self.cfg.restore_checkpoint_path} does not exist."

        print(f"Restoring checkpoint from {self.cfg.restore_checkpoint_path}")

        checkpoint = torch.load(self.cfg.restore_checkpoint_path)

        for name, model in self.models.__dict__.items():
            if isinstance(model, torch.nn.Module):
                model.load_state_dict(checkpoint["model_state_dicts"][name])

        for name, optimizer in self.optimizers.__dict__.items():
            if isinstance(optimizer, torch.optim.Optimizer):
                optimizer.load_state_dict(checkpoint["optimizer_state_dicts"][name])

        self.cfg.start_epoch = checkpoint["epoch"] + 1

    def save_metrics(
        self,
        epoch: int,
        batch_idx: int | None = None,
        report: bool = True,
    ):
        metric_log = {
            "epoch": epoch,
            "batch": batch_idx,
        }
        for name, value in self.metrics.__dict__.items():
            if isinstance(value, tm.Metric):
                metric_log[name] = value.compute().item()

        if report:
            train.report(metric_log)

        return metric_log

    def save_losses(
        self,
        epoch: int,
        batch_idx: int | None = None,
        report: bool = True,
    ):
        loss_log = {
            "epoch": epoch,
            "batch": batch_idx,
        }
        for name, value in self.losses.__dict__.items():
            if isinstance(value, tm.Metric):
                loss_log[name] = value.compute().item()

        if report:
            train.report(loss_log)

        return loss_log

    def save_metrics_and_losses(
        self,
        epoch: int,
        batch_idx: int | None = None,
        report: bool = True,
    ):
        kwargs = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "report": report,
        }
        losses = self.save_losses(**kwargs)
        metrics = self.save_metrics(**kwargs)

        log = {**metrics, **losses}
        if report:
            train.report(log)

        return log

    def reset_all_metrics_and_losses(self):
        for name, value in self.losses.__dict__.items():
            if isinstance(value, tm.Metric):
                self.losses.__dict__[name].reset()

        for name, value in self.metrics.__dict__.items():
            if isinstance(value, tm.Metric):
                self.metrics.__dict__[name].reset()
