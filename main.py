# from hydra.utils import instantiate
import inspect
import os
from dataclasses import field, make_dataclass
from typing import Any, Literal, cast

import ray
import torch
from hydra_zen import builds, instantiate, load_from_yaml, save_as_yaml
from omegaconf import DictConfig, ListConfig, OmegaConf
from ray import train
from ray.air.config import ScalingConfig
from ray.data import from_torch
from ray.data.context import DataContext
from ray.runtime_env import RuntimeEnv
from ray.train import CheckpointConfig, DataConfig, FailureConfig, RunConfig
from ray.train.torch import TorchTrainer
from torch import ge, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset
from torchmetrics import Metric

from src.data import TestDataset, TrainDataset, ValDataset
from src.experiment import Experiment
from src.lib.configs import TrainConfig

CONF_PATH = "conf.yaml"

# TRAIN_DATA = "train"

# EXPERIMENT_DIR = (
#     "/mnt/nfs/experiments" if os.path.ismount("/mnt/nfs") else "./experiments"
# )
# LOG_DIR = os.path.join(EXPERIMENT_DIR, "configs/train_logs")
# CHECKPOINT_DIR = os.path.join(EXPERIMENT_DIR, "configs/checkpoints")
# BEST_VERSION_DIR = os.path.join(EXPERIMENT_DIR, "configs/best_version")


# runtime_env = RuntimeEnv(
#     container={"image": "docker.io/alelat/mlx-ray-job:latest"},
# )

"""
        --->>> POST - INITIALIZE RAY ENVIROMENT <<<---
        --->>> POST - INITIALIZE RAY WORKERS <<<---
"""


def train_fn(config: dict):
    exp = Experiment.init_module(
        cfg_dict=config["cfg_dict"],
        cfg_name=config["train_cfg"],
        models_name=config["models"],
        datasets_name=config["datasets"],
        losses_name=config["losses"],
        optimizers_name=config["optimizers"],
        metrics_name=config["metrics"],
        tools_name=config["tools"],
    )

    if exp.cfg.restore_checkpoint_path:
        exp.restore_checkpoint()

    # dataset_shard = train.get_dataset_shard("train")
    # exp.worker_loop(dataset_shard)

    dataset_shards = _gen_sharded_datasets(exp.datasets)
    dataset_shards.foo = from_torch(TrainDataset())
    exp.worker_loop(dataset_shards)


def _gen_sharded_datasets(datasets_instance):
    dataset_shards = {
        name: train.get_dataset_shard(name)
        for name in datasets_instance.__dict__.keys()
    }

    ShardedDatasets = make_dataclass(
        "ShardedDatasets",
        [
            (name, Any, field(default_factory=lambda: None))
            for name in dataset_shards.keys()
        ],
    )

    return ShardedDatasets(**dataset_shards)


"""
        --->>> POST - INITIALIZE RAY ENVIROMENT <<<---
        --->>> PRE - INITIALIZE RAY WORKERS <<<---
"""


def main(cfg_data: dict):
    ray.init(
        logging_level="DEBUG",
        address="auto",
        runtime_env={"working_dir": "."},
    )
    # cfg_data: dict = build_configs("loop_conf")
    ray_datasets = _gen_datasets()
    datasets_config = DataConfig(
        datasets_to_split=["train"],
    )
    trainer = TorchTrainer(
        train_loop_per_worker=train_fn,
        train_loop_config=cfg_data,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        run_config=RunConfig(
            failure_config=FailureConfig(max_failures=0),
            # storage_path="/mnt/cluster_storage",
            name="experiment_name",
        ),
        # datasets={"train": from_torch(TrainDataset())},
        dataset_config=datasets_config,
        datasets=ray_datasets,
    )
    trainer.fit()


def _gen_datasets():
    cfg_data: dict = build_configs("loop_conf")
    datasets = instantiate(getattr(cfg_data["cfg_dict"], cfg_data["datasets"]))

    ray_datasets = {}
    for name in dir(datasets):
        if not name.startswith("_"):
            attr = getattr(datasets, name)
            if isinstance(attr, Dataset):
                ray_datasets[name] = from_torch(attr)

    return ray_datasets


"""
        --->>> PRE - INITIALIZE RAY ENVIROMENT <<<---
"""


def build_configs(build_type: Literal["yaml", "loop_conf"]):
    # TODO pass defaults values from either rust or previous yaml to train_config
    train_config = builds(TrainConfig, populate_full_signature=True)

    confs = {}
    for ptype in [
        "datasets",
        "models",
        "optimizers",
        "losses",
        "metrics",
        "tools",
    ]:
        class_type = inspect.signature(Experiment.__init__).parameters[ptype].annotation
        config = create_builds_config(class_type)
        confs[ptype] = config

    match build_type:
        case "yaml":
            combined_config = OmegaConf.create(
                {
                    train_config._target_: train_config,
                    **{v._target_: v for v in confs.values()},
                },
            )

            OmegaConf.save(combined_config, CONF_PATH)

            return {}

        case "loop_conf":
            cfg_dict = load_from_yaml(CONF_PATH)

            return {
                "cfg_dict": cfg_dict,
                "train_cfg": train_config._target_,
                **{k: v._target_ for k, v in confs.items()},
            }


def create_builds_config(cls):
    annotations = inspect.get_annotations(cls.__init__)
    sig = inspect.signature(cls.__init__)
    parent_defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    config_params = {}

    for name, param in annotations.items():
        if name == "self":
            continue

        if name in parent_defaults:
            config_params[name] = _handle_param(param, parent_defaults[name])

    return builds(cls, **config_params, populate_full_signature=True)


def _handle_param(param, param_default):
    """Handle the conversion of parameters into Hydra Zen configurations."""

    if isinstance(param, int | float | str | bool | type(None)):
        return param

    if isinstance(param_default, nn.Module | Dataset | Optimizer | Metric):
        return create_builds_config(param_default.__class__)

    if "tools" in f"{param.__module__}.{param.__qualname__}":
        # NOTE-dev tools we cannot cheat by recursing into any class or we get stack overflow
        # instead we inspect if tools are within the path
        # put tools classes inside some .tools. path
        return create_builds_config(param_default.__class__)

    return param_default


if __name__ == "__main__":
    gen_bindings = bool(int(os.getenv("GEN_BINDINGS", "0")))
    xp_name = "my_experiment"

    if gen_bindings:
        build_configs("yaml")

    else:
        cfg_data: dict = build_configs("loop_conf")

        # ensure experiement builds before running Ray job
        Experiment.init_module(
            cfg_dict=cfg_data["cfg_dict"],
            cfg_name=cfg_data["train_cfg"],
            models_name=cfg_data["models"],
            datasets_name=cfg_data["datasets"],
            losses_name=cfg_data["losses"],
            optimizers_name=cfg_data["optimizers"],
            metrics_name=cfg_data["metrics"],
            tools_name=cfg_data["tools"],
        )

        main(cfg_data=cfg_data)
