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
from ray.runtime_env import RuntimeEnv
from ray.train import CheckpointConfig, FailureConfig, RunConfig
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


def gen_datasets():
    cfg_data: dict = build_configs("loop_conf")
    datasets = instantiate(getattr(cfg_data["cfg_dict"], cfg_data["datasets"]))

    ray_datasets = {}
    for name in dir(datasets):
        if not name.startswith("_"):
            attr = getattr(datasets, name)
            if isinstance(attr, Dataset):
                ray_datasets[name] = from_torch(attr)

    return ray_datasets


def gen_sharded_datasets(datasets_instance):
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

    dataset_shards = train.get_dataset_shard("train")
    dataset_shards = TrainDataset()

    # print(:)
    # for shard in dataset_shards.iter_batches():
    #     print("Shard content:", shard)
    # dataset_shards = gen_sharded_datasets(exp.datasets)
    exp.worker_loop(cast(Dataset, dataset_shards))


def main():
    ray.init(
        logging_level="DEBUG",
        address="auto",
        runtime_env={"working_dir": "."},
    )
    cfg_data: dict = build_configs("loop_conf")
    # ray_datasets = gen_datasets()

    trainer = TorchTrainer(
        train_loop_per_worker=train_fn,
        train_loop_config=cfg_data,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=False),
        run_config=RunConfig(
            failure_config=FailureConfig(max_failures=0),
            # storage_path="/mnt/cluster_storage",
            name="experiment_name",
        ),
        datasets={"train": from_torch(TrainDataset())},
        # dataset_config=ray.train.DataConfig(
        #     datasets_to_split=["train"],
        # ),
        # datasets=ray_datasets,
    )
    trainer.fit()


"""

        --- PRE-INITIALIZE RAY ENVIROMENT ---

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
    # params = inspect.signature(cls.__init__).parameters
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
            param_default = parent_defaults[name]
            # config_params[name] = _handle_param(param, parent_defaults[name])
            if isinstance(param, int | float | str | bool | type(None)):
                config_params[name] = param

            elif isinstance(param_default, nn.Module | Dataset | Optimizer | Metric):
                # TODO if param_default is nn.Module
                return create_builds_config(param_default.__class__)

            elif "tools" in f"{param.__module__}.{param.__qualname__}":
                # NOTE-dev tools we cannot cheat to recurse into, instead we inspect if tools are within the path
                # put tools classes inside some .tools. path
                return create_builds_config(param_default.__class__)

            else:
                raise ValueError(
                    f"Unsupported type in create_builds_config God function: {param_default}"
                )

    return builds(cls, **config_params, populate_full_signature=True)


if __name__ == "__main__":
    gen_bindings = bool(int(os.getenv("GEN_BINDINGS", "0")))
    xp_name = "my_experiment"

    if gen_bindings:
        build_configs("yaml")

    else:
        main()

# def _extract_override_params(instance, cls):
#     sig = inspect.signature(cls.__init__)
#     override_params = {}

#     for name, param in sig.parameters.items():
#         if param.default is inspect.Parameter.empty and hasattr(instance, name):
#             attr_value = getattr(instance, name)

#             if not isinstance(attr_value, torch.nn.Parameter):
#                 override_params[name] = attr_value

#     return override_params


# NOTE 2
# def create_builds_config(cls):
#     params = inspect.signature(cls.__init__).parameters
#     parent_defaults = _get_default_args(cls)
#     config_params = {}

#     for name, param in params.items():
#         if name == "self":
#             continue

#         config_params[name] = _handle_param(param, parent_defaults)

#     return builds(cls, **config_params, populate_full_signature=True)


# def _get_default_args(cls):
#     sig = inspect.signature(cls.__init__)
#     return {
#         k: v.default
#         for k, v in sig.parameters.items()
#         if v.default is not inspect.Parameter.empty
#     }


# def _handle_param(param, parent_defaults):
#     # Check for primitive types first
#     if isinstance(param.default, int | float | str | bool | type(None)):
#         return param.default

#     if isinstance(param.default, nn.Module):
#         nested_cls = param.default.__class__
#         override_params = _extract_override_params(param.default, nested_cls)
#         return builds(nested_cls, **override_params, populate_full_signature=True)

#     elif inspect.isclass(param.default) and issubclass(param.default, nn.Module):
#         nested_cls = param.default
#         return create_builds_config(nested_cls)

#     elif isinstance(param.default, torch.optim.Optimizer):
#         optimizer_cls = param.default.__class__
#         override_params = _extract_override_params(param.default, optimizer_cls)
#         return builds(optimizer_cls, **override_params, populate_full_signature=True)

#     elif isinstance(param.default, Dataset):
#         dataset_cls = param.default.__class__
#         return builds(dataset_cls, populate_full_signature=True)

#     # Handle only one level of nesting for custom classes
#     elif inspect.isclass(param.default) and not issubclass(
#         param.default,
#         nn.Module | torch.optim.Optimizer | Dataset | type,
#     ):
#         nested_cls = param.default
#         return builds(nested_cls, populate_full_signature=True)

#     # Ensure we are not handling already instantiated objects
#     elif isinstance(param.default, object) and not inspect.isclass(param.default):
#         nested_cls = param.default.__class__
#         return builds(nested_cls, populate_full_signature=True)

#     elif param.default is inspect.Parameter.empty and param.name in parent_defaults:
#         return parent_defaults[param.name]

#     else:
#         return param.default


# def _extract_override_params(instance, cls):
#     sig = inspect.signature(cls.__init__)
#     override_params = {}

#     for name, param in sig.parameters.items():
#         if param.default is inspect.Parameter.empty and hasattr(instance, name):
#             attr_value = getattr(instance, name)

#             if not isinstance(attr_value, torch.nn.Parameter):
#                 override_params[name] = attr_value

#     return override_params

# NOTE 1
# def create_builds_config(cls):
#     params = inspect.signature(cls.__init__).parameters
#     parent_defaults = _get_default_args(cls)
#     config_params = {}

#     for name, param in params.items():
#         if name == "self":
#             continue

#         config_params[name] = _handle_param(param, parent_defaults)

#     return builds(cls, **config_params, populate_full_signature=True)


# def _get_default_args(cls):
#     sig = inspect.signature(cls.__init__)
#     return {
#         k: v.default
#         for k, v in sig.parameters.items()
#         if v.default is not inspect.Parameter.empty
#     }


# def _handle_param(param, parent_defaults):
#     if isinstance(param.default, nn.Module):
#         nested_cls = param.default.__class__
#         override_params = _extract_override_params(param.default, nested_cls)

#         return builds(nested_cls, **override_params, populate_full_signature=True)

#     elif inspect.isclass(param.default) and issubclass(param.default, nn.Module):
#         nested_cls = param.default

#         return create_builds_config(nested_cls)

#     elif param.default is inspect.Parameter.empty and param.name in parent_defaults:
#         return parent_defaults[param.name]

#     else:
#         return param.default


# def _extract_override_params(instance, cls):
#     sig = inspect.signature(cls.__init__)
#     override_params = {}

#     for name, param in sig.parameters.items():
#         if param.default is inspect.Parameter.empty and hasattr(instance, name):
#             attr_value = getattr(instance, name)

#             if not isinstance(attr_value, torch.nn.Parameter):
#                 override_params[name] = attr_value

#     return override_params


# pass

# NOTE
# - SPLIT ABOVE IS WRITE YAML
# - SPLIT BELOW IS INIT CFG

# paths = setup_experiment_paths(xp_name)
# paths = []

# test_conf = load_from_yaml("conf/conf.yaml")

# m = MyModule.init_module(
#     train_config._target_,
#     dnn_config._target_,
#     optimizer_config._target_,
# )
# print(m.model)
# print(m.optimizer)
# print(m.cfg.batch_size)

# a = instantiate(test_conf)
# print(a.)

# train_cfg: TrainConfig = instantiate(getattr(test_conf, train_config._target_))
# model: DNN = instantiate(getattr(test_conf, dnn_config._target_))
# optimizer: Adam = instantiate(
#     getattr(test_conf, optimizer_config._target_),
#     params=model.parameters(),
# )

# cfg = OmegaConf.merge(train_cfg, model)
# train_config._target_
# cfg = OmegaConf.create(
#     {
#         "train": train_cfg,
#         "dnn": model,
#     },
# )
# print(cfg)
# cfg.

# print("foo", model)
# print(type(model))
# # print("yooo", test_conf.__main__.TrainerConfig)
# print("bar", dir(model))
# print(getattr(test_conf, dnn_config._target_))
# # print("bar", DNN(input_size=10, output_size=5))
# # print("ree", BlockEg(input_size=10))
# # print(repr(d))

# print(model.input_size)
# print("Nested block:", model.block)

# NOTE if seperate desired
# save_as_yaml(dnn_config, "conf/dnn_config.yaml")
# save_as_yaml(train_config, "conf/train_config.yaml")

# combined_config_str = f"{dnn_config._target_}:\n{OmegaConf.to_yaml(dnn_config)}\n{train_config._target_}:\n{OmegaConf.to_yaml(train_config)}"

# Save the combined configuration string to a file
# with open("conf/conf.yaml", "w") as f:
#     f.write(combined_config_str)

# combined_config = OmegaConf.merge(dnn_config, train_config)

# ZenBuilds_DNN = builds(DNN, populate_full_signature=True)
# TrainBuilds_Config = builds(TrainConfig)
# save_as_yaml(config=ZenBuilds_DNN, f="dnn_config.yaml")
# save_as_yaml(config=TrainBuilds_Config, f="train_config.yaml")

# Save the combined configuration
# Load the generated YAMLs
# dnn_config = OmegaConf.load("conf/dnn_config.yaml")
# train_config = OmegaConf.load("conf/train_config.yaml")

# # cfg = train_config.merge_with({"DNN", dnn_config})
# # save_as_yaml(dnn_config, "conf/bar.yaml")

# # Combine the configurations under a single root
# config = load_from_yaml("src/conf.yaml")
# print(config)

# dnn = dnn_config(input_size=10, output_size=5)
# print(dnn)
# print(type(dnn))

# obj = load_from_yaml("src/conf.yaml")
# print(obj)
# print(type(obj))

# Load configuration from combined YAML file

# Ensure the loaded configuration is type-safe
# train_config: TrainConfig = instantiate(TrainBuilds_Config, **config.train)

# print(config.train)

# x = train_config.dataset

# config.mlp
# dnn_instance = instantiate(ZenBuilds_DNN, **config.mlp)
# dnn_instance.foo()

# This is how you can instantiate the class with custom

