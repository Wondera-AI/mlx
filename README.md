# Distributed Neural Network Experiment Manager

Simply train neural networks and manage experiments on a Ray clusters in PyTorch. This library easily distributes model training using Torch's Distributed Data Parallel, whilst using Ray to automatically manage actors, tasks, heads and workers.

## Python Packages Used:
- **Ray Train**: creates train job in Ray enviroment
- **Ray Data**: distributes datasets and loaders
- **PyTorch**: express neural networks and original datasets
- **TorchMetrics**: distributed loss and metrics allreduce operations across workers
- **HydraZen**: dynamic configuration builder with yaml config as source of truth

## SDK Features:
- Distributed and autoscaled on a both local and remote Ray clusters
- Simple and lightweight trainer inspired by Meta's Flashy
- Hierarchical configurability using HydraZen
- Static-ish type-forward experiment builder to leverage auto-generated configs
- Native PyTorch model and dataset expressability
- Easy extensability of custom losses, metrics and tools


## Companion Client
To manage, monitor and track experiments a custom Rust-based client (**mlx**) is used. The client is designed as follows.

**mlx**
- **train**
    - **new** [name]: creates a new training experiment folder from this template
    - **bind**: automatically generate the configuration yaml from the experiment definition
    - **run**: locally run the training experiment to test prior to launchin
    - **launch**: run the training experiment on a remote Ray cluster
        - [--ray-address]: address defineable also as an enviroment variable RAY_ADDRESS
        - [--prepare-batches]: create and save Ray datasets that map batches according to a user defined Dataset (see below) prior to the model being trained for greater performance on each batch iteration.
- **xp**
    - **ls**: lists the experiments run remotely
    - **logs** [name] [run]: streamed stdout of remote experiment jobs
    - **board** [name] [run]: live tensorboards of a particular experiment
    - **ray**: ray cluster monitor to view jobs, logs and cluster specific metrics
- **data**
    - **show**: displays filesystem structure of shared NFS
    - **new**: creates a new arbitrary data job folder from another template
    - **run**: run data job locally
    - **launch** [--ray-address]: run data job on a remote Ray cluster
    - **rm**: remove a folder from the shared NFS

## Get Started

Download the **mlx** client.

```bash
curl https://gist
```
Create a new experiment in a directory of choice. This will download the mlx template for our experiment 
```python
mlx new [experiment_name]
```

### Type Safety
This library encourages typing and leverages this typing for automation, discoverability and safety.

```python
class Foo
    def __init__(self, arg1: int = 3):
        pass
```
- By annotating our arguments with types and defaults we can leverage HydraZen to automatically build the configurations all the way down a stack of classes. 
- So that you can code away and automatically generation your **configuration yaml** (the source of truth)

We will see later how this structure comes together at the experiment level.

### Datasets
Write native PyTorch Datasets then define how to handle batching of the Ray data shards with **ray_map_batches**.

```python
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
        self.labels = torch.randn(1000, 10)  # Adjust labels to match output shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    @staticmethod
    def ray_map_batches(batch):
        # NOTE please read all of this !! 
        # Ray handles data in many ways and requires either a custom collate_fn or map_batches
        # our chosen solution is to define normal pytorch dataset and enforce declaration of this method

        # NOTE enables interchangeable usage of this method to map batches:
        #   1 - during training (available): method used as the collate_fn of ray.train.data.Datasets.iter_map_batches
        #   2 - prior to training (tbd): method used to create, map and save the Ray Dataset

        # NOTE path 2 is more performant but for now change the prefetch_batches cfg the iter_torch_batches
        # https://docs.ray.io/en/latest/data/api/

        batch = batch["item"]  # and yes unfornutaley you must retrieve from "item" key first (Ray from_torch handles this badly)

        # but then you get access to the __getitem__ return signature
        data = torch.stack([row[0].clone().detach() for row in batch])
        labels = torch.stack([row[1].clone().detach() for row in batch])

        return {"inputs": data, "label": labels}
```

No further class type requires an enforced method signature than datasets.


### Models
Now we get to the fun part and the part I hope most of you will spend your time in. Expressing neural networks. This is also where the type annotations and defaults can appear quirky but there is a method to this.

```python
class BlockEg(nn.Module):
    def __init__(self, input_size: int = 180):  # important to define default values
        super().__init__()
        self.layer1 = nn.Linear(input_size, 32)

    def forward(self, x):
        return self.layer1(x)


class DNN(nn.Module):
    def __init__(
        self,
        input_size: int = 180,
        output_size: int = 10,
        layer_widths: tuple[int, ...] = (5, 10, 5),
        block: BlockEg = BlockEg(),  # so that up the stack the default values are set
        fc: nn.Linear = nn.Linear(32, 10),  # initialize with whatever - we'll override
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layer_widths = layer_widths
        self.block = block  # not overriden as default set by user
        self.fc = nn.Linear(32, output_size)  # override as defaults aren't set by Torch

    def forward(self, x):
        x = self.block(x)
        return self.fc(x)
```
- By defininig default at the leaves, when you initialize defaults up the stack you dont need to pass arguments
- You only need to pass arguments into the defaults of nn libraries that didn't defined any
- But in the __ init __ you can obviously override them with runtime values

**But why default ?**
- HydraZen exposes a **builds** method that creates the config of any class via the __ init __ signature
- This allows the entire model tree to be visible, providing we can inform Hydra of the instance to recurse into
- Building configurations happens both when:
    1. yaml binding are autogenerated
    2. at runtime to instantiate the models with the yaml configuration
- This structure, although verbose, allows for many areas of increased developer efficiency

(more on configurations and yamls later)

### Experiments
Now that we have our data and model we need to write an experiment. An experiment is quite simply a class with a user defined worker_loop method. This worker loop is the method each worker will run. 

```python
class Experiment(BaseModule):
    def worker_loop(self, dataset_shards):
        # do whatever here
```

But in reality training a neural network requires generalizeable actions. Our experiment class takes a configuriation class (TrainConfig), models, datasets, losses, optimizers, metrics and tools.

```python
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
    ):
        super().__init__(
            cfg=cfg,
            models=models,
            datasets=datasets,
            optimizers=optimizers,
            losses=losses,
            metrics=metrics,
            tools=tools,
        )
```
Define as many datasets as we want, but ensure to assign arguments to self.

```python
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
```
Define as many models as you want, though ensure to map each trainable model to a given optimizer to inform how the optimizer needs to initialize. Therefore we define a required **map_params_to_optimizers** field mapping the names of the model and optimizer from Models and Optimizer classes respectively.
```python
class Models:
    def __init__(
        self,
        dnn: DNN = DNN(),
    ) -> None:
        self.dnn = dnn
        # need to inform ray of model to optimizer mapping for instantiation
        self.map_params_to_optimizers = {"dnn": "adam"}

class Optimizers:
    def __init__(
        self,
        adam: Adam = Adam(DUMMY_PARAMS),  # have to pass dummy params to compile
    ) -> None:
        self.adam = adam
```
Notice that optimizer require parameters to be initialized but since this default is only required for configuration building purposes use the provided DUMMY_PARAMS global static variable. **This will be replaced by the mapped model params at runtime.**

Finally define as many Metrics, Losses and Tools as you may want
```python
class Losses:
    def __init__(
        self,
        loss1: tm.MeanSquaredError = tm.MeanSquaredError(),
        loss2: custom_losses.HuberLossMetric = custom_losses.HuberLossMetric(),
    ) -> None:
        self.loss1 = loss1
        self.loss2 = loss2

class Metrics:
    def __init__(
        self,
        mae: tm.MeanAbsoluteError = tm.MeanAbsoluteError(),
        mbd: custom_metrics.MeanBiasDeviation = custom_metrics.MeanBiasDeviation(),
    ) -> None:
        self.mae = mae
        self.mbd = mbd

class Tools:
    def __init__(
        self,
        early_stoppage: EarlyStopping = EarlyStopping(),  # callable class
        lr_scheduler: LRScheduler = LRScheduler(),  # method driven class
        # NOTE add as many as you want ...
    ):
        self.early_stoppage = early_stoppage
        self.lr_scheduler = lr_scheduler
```
Please notice we must use torchmetrics as it allows for proper distributed loss and metric management across workers. Torchmetrics will enable metric computation across worker values.

You can easily define custom Metrics, Losses...
```python
import torch
from torchmetrics import Metric

"""Define custom METRICS wrapped around torchmetrics"""
class MeanBiasDeviation(Metric):
    def __init__(self):
        super().__init__()
        self.add_state(
            "sum_bias_deviation",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        bias_deviation = (preds - target).sum()
        self.sum_bias_deviation += bias_deviation
        self.total += target.numel()

    def compute(self):
        return self.sum_bias_deviation / self.total
```
...and tools (nothing to inherit)
```python
class EarlyStopping:
    def __init__(
        self,
        patience=11,
        min_delta=0,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            return True

        return False
```
#### Notes on preparing your model. 
To distribute our model correctly and synchronize across workers we must wrap our model in **ray.train.torch.prepare_model** within each worker. This method is Ray's wrapper of Torch's DDP. We also must set non-trainable models to the correct device.
```python
self.models.dnn = prepare_model(self.models.dnn) # trainable model preparation

self.models.resnet = self.models.resnet.to(get_device()) # non-trainable model preparation
```

#### Notes on datashards
The Experiment manager's worker_loop method defines the only non-typed argument. This is because rather than messy string concatenation a class is build at runtime to expose the same dataset arguments defined in your Datasets above.
- So you can access **datashards.train** etc...
- To iterate over batches use datashards.train.iter_torch_batches 
- Remember you can access dataset arguments from **self.datasets.train.batch_size**

A helper method from the BaseModule of the Experiment to automatically build your dataloaders (as many as there are configured Datasets) is provided.
```python
train_loader, _, _ = self.generate_loaders(dataset_shards=dataset_shards)
```

#### Under The Hood
Please refer to:
- main.py: for how configurations are build, Ray is initialized and how the worker_loop is called
- src.lib.base_classes.py: for how the experiment is initialized, as well as helper methods for model training

#### Complete Experiment
A simple end to end experiment might look like this.

```python
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
    ):
        super().__init__(
            cfg=cfg,
            models=models,
            datasets=datasets,
            optimizers=optimizers,
            losses=losses,
            metrics=metrics,
            tools=tools,
        )
        # TODO as per earlier you can override the default values here if you want runtime specific values
        # in this case we want to override our LRScheduler to leverage the dataset runtime values
        self.tools.lr_scheduler.total_steps = (
            len(self.datasets.train)
            // self.datasets.train.batch_size
            * self.cfg.num_epochs
        )

    # NOTE dataset_shards: not strictly typed but you can access all "Datasets" attributes
    def worker_loop(self, dataset_shards):
        # prepare trainable model for train synchronizations - wrapper for DDP
        self.models.dnn = prepare_model(self.models.dnn)
        # prepare non-trained model for inference
        self.models.resnet = self.models.resnet.to(get_device())

        train_loader, _, _ = self.generate_loaders(dataset_shards=dataset_shards)

        for epoch in range(self.cfg.start_epoch, self.cfg.num_epochs):

            self.train_steps(dataloader=train_loader, epoch=epoch)

            dummy_val_loss = 69
            if self.tools.early_stoppage(dummy_val_loss):
                break

            self.save_checkpoint(
                epoch=epoch,
                metrics={},
            )

    def train_steps(self, dataloader: DataLoader, epoch: int):
        self.reset_all_metrics_and_losses()
        for batch_idx, batch in enumerate(dataloader):
            x = batch["inputs"]
            y = batch["label"]

            output = self.models.dnn(x)

            # local loss compute and gradient calculation
            loss = self.losses.loss1(output, y)
            loss.backward()

            # updating losses and metrics as desired
            # NOTE: you will get warnings if you save metrics & losses without updating them
            self.losses.loss1.update(output, y)
            self.metrics.mae.update(output, y)

            self.save_metrics_and_losses(
                batch_idx=batch_idx,
                epoch=epoch,
            )

            self.optimizers.adam.step()
            self.optimizers.adam.zero_grad()

            self.tools.lr_scheduler.step(self.optimizers.adam)
```

### Yaml Configuration

As you write your models, datasets, losses etc. you will want to adjust experiment configurations. The simplest way to do this is to use the client's train bind command when ready:
```bash
mlx train bind
```
This will automatically build your configuration for the entire experiment and output a hierarchical **conf.yaml** in the src folder. Adjust and run your experiment accordingly, leveraging the typed interfaces across your experiment.

Inside the experiment **self.cfg** refers to the TrainConfig dataclass defined inside **src.lib.configs** this can be altered and provides global configurations used across the training experiment.

> Avoid removing fields from current configuration as they are used by the SDK

Elsewhere your configuration is accessible from your experiment classes (i.e. **self.datasets.train.batch_size**)

Suppose a default to an argument is not provided. The yaml will be created with ???

```yaml
src.experiment.Optimizers:
  _target_: src.experiment.Optimizers
  adam:
    _target_: torch.optim.adam.Adam
    params: ???
```
When you try running this experiment any non-defaulted argument that does not get replaced with an initial value will error. The only exception is the **param** field of Optimizers as it is overriden by the mapped model parameters as the experiment gets initialized.

Our full experiment conf.yaml in our above example will look something like this

```yaml
src.lib.configs.TrainConfig:
  _target_: src.lib.configs.TrainConfig
  name: "experiment 1"
  restore_checkpoint_path: null
  start_epoch: 0
  num_epochs: 10
  start_workers: 3
  prefetch_batches: 2
src.experiment.Datasets:
  _target_: src.experiment.Datasets
  train:
    _target_: src.data.TrainDataset
    batch_size: 32
    unused_data_arg: 0
  val:
    _target_: src.data.ValDataset
    batch_size: 32
  test:
    _target_: src.data.TestDataset
    batch_size: 32
src.experiment.Models:
  _target_: src.experiment.Models
  dnn:
    _target_: src.model.DNN
    input_size: 180
    output_size: 10
    layer_widths:
    - 5
    - 10
    - 5
    block:
      _target_: src.model.BlockEg
      input_size: 180
    fc:
      _target_: torch.nn.modules.linear.Linear
      in_features: 1
      out_features: 1
      bias: true
      device: null
      dtype: null
src.experiment.Optimizers:
  _target_: src.experiment.Optimizers
  adam:
    _target_: torch.optim.adam.Adam
    params: ???
    lr: 0.001
    betas:
    - 0.9
    - 0.999
    eps: 1.0e-08
    weight_decay: 0.0
    amsgrad: false
    foreach: null
    maximize: false
    capturable: false
    differentiable: false
    fused: null
src.experiment.Losses:
  _target_: src.experiment.Losses
  loss1:
    _target_: torchmetrics.regression.mse.MeanSquaredError
    squared: true
    num_outputs: 1
  loss2:
    _target_: src.losses.HuberLossMetric
    delta: 1.0
src.experiment.Metrics:
  _target_: src.experiment.Metrics
  mae:
    _target_: torchmetrics.regression.mae.MeanAbsoluteError
  mbd:
    _target_: src.metrics.MeanBiasDeviation
src.experiment.Tools:
  _target_: src.experiment.Tools
  early_stoppage:
    _target_: src.tools.EarlyStopping
    patience: 11
    min_delta: 0
  lr_scheduler:
    _target_: src.tools.LRScheduler
    warmup_steps: 500
    total_steps: 10000
```
All generated for us automatically.

> A thing to note is that our Ray cluster will autoscale according to GPU utilization and available workers, so **start_workers** are a user's guess but the rest will be handled by Ray. Similarly jobs will be scheduled and auto-balanced by Ray accross all ML experiments happening concurrently.

### Monitoring
After running your experiment locally with
```bash
mlx train run
```
You can initialize your job to a remote cluster with
```bash
mlx train lauch
```
Providing the Ray cluster address either as a flag **--address** or an enviroment variable **RAY_ADDRESS**

Running locally defaults to show the stdout of your workers, whilst launching remotely will default to running in **detached** mode. This means you will not be able to see the stdout logs from your terminal.

However you will be able to monitor your job through:
1. Ray dashboard using ```mlx xp ray```
2. Live tensorboard using ```mlx xp board [experiment_name] [run_version]```
3. Streamed live logs using ```mlx xp logs [experiment_name] [run_version]```

> Note you will be able to view the experiments that ran and their states using ```mlx xp ls```

### Data Jobs
Data jobs come in two variants
1. Arbitrary filesystem preparation
2. Ray dataset batch transformations

The former will be created using ```mlx data``` run / launch commands. These jobs enable filesystem storage to be mounted onto the Ceph cluster. This filesystem is shared across an SSD pool across all GPU workers and is ideal storage for training

The latter will be created (tbd) by calling the remote training job as follows
```bash
mlx train launch --prepare-batches
```
such that a user defined ray_map_batches method on their datasets, will be transformed into a Ray dataset according to this method prior to the model being trained. These Ray datasets will be used by the experiment at runtime for greater performance on the batch iterations. These datasets will automatically get garbage collected following a specific policy.


## Roadmap
**Immediate Priorities**
- [ ] Client build
- [ ] Connecting all supplied compute nodes to Kube cluster
- [ ] Connecting to live Ray cluster

**Future Enhancements**
- [ ] Prepare batches using Ray dataset
- [ ] Model inference serving
- [ ] Hyperparameter tuning
- [ ] Model parallelism (DeepSpeed Zero-3 and FSDP)
- [ ] JAX support
- [ ] Gradient flow analysis and optimize dynamics


## Improvements
Ray is a great technology for distributed computing and has a lot of great features to support distributed ML model training. However future versions of this library should be targeted with greater type and memory safety available through Rust. The proposed end state solution could look like:

- A Rayon-tokio Ray-esque cluster scheduler to manage jobs, actors, tasks and workers
- A custom Rust Burn based backend for distributed training
- XLA bindings on this backend to natively support CUDA and provide user level JAX bindings
- Rust data instantiation and loading, iterfaceable in Python at runtime with the same memory address
- Fully customizeable experiment dashboarding tools
- Transfer learning surgery enviroment
- Potential to extend to GUI-based model experiment expression

> Any other suggestions or improvements please open an issue on this repo with the detailed request.
