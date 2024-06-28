from dataclasses import dataclass


@dataclass
class TrainConfig:
    name: str
    restore_checkpoint_path: str
    start_epoch: int
    num_epochs: int
    start_workers: int
    prefetch_batches: int
