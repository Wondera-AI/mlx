import torch
import torch.nn.functional as F
from torchmetrics import Metric

"""Define custom LOSSES wrapped around torchmetrics"""


class HuberLossMetric(Metric):
    def __init__(self, delta=1.0):
        super().__init__()
        self.delta = delta
        self.add_state(
            "sum_huber_loss",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        error = preds - target
        is_small_error = torch.abs(error) <= self.delta
        small_error_loss = 0.5 * error**2
        large_error_loss = self.delta * (torch.abs(error) - 0.5 * self.delta)
        huber_loss = torch.where(
            is_small_error,
            small_error_loss,
            large_error_loss,
        ).sum()
        self.sum_huber_loss += huber_loss
        self.total += target.numel()

    def compute(self):
        return self.sum_huber_loss / self.total
