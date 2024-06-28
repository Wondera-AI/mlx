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
