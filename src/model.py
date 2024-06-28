from torch import nn


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
