# NOTE these are simple example tools you can build configure and use in your trainer


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


class LRScheduler:
    def __init__(
        self,
        warmup_steps=500,
        total_steps=10000,
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self, optimizer):
        self.current_step += 1
        lr = self.lr_lambda(self.current_step)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

    def lr_lambda(self, current_step):
        if current_step < self.warmup_steps:
            return float(current_step) / float(max(1, self.warmup_steps))

        return max(
            0.0,
            float(self.total_steps - current_step)
            / float(max(1, self.total_steps - self.warmup_steps)),
        )
