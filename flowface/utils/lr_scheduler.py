from oneflow.optim import lr_scheduler
from oneflow.optim.lr_scheduler import _LRScheduler
# from torch.optim.lr_scheduler import _LRScheduler


class PolyScheduler(_LRScheduler):
    def __init__(self, optimizer, base_lr, max_steps, warmup_steps, last_step=-1):
        self.base_lr = base_lr
        self.warmup_lr_init = 0.0001
        self.max_steps: int = max_steps
        self.warmup_steps: int = warmup_steps
        self.power = 2
        self.last_step = last_step
        super(PolyScheduler, self).__init__(optimizer, -1, False)

    def get_warmup_lr(self, step):
        # alpha = float(step) / float(self.warmup_steps)
        # return self.base_lr * alpha 

        alpha = float(self.last_step) / float(self.warmup_steps)
        return self.base_lr * alpha 

    def get_lr(self, base_lr, step):
        if self.last_step == -1:
            return self.warmup_lr_init 
        if step < self.warmup_steps:
            return self.get_warmup_lr(step)
        else:
            alpha = pow(
                1
                - float(step - self.warmup_steps)
                / float(self.max_steps - self.warmup_steps),
                self.power,
            )
            return base_lr * alpha 

            

if __name__ == "__main__":
    import oneflow as flow
    MAX_STEPS = 10000
    model = flow.nn.Linear(2, 2)
    optimizer = flow.optim.SGD(model.parameters(), 0.1, 0.9, 5e-4)
    lr_scheduler = PolyScheduler(optimizer, 0.1, MAX_STEPS, 1000, -1)
    lr = []
    for i in range(MAX_STEPS):
        lr_scheduler.step()
        lr.append(lr_scheduler.get_last_lr()[0])

    from matplotlib import pyplot as plt
    plt.plot(lr)
    plt.savefig("lr.jpg")