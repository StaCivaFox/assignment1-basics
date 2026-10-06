import torch
import math
from collections.abc import Callable, Iterable

class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            # Get hyperparameters for this group
            lr, betas, eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                # States for one param include m, v, t
                state = self.state[p]
                prev_m = state.get("m", torch.zeros_like(p))
                prev_v = state.get("v", torch.zeros_like(p))
                t = state.get("t", 1)
                # Get the grad for update
                grad = p.grad.data
                # Compute adjusted learning rate for iteration t
                lr_t = lr * math.sqrt(1 - betas[1] ** t) / (1 - betas[0] ** t)
                # Apply weight decay
                p.data -= lr * weight_decay * p.data
                # Update the first moment estimate
                m = betas[0] * prev_m + (1 - betas[0]) * grad
                # Update the second moment estimate
                v = betas[1] * prev_v + ((1 - betas[1]) * torch.square(grad))
                # Apply moment-adjusted weight update
                p.data -= lr_t * m / (torch.sqrt(v) + eps)
                # Record state
                state["m"] = m
                state["v"] = v
                state["t"] = t + 1

def get_lr_cosine(
        t: int,
        max_lr: float,
        min_lr: float,
        warmup_it_num: int,
        final_it_num: int,
):
    if t < warmup_it_num:
        return t * max_lr / warmup_it_num
    elif warmup_it_num <= t <= final_it_num:
        return min_lr + 0.5 * (1 + math.cos(math.pi * (t - warmup_it_num) / (final_it_num - warmup_it_num))) * (max_lr - min_lr)
    else:
        return min_lr