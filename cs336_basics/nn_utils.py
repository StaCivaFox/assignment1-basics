import torch
from einops import einsum, rearrange
from jaxtyping import Bool, Float, Int

def cross_entropy(logits: Float[torch.Tensor, "... vocab_size"], targets: Int[torch.Tensor, "..."]):
    logits = rearrange(logits, "... vocab_size -> (...) vocab_size") # logits: (batch_size, vocab_size), 
                                                                    # where batch_size can be many top-level batches of many sequences combined
    targets = rearrange(targets, "... -> (...)") # targets: (batch_size)
    logits = logits - logits.amax(dim=-1, keepdim=True)
    logits_log_sum_per_pos = torch.log(torch.exp(logits).sum(dim=-1)) # (batch_size)
    target_logits_per_pos = torch.gather(logits, dim=-1, index=targets.unsqueeze(dim=-1)).squeeze(dim=-1) # (batch_size)
    return torch.mean(logits_log_sum_per_pos - target_logits_per_pos)

def gradient_clipping(params, max_norm):
    eps = 1e-6
    grads = [p.grad for p in params if p.grad is not None]
    norm = torch.tensor(0.0, device=grads[0].device)

    for g in grads:
        norm += torch.square(g).sum()
    norm = torch.sqrt(norm)
    clip_factor = min(max_norm / (norm + eps), 1)
    for g in grads:
        g *= clip_factor