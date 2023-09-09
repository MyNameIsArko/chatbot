import torch
import torch.nn.functional as F
import math
import config


def nucleus_search(logits):
    probs = F.softmax(logits, dim=0)
    idx = probs.argsort(descending=True)

    amount = 0
    p = 0
    for i in idx:
        p += probs[i]
        amount += 1
        if p > config.nucleus_prob:
            break

    p_idx = idx[:amount]
    b_idx = idx[amount:]

    probs[b_idx] = 0
    probs[p_idx] = F.softmax(logits[p_idx], dim=0)

    return torch.multinomial(probs, 1)


def positional_encoding():
    total_len = config.max_seq_len + config.max_mem_len + config.max_cmem_len
    position = torch.arange(total_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, config.dim_model, 2) * (-math.log(10000.0) / config.dim_model))
    pe = torch.zeros(total_len, 1, config.dim_model, device=config.device)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    return pe


def circulant_shift(x, shift):
    batch_size, num_heads, height, width = x.shape
    i = torch.arange(width, device=config.device).roll(shift).unsqueeze(0)
    i = i.flip(1).repeat(1, 2).unfold(dimension=1, size=width, step=1).flip(-1).unsqueeze(0)
    i = i.repeat(batch_size, num_heads, 1, 1)[:, :, :height]
    return x.gather(3, i)


def full_attn(q, k, v):
    dots = torch.einsum('bhid,bhjd->bhij', q, k) / math.sqrt(config.dim_model)
    attn = dots.softmax(dim=-1)
    return torch.einsum('bhij,bhjd->bhid', attn, v)
