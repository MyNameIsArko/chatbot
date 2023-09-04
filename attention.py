import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import config
from utils import circulant_shift


class MultiHeadAttention(nn.Module):
    def __init__(self, R):
        super().__init__()

        assert config.dim_model // config.num_heads, 'Model hidden dimension needs to be dividable by heads number'

        self.R = R
        self.dim_k = config.dim_model // config.num_heads

        self.u = nn.Parameter(torch.randn(1, config.num_heads, 1, self.dim_k))
        self.v = nn.Parameter(torch.randn(1, config.num_heads, 1, self.dim_k))

        self.q_linear = nn.Linear(config.dim_model, config.dim_model, bias=False)
        self.k_linear = nn.Linear(config.dim_model, config.dim_model, bias=False)
        self.v_linear = nn.Linear(config.dim_model, config.dim_model, bias=False)
        self.r_linear = nn.Linear(config.dim_model, config.dim_model, bias=False)

        self.dropout = nn.Dropout(config.dropout)
        self.out = nn.Linear(config.dim_model, config.dim_model, bias=False)

    def forward(self, q, k, v, mask=None):
        seq_len = q.shape[1]
        total_len = v.shape[1]

        k = self.k_linear(k).view(config.batch_size, -1, config.num_heads, self.dim_k)
        q = self.q_linear(q).view(config.batch_size, -1, config.num_heads, self.dim_k)
        v = self.v_linear(v).view(config.batch_size, -1, config.num_heads, self.dim_k)
        r = self.r_linear(self.R[-total_len:]).view(1, total_len, -1, self.dim_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        r = r.transpose(1, 2)

        ac = torch.einsum('bhid,bhjd->bhij', q + self.u, k)
        bd = torch.einsum('bhid,bhjd->bhij', q + self.v, r)
        bd = circulant_shift(bd, -seq_len+1)

        score = (ac + bd) / math.sqrt(self.dim_k)

        if mask is not None:
            mask = mask.unsqueeze(1)
            score = score.masked_fill(mask == 0, float('-inf'))

        score = F.softmax(score, dim=-1)

        score = self.dropout(score)
        score = torch.matmul(score, v)

        concat = score.transpose(1, 2).contiguous().view(config.batch_size, -1, config.dim_model)
        out = self.out(concat)

        return out
