import torch
import torch.nn as nn
from attention import MultiHeadAttention
import config


class Compression(nn.Module):
    def __init__(self):
        super().__init__()
        self.comp = nn.Conv1d(config.dim_model, config.dim_model, kernel_size=config.compression_rate, stride=config.compression_rate)

    def forward(self, x):
        #  x shape: [batch_size, seq_len, dim_model]
        x = x.permute(0, 2, 1)
        c_x = self.comp(x)
        return c_x.permute(0, 2, 1)


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_inner),
            nn.Dropout(config.dropout),
            nn.SiLU(),
            nn.Linear(config.dim_inner, config.dim_model),
            nn.Dropout(config.dropout)
        )
        self.layer_norm = nn.LayerNorm(config.dim_model)

    def forward(self, x):
        return self.layer_norm(self.net(x) + x)


class EncoderLayer(nn.Module):
    def __init__(self, R):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.dim_model)
        self.layer_norm2 = nn.LayerNorm(config.dim_model)

        self.dropout = nn.Dropout(config.dropout)

        self.attn = MultiHeadAttention(R)
        self.ff = FeedForward()

    def forward(self, x, mem, c_mem, mask):
        h = torch.cat((c_mem, mem, x), dim=1)
        out, dots = self.attn(x, h, h, mask, return_dots=True)
        out = self.layer_norm2(out + x)
        out = self.dropout(out)
        out = self.ff(out)
        return out, dots


class DecoderLayer(nn.Module):
    def __init__(self, R):
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(config.dim_model)
        self.layer_norm2 = nn.LayerNorm(config.dim_model)
        self.layer_norm3 = nn.LayerNorm(config.dim_model)

        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.attn1 = MultiHeadAttention(R)
        self.attn2 = MultiHeadAttention(R)

        self.ff = FeedForward()

    def forward(self, x, e, inp_mask, tgt_mask):
        z = self.layer_norm2(self.attn1(x, x, x, tgt_mask) + x)
        z = self.dropout1(z)
        out = self.layer_norm3(self.attn2(z, e, e, inp_mask) + z)
        out = self.dropout2(out)
        out = self.ff(out)
        return out
