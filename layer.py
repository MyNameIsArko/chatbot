import torch
import torch.nn as nn
from attention import MultiHeadAttention
import config


class FeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.dim_model, config.dim_inner),
            nn.Dropout(config.dropout),
            nn.ReLU(),
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

    def forward(self, x, mem, mask):
        h = torch.cat((mem, x), dim=1)
        out = self.layer_norm2(self.attn(x, h, h, mask) + x)
        out = self.dropout(out)
        out = self.ff(out)
        return out


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
