import math

import torch
import torch.nn as nn
import config
from layer import EncoderLayer, DecoderLayer
from utils import positional_encoding


class Seq2Seq(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()

        self.mem = None
        self.mem_mask = None

        self.R = positional_encoding()

        self.embedding = nn.Embedding(num_tokens, config.dim_model)
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.R) for _ in range(config.num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.R) for _ in range(config.num_layers)])
        self.out = nn.Linear(config.dim_model, num_tokens)

    def forward(self, input, target, train=True):
        if self.mem is None:
            self.set_up_mem()

        input_mask = (input != config.pad_idx).to(torch.int).unsqueeze(1)
        target_mask = (target != config.pad_idx).to(torch.int).unsqueeze(1)

        input_total_mask = torch.cat((self.mem_mask, input_mask), dim=-1)

        if train:
            target_nopeak = torch.tril(torch.ones(1, config.max_seq_len - 1, config.max_seq_len - 1, dtype=torch.int, device=config.device))

            target_mask = target_mask & target_nopeak

        x = self.embedding(input) * math.sqrt(config.dim_model)
        y = self.embedding(target) * math.sqrt(config.dim_model)

        x = self.dropout(x)
        y = self.dropout(y)

        for i in range(config.num_layers):
            x_ = x.detach().clone()
            x = self.encoder_layers[i](x, self.mem[i], input_total_mask)
            self.add_to_memory(x_, i)

        self.add_to_memory_mask(input_mask)

        for i in range(config.num_layers):
            y = self.decoder_layers[i](y, x, input_mask, target_mask)

        out = self.out(y)
        return out

    def set_up_mem(self):
        self.mem = [torch.zeros(config.batch_size, 0, config.dim_model, device=config.device) for _ in range(config.num_layers)]
        self.mem_mask = torch.zeros(config.batch_size, 1, 0, device=config.device)

    def add_to_memory(self, x, i):
        self.mem[i] = torch.cat((self.mem[i], x), dim=1)[:, -config.max_mem_len:]

    def add_to_memory_mask(self, x_mask):
        self.mem_mask = torch.cat((self.mem_mask, x_mask), dim=-1)[:, :, -config.max_mem_len:]

    def clear_memory(self):
        self.mem = None
        self.mem_mask = None
