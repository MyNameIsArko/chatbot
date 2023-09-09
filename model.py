import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import config
from layer import EncoderLayer, DecoderLayer, Compression
from utils import positional_encoding, full_attn


class Seq2Seq(nn.Module):
    def __init__(self, num_tokens):
        super().__init__()

        self.mem = None
        self.c_mem = None
        self.mem_mask = None

        self.R = positional_encoding()

        self.embedding = nn.Embedding(num_tokens, config.dim_model)
        self.dropout = nn.Dropout(config.dropout)
        self.compressor = Compression()
        self.encoder_layers = nn.ModuleList([EncoderLayer(self.R) for _ in range(config.num_layers)])
        self.decoder_layers = nn.ModuleList([DecoderLayer(self.R) for _ in range(config.num_layers)])
        self.out = nn.Linear(config.dim_model, num_tokens)

    def forward(self, input, target, train=True, save=True):
        if self.mem is None:
            self.set_up_mem()

        input_mask = (input != config.pad_idx).to(torch.int).unsqueeze(1)
        target_mask = (target != config.pad_idx).to(torch.int).unsqueeze(1)

        c_mem_mask = torch.ones(config.batch_size, 1, self.c_mem[0].shape[1], device=config.device)
        input_total_mask = torch.cat((c_mem_mask, self.mem_mask, input_mask), dim=-1)

        if train:
            target_nopeak = torch.tril(torch.ones(1, config.max_seq_len - 1, config.max_seq_len - 1, dtype=torch.int, device=config.device))

            target_mask = target_mask & target_nopeak

        x = self.embedding(input) * math.sqrt(config.dim_model)
        y = self.embedding(target) * math.sqrt(config.dim_model)

        x = self.dropout(x)
        y = self.dropout(y)

        dots_returns = []
        for i in range(config.num_layers):
            x_ = x.detach().clone()
            x, dots = self.encoder_layers[i](x, self.mem[i], self.c_mem[i], input_total_mask)
            dots_returns.append(dots)
            if save:
                self.add_to_memory(x_, i)

        if save:
            self.add_to_memory_mask(input_mask)

        for i in range(config.num_layers):
            y = self.decoder_layers[i](y, x, input_mask, target_mask)

        out = self.out(y)

        if not train:
            return out

        aux_loss = 0

        for i in range(config.num_layers):
            attn = self.encoder_layers[i].attn
            attn.k_linear.weight.detach_()
            attn.v_linear.weight.detach_()

            cmem_k = attn.k_linear(self.c_mem[i])
            cmem_v = attn.v_linear(self.c_mem[i])

            cmem_k = cmem_k.view(config.batch_size, -1, config.num_heads, attn.dim_k)
            cmem_v = cmem_v.view(config.batch_size, -1, config.num_heads, attn.dim_k)

            cmem_k = cmem_k.transpose(1, 2)
            cmem_v = cmem_v.transpose(1, 2)

            q, k, v = dots_returns[i]

            old_mem_range = slice(-self.mem[i].shape[1]-config.max_seq_len, -config.max_seq_len)  # get memory part from dots
            old_mem_k = k[:, :, old_mem_range].clone()
            old_mem_v = v[:, :, old_mem_range].clone()

            q = q.detach()
            old_mem_k = old_mem_k.detach()
            old_mem_v = old_mem_v.detach()

            aux_loss += F.mse_loss(
                full_attn(q, old_mem_k, old_mem_v),
                full_attn(q, cmem_k, cmem_v)
            )

        aux_loss *= config.compression_loss_weight / config.num_layers

        return out, aux_loss

    def set_up_mem(self):
        self.mem = [torch.zeros(config.batch_size, 0, config.dim_model, device=config.device) for _ in range(config.num_layers)]
        self.c_mem = [torch.zeros(config.batch_size, 0, config.dim_model, device=config.device) for _ in range(config.num_layers)]
        self.mem_mask = torch.zeros(config.batch_size, 1, 0, device=config.device)

    def add_to_memory(self, x, i):
        self.mem[i] = torch.cat((self.mem[i], x), dim=1)
        all_mem_size = self.mem[i].shape[1]
        if all_mem_size > config.max_mem_len:
            old_mem = self.mem[i][:, :all_mem_size-config.max_mem_len]
            self.mem[i] = self.mem[i][:, -config.max_mem_len:]
            old_c_mem = self.compressor(old_mem)
            self.c_mem[i] = torch.cat((self.c_mem[i], old_c_mem), dim=1)[:, -config.max_mem_len:]

    def add_to_memory_mask(self, x_mask):
        self.mem_mask = torch.cat((self.mem_mask, x_mask), dim=-1)[:, :, -config.max_mem_len:]

    def clear_memory(self):
        self.mem = None
        self.mem_mask = None
        self.c_mem = None
