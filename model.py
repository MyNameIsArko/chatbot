import math

import torch.nn as nn
import torch

import config


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout, max_len):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim_model, 2) * (-math.log(10000.0) / dim_model))
        pos_encoding = torch.zeros(max_len, 1, dim_model)
        pos_encoding[:, 0, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, token_embedding):
        token_embedding = token_embedding + self.pos_encoding[:token_embedding.size(0)]
        return self.dropout(token_embedding)


class Seq2Seq(nn.Module):
    def __init__(self, num_tokens, dim_model=512, num_heads=8, num_encoder_layers=6, num_decoder_layers=6,
                 dropout_p=0.1):
        super().__init__()

        self.dim_model = dim_model

        self.positional_encoding = PositionalEncoding(dim_model, dropout_p, 5000)
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(dim_model, num_heads, num_encoder_layers, num_decoder_layers,
                                          dropout=dropout_p)
        self.out = nn.Linear(dim_model, num_tokens)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.out.bias.data.zero_()
        self.out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt):
        # src shape: [batch_size, seq_len]
        # tgt shape: [batch_size, seq_len]

        tgt_mask = self.get_tgt_mask(tgt.size(1), device=tgt.device)
        src_pad_mask = self.create_pad_mask(src)
        tgt_pad_mask = self.create_pad_mask(tgt)

        src = self.embedding(src) * math.sqrt(self.dim_model)  # [batch_size, seq_len, dim_model]
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)  # [batch_size, seq_len, dim_model]

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        src = src.permute(1, 0, 2)  # [seq_len, batch_size, dim_model]
        tgt = tgt.permute(1, 0, 2)  # [seq_len, batch_size, dim_model]

        t_out = self.transformer(src, tgt, tgt_mask=tgt_mask, src_key_padding_mask=src_pad_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.out(t_out)

        return out

    def get_tgt_mask(self, size, device=torch.device('cpu')):
        return ~torch.tril(torch.ones(size, size, dtype=torch.bool, device=device))

    def create_pad_mask(self, matrix):
        return matrix == config.padding_idx


if __name__ == '__main__':
    a = Seq2Seq(40)
    mask = a.get_tgt_mask(5)
    print(mask)
