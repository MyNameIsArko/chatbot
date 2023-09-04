import os.path
import pickle

import torch
from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pickle
import torchtext.transforms as T
import torch.nn as nn
import config
import os


class ChatDataset(Dataset):
    def __init__(self, chat_path, transform=None):
        with open(chat_path, 'rb') as file:
            self.dialogs = pickle.load(file)

        self.transform = transform

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        dialog = self.dialogs[idx]

        if self.transform:
            dialog = [self.transform(d) for d in dialog]

        while len(dialog) < config.num_dialogs:
            dialog.append(torch.tensor([config.sos_idx, config.eos_idx] + [config.pad_idx for _ in range(config.max_seq_len - 2)]))

        if len(dialog) > config.num_dialogs:
            start_idx = torch.randint(0, len(dialog) - config.num_dialogs, (1,))
            dialog = dialog[start_idx:start_idx+config.num_dialogs]

        out = tuple(dialog)
        return out

    def __iter__(self):
        for dialog in self.dialogs:
            for text in dialog:
                yield text


class TokenizerTransform(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, x):
        return self.tokenizer(x)


def get_data():
    tokenizer = get_tokenizer('basic_english')

    data = ChatDataset('./data/dialogs.pkl')

    vocab = build_vocab_from_iterator(map(tokenizer, data), specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    vocab.set_default_index(config.unk_idx)

    with open('data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    transforms = T.Sequential(
        TokenizerTransform(tokenizer),
        T.VocabTransform(vocab),
        T.Truncate(config.max_seq_len - 2),
        T.AddToken(token=config.sos_idx, begin=True),
        T.AddToken(token=config.eos_idx, begin=False),
        T.ToTensor(),
        T.PadTransform(config.max_seq_len, config.pad_idx)
    )

    data = ChatDataset('./data/dialogs.pkl', transform=transforms)
    print(f'Workers amount: {os.cpu_count()}')
    loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=os.cpu_count())

    num_tokens = len(vocab)

    return loader, num_tokens, vocab
