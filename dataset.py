import os.path
import pickle

from torch.utils.data import Dataset, DataLoader
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import SQuAD1
import torchtext.transforms as T
import torch.nn as nn
import config


class ChatDataset(Dataset):
    def __init__(self, chat_path, transform=None):
        self.dialogs = open(chat_path, 'r').readlines()

        self.transform = transform

    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        text = self.dialogs[idx]

        q, a = text.split('\t')
        q = q.strip()
        a = a.strip()

        if self.transform:
            q = self.transform(q)
            a = self.transform(a)

        return q, a

    def __iter__(self):
        for text in self.dialogs:
            yield text


class TokenizerTransform(nn.Module):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer

    def forward(self, x):
        return self.tokenizer(x)


def get_data():
    tokenizer = get_tokenizer('basic_english')

    data = ChatDataset('./data/dialogs.txt')

    vocab = build_vocab_from_iterator(map(tokenizer, data), specials=['<unk>', '<pad>', '<sos>', '<eos>'])
    vocab.set_default_index(config.unknown_idx)

    with open('data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    transforms = T.Sequential(
        TokenizerTransform(tokenizer),
        T.VocabTransform(vocab),
        T.Truncate(config.max_seq_len - 2),
        T.AddToken(token=config.sos_idx, begin=True),
        T.AddToken(token=config.eos_idx, begin=False),
        T.ToTensor(),
        T.PadTransform(config.max_seq_len, config.padding_idx)
    )

    data = ChatDataset('./data/dialogs.txt', transform=transforms)
    loader = DataLoader(data, batch_size=config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=12)

    num_tokens = len(vocab)
    example = data[0]

    return loader, num_tokens, vocab, example


if __name__ == '__main__':
    print(get_data())