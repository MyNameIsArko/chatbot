from time import sleep

import torch
import torch.cuda
from torchtext.data import get_tokenizer
import torchtext.transforms as T
import pickle

import config
from dataset import TokenizerTransform
from model import Seq2Seq
from utils import nucleus_search


def test():
    tokenizer = get_tokenizer('basic_english')

    with open('data/vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    transforms = T.Sequential(
        TokenizerTransform(tokenizer),
        T.VocabTransform(vocab),
        T.Truncate(config.max_seq_len - 2),
        T.AddToken(token=config.sos_idx, begin=True),
        T.AddToken(token=config.eos_idx, begin=False),
        T.ToTensor(),
        T.PadTransform(config.max_seq_len, config.pad_idx)
    )

    model = Seq2Seq(num_tokens=len(vocab)).to(config.device)
    model.load_state_dict(torch.load('saves/model.pth'))
    model.eval()

    while True:
        inp = input('> ')

        x = transforms(inp).unsqueeze(0).repeat(config.batch_size, 1).to(config.device)
        pred = ['<sos>']

        while pred[-1] != '<eos>':
            y = torch.tensor(vocab.lookup_indices(pred), device=config.device).unsqueeze(0).repeat(config.batch_size, 1)
            logits = model(x, y, train=False)  # [batch_size, seq_len, num_tokens]
            logits = logits[0][-1]  # [num_tokens]
            out = nucleus_search(logits)
            out = vocab.lookup_token(out)

            if out != '<eos>':
                print(out, end=' ')
            pred.append(out)
            sleep(0.05)

        print()


if __name__ == '__main__':
    test()
