import torch
import torch.nn.functional as F

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


if __name__ == '__main__':
    logits = torch.tensor([4.2116, 0.5297, 2.6494, 8.3370, -0.1951, 3.9561, 3.5626, 6.3043, 0.6830, 1.8885], dtype=torch.float)
    print(nucleus_search(logits))