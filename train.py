import os

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

import config
from dataset import get_data
from model import Seq2Seq

import wandb


def get_ratio(currect_epoch, total_epoch):
    return 1 - (currect_epoch / total_epoch) / 2  # To stop at 0.5


def train():
    print(f'Device used: {config.device}')

    loader, num_tokens, vocab, example = get_data()

    model = Seq2Seq(num_tokens=num_tokens).to(config.device)

    opt = optim.AdamW(model.parameters(), lr=config.lr)
    warmup_scheduler = optim.lr_scheduler.LinearLR(opt, 0.1, 1, config.num_warmup)
    scheduler = optim.lr_scheduler.StepLR(opt, 100, 0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=config.padding_idx)

    os.makedirs('saves', exist_ok=True)
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    wandb.init(
        project='chatbot',
        name=date,
        config={
            'learning_rate': config.lr,
            'num_epochs': config.num_epochs,
            'max_seq_len': config.max_seq_len,
            'batch_size': config.batch_size,
            'num_warmup': config.num_warmup
        },
        sync_tensorboard=True
    )
    writer = SummaryWriter(f'logs/chatbot_{date}')
    writer.add_graph(model, [x.unsqueeze(0).to(config.device) for x in example])

    t = tqdm(range(1, config.num_epochs + 1))

    global_step = 0

    for epoch in t:
        total_loss = 0

        for batch in loader:
            x, y = batch

            x = x.to(config.device)
            y = y.to(config.device)

            y_input = y[:, :-1]
            y_target = y[:, 1:]

            pred = model(x, y_input)  # [seq_len, batch_size, num_tokens]

            pred = pred.permute(1, 2, 0)  # [batch_size, num_tokens, seq_len]

            loss = loss_fn(pred, y_target)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.detach().item()

            writer.add_scalar('lr', opt.param_groups[0]['lr'], global_step)
            global_step += 1

        if epoch < config.num_warmup:
            warmup_scheduler.step()
        else:
            scheduler.step()

        t.set_postfix(loss=(total_loss / len(loader)))
        writer.add_scalar('loss', total_loss / len(loader), epoch)

        if epoch % config.test_interval == 0:
            model.eval()
            pred = model(x, y_input)  # [seq_len, batch_size, num_tokens]
            pred = pred.permute(1, 2, 0)  # [batch_size, num_tokens, seq_len]
            ans = pred.argmax(dim=1)[0].detach().tolist()
            ans = ['<sos>'] + vocab.lookup_tokens(ans)
            qs = vocab.lookup_tokens(x[0].tolist())

            text = ' '.join(qs) + '\n\n' + ' '.join(ans)
            writer.add_text('text_generated', text, epoch)
            model.train()

        if epoch % config.save_interval == 0:
            torch.save(model.state_dict(), f'saves/model_{epoch}.pth')


if __name__ == '__main__':
    train()
