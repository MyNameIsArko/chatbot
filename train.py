import os

import random
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


def train():
    print(f'Device used: {config.device}')

    loader, num_tokens, vocab = get_data()

    model = Seq2Seq(num_tokens=num_tokens).to(config.device)

    opt = optim.AdamW(model.parameters(), lr=config.lr)

    scheduler = optim.lr_scheduler.OneCycleLR(opt, max_lr=config.lr, epochs=config.num_epochs, steps_per_epoch=len(loader))

    loss_fn = nn.CrossEntropyLoss(ignore_index=config.pad_idx, label_smoothing=0.1)

    os.makedirs('saves', exist_ok=True)
    date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    wandb.init(
        project='chatbot',
        name=date,
        config={
            'learning_rate': config.lr,
            'num_epochs': config.num_epochs,
            'max_seq_len': config.max_seq_len,
            'num_dialogs': config.num_dialogs,
            'batch_size': config.batch_size,
            'num_layers': config.num_layers,
            'num_heads': config.num_heads,
            'dim_model': config.dim_model,
            'dim_inner': config.dim_inner,
            'compression_loss_weight': config.compression_loss_weight,
            'clip': config.clip,
            'max_mem_len': config.max_mem_len,
            'max_cmem_len': config.max_cmem_len,
            'compression_rate': config.compression_rate,
        },
        sync_tensorboard=True
    )
    writer = SummaryWriter(f'logs/chatbot_{date}')

    t = tqdm(range(1, config.num_epochs + 1), position=0, leave=True)

    global_step = 0

    for epoch in t:
        total_loss = 0
        t2 = tqdm(loader, position=1, leave=False)
        for dialog in t2:
            model.clear_memory()
            attention_loss = 0
            aux_loss = 0
            for i in range(len(dialog) - 1):
                x = dialog[i].to(config.device)
                y = dialog[i+1].to(config.device)

                y_input = y[:, :-1]
                y_target = y[:, 1:]

                pred, al = model(x, y_input)

                pred = pred.permute(0, 2, 1)  # [batch_size, num_tokens, seq_len]

                attention_loss += loss_fn(pred, y_target)
                aux_loss += al

                if random.random() < config.dropout / len(dialog):
                    model.clear_memory()

            opt.zero_grad()
            loss = attention_loss + aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.clip)
            opt.step()

            loss = loss.detach().item()

            total_loss += loss

            t2.set_postfix(attention_loss=attention_loss.detach().item(), aux_loss=aux_loss.detach().item(), lr=opt.param_groups[0]['lr'])
            writer.add_scalar('step_loss', loss, global_step)
            writer.add_scalar('lr', opt.param_groups[0]['lr'], global_step)
            global_step += 1

            scheduler.step()  # update lr on every step instead of every epoch

        t.set_postfix(loss=(total_loss / len(loader)))
        writer.add_scalar('epoch_loss', total_loss / len(loader), epoch)

        if epoch % config.test_interval == 0:
            model.clear_memory()
            model.eval()
            x = dialog[0].to(config.device)
            y = dialog[1].to(config.device)
            y_input = y[:, :-1]
            pred, _ = model(x, y_input)  # [seq_len, batch_size, num_tokens]
            pred = pred.permute(0, 2, 1)  # [batch_size, num_tokens, seq_len]
            ans = pred.argmax(dim=1)[0].detach().tolist()
            ans = ' '.join(vocab.lookup_tokens(ans))
            qs = ' '.join(vocab.lookup_tokens(x[0].tolist()))

            text = qs + '\n\n' + ans
            writer.add_text('text_generated', text, epoch)
            model.train()

        if epoch % config.save_interval == 0:
            torch.save(model.state_dict(), f'saves/model_{epoch}.pth')


if __name__ == '__main__':
    train()
