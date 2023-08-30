import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

unknown_idx = 0
padding_idx = 1
sos_idx = 2
eos_idx = 3
max_seq_len = 52
batch_size = 32
num_epochs = 600
save_interval = 10
test_interval = 10
num_warmup = 10
lr = 1e-4
nucleus_prob = 0.5
