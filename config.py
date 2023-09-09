import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_layers = 6
num_heads = 8
dim_model = 512
dim_inner = 2048
dropout = 0.1
max_seq_len = 22
max_mem_len = 80
clip = 0.1
compression_rate = 3
max_cmem_len = 100
compression_loss_weight = 0.1

unk_idx = 0
pad_idx = 1
sos_idx = 2
eos_idx = 3
batch_size = 16
num_dialogs = 10
num_epochs = 40  # 200
save_interval = 10
test_interval = 10
epoch_warmup = 2
epoch_cosine_decay = 10
mult_cosine_decay = 2
lr = 1e-4
nucleus_prob = 0.5
