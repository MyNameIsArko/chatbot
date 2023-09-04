import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_layers = 2  # 6
num_heads = 4  # 8
dim_model = 256  # 512
dim_inner = 512  # 2048
dropout = 0.1
max_seq_len = 52
max_mem_len = 200

unk_idx = 0
pad_idx = 1
sos_idx = 2
eos_idx = 3
batch_size = 16
num_dialogs = 10
num_epochs = 200
save_interval = 10
test_interval = 10
num_warmup = 10
lr = 1e-2
nucleus_prob = 0.5
