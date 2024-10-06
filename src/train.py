# train.py

import os
import time
import math
import pickle
from contextlib import nullcontext
import importlib.util
import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from configurator import get_config

from model import minGemma, get_model_config, CharacterTokenizer

# Config
config = get_config()

class Args:
    pass

args = Args()
for key, value in config.items():
    setattr(args, key, value)


args.out_dir = getattr(args, 'out_dir', 'out')
args.eval_interval = getattr(args, 'eval_interval', 2000)
args.log_interval = getattr(args, 'log_interval', 1)
args.eval_iters = getattr(args, 'eval_iters', 200)
args.eval_only = getattr(args, 'eval_only', False)
args.always_save_checkpoint = getattr(args, 'always_save_checkpoint', True)
args.init_from = getattr(args, 'init_from', 'scratch')
args.dataset = getattr(args, 'dataset', 'openwebtext')
args.gradient_accumulation_steps = getattr(args, 'gradient_accumulation_steps', 5 * 8)
args.batch_size = getattr(args, 'batch_size', 12)
args.block_size = getattr(args, 'block_size', 1024)
args.n_layer = getattr(args, 'n_layer', 24)
args.n_head = getattr(args, 'n_head', 16)
args.n_embd = getattr(args, 'n_embd', 2048)
args.dropout = getattr(args, 'dropout', 0.0)
args.bias = getattr(args, 'bias', False)
args.learning_rate = getattr(args, 'learning_rate', 6e-4)
args.max_iters = getattr(args, 'max_iters', 600000)
args.weight_decay = getattr(args, 'weight_decay', 1e-1)
args.beta1 = getattr(args, 'beta1', 0.9)
args.beta2 = getattr(args, 'beta2', 0.95)
args.grad_clip = getattr(args, 'grad_clip', 1.0)
args.decay_lr = getattr(args, 'decay_lr', True)
args.warmup_iters = getattr(args, 'warmup_iters', 2000)
args.lr_decay_iters = getattr(args, 'lr_decay_iters', 600000)
args.min_lr = getattr(args, 'min_lr', 6e-5)
args.backend = getattr(args, 'backend', 'nccl')
args.device = getattr(args, 'device', 'cuda')
args.dtype = getattr(args, 'dtype', 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
args.compile = getattr(args, 'compile', True)

# DoLa parameters
args.use_dola = getattr(args, 'use_dola', True)
args.mature_layer = getattr(args, 'mature_layer', args.n_layer - 1)  # Last layer
args.premature_layer = getattr(args, 'premature_layer', args.n_layer // 2)  # Middle layer
args.relative_top = getattr(args, 'relative_top', 0.1)

# Initialize distributed training
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        init_process_group(backend=args.backend)
        args.distributed = True
        args.device = torch.device('cuda', args.local_rank)
    else:
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1
        args.distributed = False
        args.device = torch.device(args.device)
    return args.device, args.distributed, args.rank, args.world_size

device, distributed, rank, world_size = setup_distributed()

# Set random seed for reproducibility
seed = getattr(args, 'seed', 1337)
torch.manual_seed(seed + rank)
np.random.seed(seed + rank)

# Set data type
dtype = args.dtype
if dtype == 'float32':
    ptdtype = torch.float32
elif dtype == 'bfloat16':
    ptdtype = torch.bfloat16
elif dtype == 'float16':
    ptdtype = torch.float16
else:
    raise ValueError(f"Unsupported dtype: {dtype}")

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

# Load tokenizer and get character mappings
config_model, tokenizer = get_model_config()
chars = list(tokenizer.stoi.keys())

# Function to load data
def load_data(file_path, tokenizer):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    return data

# Load the data
data_dir = os.path.join('data', args.dataset)
train_txt_path = os.path.join(data_dir, 'train.txt')
val_txt_path = os.path.join(data_dir, 'val.txt')

if os.path.exists(train_txt_path) and os.path.exists(val_txt_path):
    train_data = load_data(train_txt_path, tokenizer)
    val_data = load_data(val_txt_path, tokenizer)
else:
    raise FileNotFoundError(f"train.txt and val.txt not found in {data_dir}")

# Custom Dataset class
class CharDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.block_size]
        y = self.data[idx + 1:idx + self.block_size + 1]
        return x, y

def get_dataloader(dataset):
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True,
        num_workers=4
    )
    return dataloader

# Create datasets and dataloaders
train_dataset = CharDataset(train_data, args.block_size)
val_dataset = CharDataset(val_data, args.block_size)

train_loader = get_dataloader(train_dataset)
val_loader = get_dataloader(val_dataset)

# Adjust the model configuration as per args
config_model.num_hidden_layers = args.n_layer
config_model.num_attention_heads = args.n_head
config_model.hidden_size = args.n_embd
config_model.max_position_embeddings = args.block_size
config_model.vocab_size = len(tokenizer.stoi)
config_model.device = args.device

# Initialize the model
model = minGemma(config_model, tokenizer).to(device)

# load from a checkpoint
if args.init_from == 'resume':
    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        optimizer_state = checkpoint['optimizer']
        start_iter = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
    else:
        print(f"Checkpoint not found at {ckpt_path}")
        start_iter = 0
        best_val_loss = float('inf')
else:
    start_iter = 0
    best_val_loss = float('inf')

# Use PyTorch 2.0 compile if enabled
if args.compile:
    model = torch.compile(model)

# Prepare optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay
)

# Learning rate scheduler
if args.decay_lr:
    def get_lr(it):
        if it < args.warmup_iters:
            return args.learning_rate * it / args.warmup_iters
        if it > args.lr_decay_iters:
            return args.min_lr
        decay_ratio = (it - args.warmup_iters) / (args.lr_decay_iters - args.warmup_iters)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
        return args.min_lr + coeff * (args.learning_rate - args.min_lr)
else:
    def get_lr(it):
        return args.learning_rate

# Training Loop with DoLa Integration

def train():
    model.train()
    iter_num = start_iter
    raw_model = model.module if hasattr(model, "module") else model
    t0 = time.time()

    while True:
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            # Determine and set the learning rate for this iteration
            lr = get_lr(iter_num)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # Zero gradients
            optimizer.zero_grad()
            # Set up hooks
            if args.use_dola:
                mature_hidden_states = {}
                premature_hidden_states = {}

                def get_activation(name):
                    def hook(model, input, output):
                        if name == 'mature':
                            mature_hidden_states['output'] = output.detach()
                        elif name == 'premature':
                            premature_hidden_states['output'] = output.detach()
                    return hook

                mature_layer_module = model.model.layers[args.mature_layer]
                premature_layer_module = model.model.layers[args.premature_layer]

                mature_hook = mature_layer_module.register_forward_hook(get_activation('mature'))
                premature_hook = premature_layer_module.register_forward_hook(get_activation('premature'))

            # Forward pass
            logits, loss = model(x, y)

            # Remove hooks after forward pass
            if args.use_dola:
                mature_hook.remove()
                premature_hook.remove()

                # Compute DoLa loss
                dola_loss = compute_dola_loss(model, y, mature_hidden_states['output'], premature_hidden_states['output'])
                loss += dola_loss  # You might want to weight the dola_loss differently

            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            # Logging
            if iter_num % args.log_interval == 0 and rank == 0:
                lossf = loss.item()
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")

            # Evaluation
            if iter_num % args.eval_interval == 0 and iter_num > 0:
                evaluate()
                # Save checkpoint if necessary
                if args.always_save_checkpoint and rank == 0:
                    checkpoint = {
                        'model': raw_model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                    ckpt_path = os.path.join(args.out_dir, 'ckpt.pt')
                    torch.save(checkpoint, ckpt_path)
                    print(f"Checkpoint saved at iter {iter_num}")

            iter_num += 1
            if iter_num >= args.max_iters:
                print("Training completed.")
                return

def evaluate():
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            logits, loss = model(x, y)
            losses.append(loss.item())
    avg_loss = np.mean(losses)
    if rank == 0:
        print(f"Validation loss: {avg_loss:.4f}")
    model.train()

def compute_dola_loss(model, y, mature_hidden, premature_hidden):
    # Compute logits from these hidden states
    embedder_weight = model.embedder.weight
    mature_logits = torch.matmul(mature_hidden, embedder_weight.t())
    premature_logits = torch.matmul(premature_hidden, embedder_weight.t())

    # Compute difference in logits
    diff_logits = mature_logits - premature_logits

    # Optionally apply relative top filtering
    if args.relative_top > 0.0:
        diff_logits = apply_relative_top_filter(diff_logits, args.relative_top)

    # Compute loss
    diff_logits = F.log_softmax(diff_logits, dim=-1)
    dola_loss = F.nll_loss(diff_logits.view(-1, diff_logits.size(-1)), y.view(-1))

    return dola_loss

def apply_relative_top_filter(logits, relative_top):
    # Implement relative top filtering
    logits_normalized = F.log_softmax(logits, dim=-1)
    sorted_logits, _ = torch.sort(logits_normalized, descending=True, dim=-1)
    threshold_index = int(relative_top * logits.size(-1))
    threshold = sorted_logits[:, :, threshold_index].unsqueeze(-1)
    mask = logits_normalized < threshold
    logits = logits.masked_fill(mask, float('-inf'))
    return logits

if __name__ == '__main__':
    try:
        train()
    finally:
        if distributed:
            destroy_process_group()
