import os
import time
import math
import pickle
from contextlib import nullcontext
import importlib.util

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import deepspeed

from configurator import get_config

# Get configurations from configurator.py
config = get_config()

# Create an Args class to hold configurations as attributes
class Args:
    pass

args = Args()
for key, value in config.items():
    setattr(args, key, value)

# -----------------------------------------------------------------------------

# Initialize distributed training
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        torch.cuda.set_device(args.local_rank)
        deepspeed.init_distributed()
        args.distributed = True
        args.device = torch.device('cuda', args.local_rank)
    else:
        args.local_rank = 0
        args.rank = 0
        args.world_size = 1
        args.distributed = False
        args.device = torch.device(getattr(args, 'device', 'cuda'))
    return args.device, args.distributed, args.rank, args.world_size

device, distributed, rank, world_size = setup_distributed()

seed = getattr(args, 'seed', 1337)
torch.manual_seed(seed + rank)
np.random.seed(seed + rank)

dtype = getattr(args, 'dtype', 'bfloat16')
if dtype == 'float32':
    ptdtype = torch.float32
elif dtype == 'bfloat16':
    ptdtype = torch.bfloat16
elif dtype == 'float16':
    ptdtype = torch.float16
else:
    raise ValueError(f"Unsupported dtype: {dtype}")

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, data_path, block_size):
        self.data = np.memmap(data_path, dtype=np.uint16, mode='r')
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = torch.from_numpy((self.data[idx:idx + self.block_size]).astype(np.int64))
        y = torch.from_numpy((self.data[idx + 1:idx + 1 + self.block_size]).astype(np.int64))
        return x, y

def get_dataloader(split):
    data_dir = os.path.join('data', args.dataset)
    data_path = os.path.join(data_dir, f'{split}.bin')
    dataset = TextDataset(data_path, args.block_size)
    sampler = torch.utils.data.distributed.DistributedSampler(dataset) if args.distributed else None
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        pin_memory=True
    )
    return dataloader

train_loader = get_dataloader('train')
val_loader = get_dataloader('val')
# Load tokenizer (Come back to this later)  
tokenizer = AutoTokenizer.from_pretrained(args.Gemma)
model = AutoModelForCausalLM.from_pretrained(args.Gemma, torch_dtype=ptdtype)

# Adjust model configurations if needed
model.config.n_layer = args.n_layer
model.config.n_head = args.n_head
model.config.n_embd = args.n_embd
model.config.use_cache = False  # Disable cache for training
model.to(device)

# Use torch.compile if enabled
if getattr(args, 'compile', False):
    model = torch.compile(model)

# Prepare optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(args.beta1, args.beta2),
    weight_decay=args.weight_decay
)

# Load DeepSpeed configuration if specified
if hasattr(args, 'deepspeed_config'):
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)
else:
    ds_config = None

# DeepSpeed initialization
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
model, optimizer, _, scheduler = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model_parameters,
    optimizer=optimizer,
    lr_scheduler=None,
    config_params=ds_config
)

# Learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=args.warmup_iters,
    num_training_steps=args.lr_decay_iters
)

# Training Loop with DoLa Integration
def train():
    model.train()
    total_loss = 0.0
    step = 0
    while True:
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            # Forward pass
            outputs = model(x, labels=y)
            loss = outputs.loss

            # If DoLa is enabled, compute DoLa loss and combine
            if getattr(args, 'use_dola', False):
                dola_loss = compute_dola_loss(
                    model,
                    x,
                    y,
                    args.mature_layer,
                    args.premature_layer,
                    args.relative_top
                )
                loss += dola_loss  # You might want to weight the dola_loss

            # Backward pass and optimization
            model.backward(loss)
            model.step()
            if getattr(args, 'decay_lr', False):
                scheduler.step()
            if step % args.log_interval == 0 and rank == 0:
                print(f"Step {step}, Loss: {loss.item()}")

            if step % args.eval_interval == 0 and step > 0:
                evaluate()

            if step >= args.max_iters:
                return

            step += 1

def evaluate():
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            outputs = model(x, labels=y)
            loss = outputs.loss

            eval_loss += loss.item()

    avg_loss = eval_loss / len(val_loader)
    if rank == 0:
        print(f"Validation Loss: {avg_loss}")
    model.train()

def compute_dola_loss(model, x, y, mature_layer, premature_layer, relative_top):
    # Implement DoLa loss computation
    outputs = model(
        input_ids=x,
        output_hidden_states=True,
        return_dict=True
    )

    hidden_states = outputs.hidden_states  # List of hidden states from each layer

    # Get logits from mature and premature layers then computing differences
    mature_logits = model.lm_head(hidden_states[mature_layer])
    premature_logits = model.lm_head(hidden_states[premature_layer])
    diff_logits = mature_logits - premature_logits

    # Apply relative top filtering
    if relative_top > 0.0:
        diff_logits = apply_relative_top_filter(diff_logits, relative_top)
    # NLL Loss
    diff_logits = F.log_softmax(diff_logits, dim=-1)
    loss = F.nll_loss(diff_logits.view(-1, diff_logits.size(-1)), y.view(-1))

    return loss

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
    train()
