# benchmark.py

"""
A benchmarking script for the minGemma model.
"""

import os
import time
from contextlib import nullcontext
import numpy as np
import torch

# Import the minGemma model and related functions from model.py
from model import minGemma, get_model_config, CharacterTokenizer

# Import get_config from your configurator.py
from configurator import get_config

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

config = get_config()

# Create an Args class to hold configurations as attributes
class Args:
    pass

args = Args()
for key, value in config.items():
    setattr(args, key, value)

# Default configuration values for benchmarking
args.batch_size = getattr(args, 'batch_size', 12)
args.block_size = getattr(args, 'block_size', 256)  # Adjusted to fit model's max_position_embeddings
args.bias = getattr(args, 'bias', False)
args.real_data = getattr(args, 'real_data', True)
args.seed = getattr(args, 'seed', 1337)
args.device = getattr(args, 'device', 'cuda' if torch.cuda.is_available() else 'cpu')
args.dtype = getattr(args, 'dtype', 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16')
args.compile = getattr(args, 'compile', True)
args.profile = getattr(args, 'profile', False)

# -----------------------------------------------------------------------------

# Set random seed for reproducibility
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# Enable TF32 for speed
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

device_type = 'cuda' if 'cuda' in args.device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[args.dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

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

if args.real_data:
    data_dir = os.path.join('data', 'openwebtext')  # Adjust dataset name if needed
    train_txt_path = os.path.join(data_dir, 'train.txt')
    if os.path.exists(train_txt_path):
        train_data = load_data(train_txt_path, tokenizer)
    else:
        raise FileNotFoundError(f"train.txt not found in {data_dir}")

    def get_batch(split):
        data = train_data
        ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
        x = torch.stack([data[i:i + args.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + args.block_size + 1] for i in ix])
        x = x.to(args.device)
        y = y.to(args.device)
        return x, y
else:
    # Alternatively, if fixed data is desired to not care about data loading
    vocab_size = len(tokenizer.stoi)
    x = torch.randint(vocab_size, (args.batch_size, args.block_size), device=args.device)
    y = torch.randint(vocab_size, (args.batch_size, args.block_size), device=args.device)
    get_batch = lambda split: (x, y)

# -----------------------------------------------------------------------------
# Model Initialization
# -----------------------------------------------------------------------------

# Adjust the model configuration as per args
config_model.max_position_embeddings = args.block_size
config_model.vocab_size = len(tokenizer.stoi)
config_model.device = args.device

# Instantiate the model
model = minGemma(config_model, tokenizer).to(args.device)

# Use torch.compile if enabled
if args.compile and hasattr(torch, 'compile'):
    print("Compiling model...")
    model = torch.compile(model)

# Prepare optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=1e-2
)

# -----------------------------------------------------------------------------
# Benchmarking
# -----------------------------------------------------------------------------

if args.profile:
    # Profiling with PyTorch profiler
    wait, warmup, active = 5, 5, 5
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./bench_log'),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,  # Set to True if stack traces are needed
        with_flops=True,
        with_modules=False,
    ) as prof:

        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")

            prof.step()  # Notify the profiler at the end of each step
else:
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]):  # Burn-in and benchmark
        t0 = time.time()
        X, Y = get_batch('train')
        for k in range(num_steps):
            with ctx:
                logits, loss = model(X, Y)
            X, Y = get_batch('train')
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            lossf = loss.item()
            print(f"{k}/{num_steps} loss: {lossf:.4f}")
        torch.cuda.synchronize()
        t1 = time.time()
        dt = t1 - t0
        if stage == 1:
            print(f"Time per iteration: {dt / num_steps * 1000:.4f} ms")
