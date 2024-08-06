import os
import time
import math

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

# default config values for training


#data
dataset = 'coqa'


# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or 'mps' for the macbook

# DDP
backend = 'nccl'

# import os
# import time
# import math
# import torch
# import numpy as np
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.distributed import init_process_group, destroy_process_group
# from contextlib import nullcontext
# import wandb
# from model import GPTConfig, GPT
# from config_loader import get_config
# def setup_training_environment(config):
#     # Set up distributed training if applicable
#     if int(os.environ.get('RANK', -1)) != -1:
#         init_process_group(backend=config['backend'])
#         ddp_rank = int(os.environ['RANK'])
#         ddp_local_rank = int(os.environ['LOCAL_RANK'])
#         world_size = int(os.environ['WORLD_SIZE'])
#         device = f'cuda:{ddp_local_rank}'
#         torch.cuda.set_device(device)
#         is_master = ddp_rank == 0
#     else:
#         world_size = 1
#         is_master = True
#         device = config['device']
#     # Set up dtype and autocast context
#     dtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[config['dtype']]
#     device_type = 'cuda' if 'cuda' in device else 'cpu'
#     ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=dtype)
#     return device, is_master, world_size, ctx
# def init_model(config, device):
#     model_args = {k: config[k] for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size', 'dropout']}
    
#     if config['init_from'] == 'scratch':
#         model = GPT(GPTConfig(**model_args))
#     elif config['init_from'] == 'resume':
#         ckpt_path = os.path.join(config['out_dir'], 'ckpt.pt')
#         checkpoint = torch.load(ckpt_path, map_location=device)
#         model = GPT(GPTConfig(**checkpoint['model_args']))
#         model.load_state_dict(checkpoint['model'])
#     elif config['init_from'].startswith('gpt2'):
#         model = GPT.from_pretrained(config['init_from'], dict(dropout=config['dropout']))
#         for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
#             model_args[k] = getattr(model.config, k)
#     if config['block_size'] < model.config.block_size:
#         model.crop_block_size(config['block_size'])
#     model.to(device)
    
#     if config['compile']:
#         model = torch.compile(model)
#     return model, model_args
# def get_batch(split, config, data_dir):
#     data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
#     ix = torch.randint(len(data) - config['block_size'], (config['batch_size'],))
#     x = torch.stack([torch.from_numpy((data[i:i+config['block_size']]).astype(np.int64)) for i in ix])
#     y = torch.stack([torch.from_numpy((data[i+1:i+1+config['block_size']]).astype(np.int64)) for i in ix])
#     return x.to(config['device']), y.to(config['device'])
# def estimate_loss(model, config, ctx, data_dir):
#     out = {}
#     model.eval()
#     for split in ['train', 'val']:
#         losses = torch.zeros(config['eval_iters'])
#         for k in range(config['eval_iters']):
#             X, Y = get_batch(split, config, data_dir)
#             with ctx:
#                 logits, loss = model(X, Y)
#             losses[k] = loss.item()
#         out[split] = losses.mean()
#     model.train()
#     return out
# def get_lr(iter_num, config):
#     if iter_num < config['warmup_iters']:
#         return config['learning_rate'] * iter_num / config['warmup_iters']
#     if iter_num > config['lr_decay_iters']:
#         return config['min_lr']
#     decay_ratio = (iter_num - config['warmup_iters']) / (config['lr_decay_iters'] - config['warmup_iters'])
#     coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
#     return config['min_lr'] + coeff * (config['learning_rate'] - config['min_lr'])
# def train():
#     # Load configuration
#     config = get_config()
    
#     # Setup training environment
#     device, is_master, world_size, ctx = setup_training_environment(config)
    
#     # Initialize model
#     model, model_args = init_model(config, device)
    
#     # Setup optimizer
#     optimizer = model.configure_optimizers(config['weight_decay'], config['learning_rate'], 
#                                            (config['beta1'], config['beta2']), device)
    
#     # Setup gradient scaler
#     scaler = torch.cuda.amp.GradScaler(enabled=(config['dtype'] == 'float16'))
    
#     # Wrap model in DDP if using distributed training
#     if world_size > 1:
#         model = DDP(model, device_ids=[int(device.split(':')[1])])
    
#     # Setup data directory
#     data_dir = os.path.join('data', config['dataset'])
    
#     # Initialize wandb if enabled
#     if config['wandb_log'] and is_master:
#         wandb.init(project=config['wandb_project'], name=config['wandb_run_name'], config=config)
    
#     # Main training loop
#     iter_num = 0
#     best_val_loss = float('inf')
#     t0 = time.time()
#     running_mfu = -1.0
    
#     while True:
#         # Learning rate decay
#         lr = get_lr(iter_num, config) if config['decay_lr'] else config['learning_rate']
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr
        
#         # Evaluate model
#         if iter_num % config['eval_interval'] == 0 and is_master:
#             losses = estimate_loss(model, config, ctx, data_dir)
#             print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
#             if config['wandb_log']:
#                 wandb.log({"iter": iter_num, "train/loss": losses['train'], "val/loss": losses['val'], "lr": lr})
            
#             if losses['val'] < best_val_loss or config['always_save_checkpoint']:
#                 best_val_loss = losses['val']
#                 if iter_num > 0:
#                     checkpoint = {
#                         'model': model.state_dict(),
#                         'optimizer': optimizer.state_dict(),
#                         'model_args': model_args,
#                         'iter_num': iter_num,
#                         'best_val_loss': best_val_loss,
#                         'config': config,
#                     }
#                     print(f"saving checkpoint to {config['out_dir']}")
#                     torch.save(checkpoint, os.path.join(config['out_dir'], 'ckpt.pt'))
        
#         if iter_num == 0 and config['eval_only']:
#             break
        
#         # Forward and backward pass
#         for micro_step in range(config['gradient_accumulation_steps']):
#             X, Y = get_batch('train', config, data_dir)
#             with ctx:
#                 logits, loss = model(X, Y)
#                 loss = loss / config['gradient_accumulation_steps']
#             scaler.scale(loss).backward()
        
#         # Gradient clipping
#         if config['grad_clip'] != 0.0:
#             scaler.unscale_(optimizer)
#             torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
        
#         # Step optimizer
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad(set_to_none=True)
        
#         # Timing and MFU calculation
#         if iter_num % config['log_interval'] == 0:
#             t1 = time.time()
#             dt = t1 - t0
#             t0 = t1
            
#             # This is the CPU-GPU sync point
#             lossf = loss.item() * config['gradient_accumulation_steps']
            
#             if iter_num > 0:  # Skip the first iteration
#                 flops_per_iter = model.estimate_mfu(config['batch_size'] * config['gradient_accumulation_steps'], dt)
#                 mfu = flops_per_iter / (config['log_interval'] * config['gradient_accumulation_steps'])
#                 running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
            
#             if is_master:
#                 print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
            
#             if config['wandb_log']:
#                 wandb.log({
#                     "iter": iter_num,
#                     "loss": lossf,
#                     "lr": lr,
#                     "mfu": running_mfu,
#                 })
        
#         iter_num += 1
        
#         # Check for termination
#         if iter_num > config['max_iters']:
#             break
    
#     # Cleanup
#     if world_size > 1:
#         destroy_process_group()
# if __name__ == '__main__':
#     train()


# Training

learning_rate = 3e-4
weight_decay = 0.01
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
max_iters = 10000
eval_interval = 250
batch_size = 32

start_time = time.time()
for iter in range(max_iters):
    xb, yb = get_batch('train', batch_size)
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    if iter % eval_interval == 0 or iter == max_iters - 1:
        current_time = time.time()
        elapsed_time = current_time - start_time
        losses = estimate_loss(model, batch_size)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}, time elapsed: {elapsed_time:.2f} seconds")

# Saving your model
torch.save(model.state_dict(), f'models/{model.__class__.__name__}'
           f'-vocab_size{config.vocab_size}'
           f'-max_position_embeddings{config.max_position_embeddings}'
           f'-num_hidden_layers{config.num_hidden_layers}'
           f'-num_attention_heads{config.num_attention_heads}'
           f'-num_key_value_heads{config.num_key_value_heads}'
           f'-hidden_size{config.hidden_size}'
           f'-intermediate_size{config.intermediate_size}'
           f'-head_dim{config.head_dim}'
           f'-rms_norm_eps{config.rms_norm_eps}'
           f'-rope_theta{config.rope_theta}'
           f'--{time.strftime("%Y-%m-%d|%H-%M-%S")}.pth')

# Inference
input_str = "JULIET:\nO Romeo, Romeo! wherefore art thou R"
max_useable_output_len = config.max_position_embeddings - len(input_str)
output = model.generate(input_str, output_len = max_useable_output_len)
print(output)