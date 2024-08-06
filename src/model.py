# Pytorch, Config, Model, and Dataset Imports
import torch
import torch.nn as nn
from torch.nn import functional as F
import dataclasses
from typing import Optional
import re
from typing import Any, List, Sequence, Tuple, Union
import time

# load the dataset
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text and how many there are
chars = sorted(list(set(text)))
v = len(chars)

class CharacterTokenizer:
    def __init__(self, chars: List[str]):
        """ Create mappings from characters to integers and vice versa """
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, s: str) -> List[int]:
        """Converts a string into a list of character IDs"""
        return [self.stoi.get(c) for c in s] 

    def decode(self, t: List[int]) -> str:
        """Converts a list of character IDs back into a string."""
        return ''.join([self.itos.get(i) for i in t])

@dataclasses.dataclass
class GemmaConfig:
    vocab_size: int = v
    max_position_embeddings: int = 256
    num_hidden_layers: int = 4
    num_attention_heads: int = 4
    num_key_value_heads: int = 1
    hidden_size: int = 128
    intermediate_size: int = 512
    head_dim: int = 32
    rms_norm_eps: float = 1e-6
    tokenizer: Optional[str] = None 
    rope_theta = 100.0
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model_config(variant: str = None) -> GemmaConfig:
        return GemmaConfig(), CharacterTokenizer(chars)

config, tokenizer = get_model_config()

def apply_rotary_emb(x: torch.Tensor, dim: int, theta: float = 10000.0) -> torch.Tensor:
    seq_len = x.size(1)
    device = x.device
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, device=device).float() / dim))
    t = torch.arange(seq_len, device=device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs) 
    x_ = torch.view_as_complex(torch.stack(torch.chunk(x.transpose(1, 2).float(), 2, dim=-1), dim=-1))
    x_out = torch.view_as_real(x_ * freqs_cis.unsqueeze(0)).type_as(x)
    x_out = torch.cat(torch.chunk(x_out, 2, dim=-1), dim=-2)
    x_out = x_out.reshape(x_out.shape[0], x_out.shape[1], x_out.shape[2], -1).transpose(1, 2)
    return x_out

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, add_unit_offset: bool = True):
        super().__init__() 
        self.eps = eps 
        self.add_unit_offset = add_unit_offset 
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        x = self._norm(x.float()).type_as(x)
        if self.add_unit_offset:
            output = x * (1 + self.weight)
        else:
            output = x * self.weight
        return output

class GemmaMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        gate = self.gate_proj(x)
        gate = gate * torch.sigmoid(gate) # implements Swiglu instead of gated GELU
        up = self.up_proj(x)
        fuse = gate * up
        outputs = self.down_proj(fuse)
        return outputs

class GemmaAttention(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        self.theta = config.rope_theta
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.qkv_proj = nn.Linear(self.hidden_size, (self.num_heads + 2 * self.num_kv_heads) * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        mask_negatives = torch.full((1, 1, config.max_position_embeddings, config.max_position_embeddings), -2.3819763e38).to(torch.float)
        mask = torch.triu(mask_negatives, diagonal=1).to(config.device)
        self.register_buffer('mask', mask)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states_shape = hidden_states.shape
        assert len(hidden_states_shape) == 3
        batch_size, input_len, _ = hidden_states_shape
        qkv = self.qkv_proj(hidden_states)
        xq, xk, xv = qkv.split([self.q_size, self.kv_size, self.kv_size],dim=-1)
        xq = xq.view(batch_size, -1, self.num_heads, self.head_dim)
        xk = xk.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xv = xv.view(batch_size, -1, self.num_kv_heads, self.head_dim)
        xq = apply_rotary_emb(xq, self.head_dim, self.theta)
        xk = apply_rotary_emb(xk, self.head_dim, self.theta)
        if self.num_kv_heads != self.num_heads:
            xk = torch.repeat_interleave(xk, self.num_queries_per_kv, dim=2)
            xv = torch.repeat_interleave(xv, self.num_queries_per_kv, dim=2)
        q = xq.transpose(1, 2)
        k = xk.transpose(1, 2)
        v = xv.transpose(1, 2)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        scores = scores + self.mask[...,:input_len, :input_len]
        scores = F.softmax(scores, dim=-1)
        output = torch.matmul(scores, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, input_len, -1)
        output = self.o_proj(output)
        return output

class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.self_attn = GemmaAttention(config)
        self.mlp = GemmaMLP(
            hidden_size = config.hidden_size,
            intermediate_size = config.intermediate_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps = config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states=hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

class GemmaBody(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.layers = nn.ModuleList(GemmaDecoderLayer(config) for _ in range(config.num_hidden_layers))
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            hidden_states = layer(hidden_states=hidden_states)
        hidden_states = self.norm(hidden_states)
        return hidden_states

class minGemma(nn.Module):
    def __init__(self, config: GemmaConfig, tokenizer: CharacterTokenizer):
        super().__init__()
        self.config = config
        assert config.hidden_size % config.num_attention_heads == 0
        self.max_seq_len = config.max_position_embeddings
        self.head_dim = config.head_dim
        self.vocab_size = config.vocab_size
        self.tokenizer = tokenizer
        self.embedder = nn.Embedding(self.vocab_size, config.hidden_size)
        self.model = GemmaBody(config)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_token_ids: torch.Tensor, target_token_ids: torch.Tensor = None) -> torch.Tensor:
        hidden_states = self.embedder(input_token_ids)
        hidden_states = hidden_states * (self.config.hidden_size**0.5)
        hidden_states = self.model(hidden_states=hidden_states)
        embedder_weight = self.embedder.weight
        logits = torch.matmul(hidden_states, embedder_weight.t())
        if target_token_ids is None:
            loss = None
        else:
            batch_size, input_len, vocab_size = logits.shape
            loss = self.criterion(logits.view(batch_size*input_len, vocab_size), target_token_ids.view(batch_size*input_len))
        return logits, loss

    @torch.no_grad()
    def Sampler(self, logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
        logits = logits[:,-1,:]
        logits.div_(temperature)
        probs = torch.softmax(logits, dim=-1, dtype=torch.float)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort, dim=-1)
        top_ps_mask = (probs_sum - probs_sort) > top_p
        probs_sort = torch.where(top_ps_mask, 0, probs_sort)
        top_ks_mask = torch.arange(probs_idx.shape[-1], device=probs_idx.device)
        top_ks_mask = top_ks_mask.expand(probs_idx.shape[0], -1)
        top_ks_mask = top_ks_mask >= top_k
        probs_sort = torch.where(top_ks_mask, 0, probs_sort)
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        probs = torch.gather(probs_sort, dim=-1, index=torch.argsort(probs_idx, dim=-1))
        next_token_id = torch.multinomial(probs, num_samples=1)
        return next_token_id

    def generate(self, prompt: str, output_len: int = 100, temperature: float = 0.95, top_p: float = 1.0, top_k: int = 65) -> str:
        tokens = self.tokenizer.encode(prompt)
        tokens = torch.tensor(tokens, device=config.device).unsqueeze(0)
        assert len(tokens) + output_len <= self.config.max_position_embeddings
        for i in range(output_len):
            logits, _ = self(tokens[:,:self.max_seq_len])
            next_token = self.Sampler(logits, temperature, top_p, top_k)
            tokens = torch.cat((tokens, next_token), dim=1)
        output = self.tokenizer.decode(tokens.squeeze(0).tolist())
        return output

# Train and test splits
data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split, batch_size):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - config.max_position_embeddings, (batch_size,))
    x = torch.stack([data[i:i+config.max_position_embeddings] for i in ix])
    y = torch.stack([data[i+1:i+config.max_position_embeddings+1] for i in ix])
    x, y = x.to(config.device), y.to(config.device)
    return x, y

@torch.no_grad()
def estimate_loss(model, batch_size, eval_iters = 10):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, batch_size)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Configuration and Tokenizer
config, tokenizer = get_model_config()

# Flag to determine whether to use a new model or load a pretrained model
use_pretrained = False  # Set to True to load a pretrained model

# Model instantiation
if use_pretrained:
    # Initialize a blank model
    model = minGemma(config, tokenizer).to(config.device)

    # Path to the pretrained model
    path = 'models/minGemma-vocab_size65-max_position_embeddings256-num_hidden_layers4-num_attention_heads4-num_key_value_heads1-hidden_size128-intermediate_size512-head_dim32-rms_norm_eps1e-06-rope_theta100.0--2024-02-23|02-10-08.pth'

    # Load the saved state dictionary
    model.load_state_dict(torch.load(path))

    model.eval()

    print("Loaded pretrained model.")
else:
    # Instantiate a new model
    model = minGemma(config, tokenizer).to(config.device)

    print("Instantiated a new model.")

# Print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e3, 'K parameters')
# print(model)
