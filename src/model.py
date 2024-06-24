import torch
import numpy as np
import torch.nn as nn
import math

"""
To-do:
- configure the model
- initialize said model from huggingface
- create a function that initializes the weight of model
- create a function that initializes the bias of model?

Other things to consider:
- The volume size of inputs and outputs (creating the restrictions)
- The number of layers
- The number of attention heads
- The number of embeddings

And a lot more stuff

"""


config = {
 """

Add the configurations of: 
Gemma-2b
Gemma2-b instruct 
Gemma-7b
Gemma-7b instruct
"""
 
}



import tensorflow as tf

def gelu(x):
    """Gaussian Error Linear Unit."""
    return 0.5 * x * (1 + tf.tanh(tf.sqrt(2 / np.pi) * (x + 0.044715 * tf.pow(x, 3))))

class NanoGemmaConfig:
    def __init__(self, vocab_size=50257, n_layer=12, n_head=12, n_embd=768, max_seq_len=1024):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_seq_len = max_seq_len

class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = tf.keras.layers.Dropout(0.1)
        
        self.c_attn = tf.keras.layers.Dense(3 * self.n_embd)
        self.c_proj = tf.keras.layers.Dense(self.n_embd)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.n_head, self.n_embd // self.n_head))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, x, mask=None):
        batch_size = tf.shape(x)[0]
        
        qkv = self.c_attn(x)
        q, k, v = tf.split(qkv, 3, axis=-1)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        attn = tf.matmul(q, k, transpose_b=True) / tf.math.sqrt(tf.cast(self.n_embd // self.n_head, tf.float32))
        
        if mask is not None:
            attn += (mask * -1e9)
        
        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout(attn)
        
        out = tf.matmul(attn, v)
        out = tf.transpose(out, perm=[0, 2, 1, 3])
        out = tf.reshape(out, (batch_size, -1, self.n_embd))
        
        return self.c_proj(out)

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.c_fc = tf.keras.layers.Dense(4 * config.n_embd, activation=gelu)
        self.c_proj = tf.keras.layers.Dense(config.n_embd)
        self.dropout = tf.keras.layers.Dropout(0.1)

    def call(self, x):
        x = self.c_fc(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, config):
        super().__init__()
        self.attn = AttentionBlock(config)
        self.ff = FeedForward(config)
        self.ln1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.ln2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class NanoGemma(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.wte = tf.keras.layers.Embedding(config.vocab_size, config.n_embd)
        self.wpe = tf.keras.layers.Embedding(config.max_seq_len, config.n_embd)
        
        self.drop = tf.keras.layers.Dropout(0.1)
        self.h = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.ln_f = tf.keras.layers.LayerNormalization(epsilon=1e-5)

    def call(self, inputs, past=None):
        b, t = inputs.shape
        
        token_embeddings = self.wte(inputs)
        position_embeddings = self.wpe(tf.range(t))
        
        x = self.drop(token_embeddings + position_embeddings)
        
        mask = 1 - tf.linalg.band_part(tf.ones((t, t)), -1, 0)
        mask = tf.expand_dims(tf.expand_dims(mask, 0), 0)
        
        for block in self.h:
            x = block(x, mask)
        
        x = self.ln_f(x)
        
        logits = tf.matmul(x, self.wte.embeddings, transpose_b=True)
        
        return logits

# Example usage
config = NanoGemmaConfig()
model = NanoGemma(config)

# Dummy input
dummy_input = tf.random.uniform((1, 50), minval=0, maxval=config.vocab_size, dtype=tf.int32)
output = model(dummy_input)
print(output.shape)  # Should print (1, 50, 50257) for the default config