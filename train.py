import os
import math
import time
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import wandb
import tiktoken
from tqdm import tqdm

# Configurations
@dataclass
class ModelConfig:
    block_size: int = 512
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.2
    bias: bool = True

@dataclass
class TrainConfig:
    batch_size: int = 12
    gradient_accumulation_steps: int = 5
    learning_rate: float = 6e-4
    max_iters: int = 10000
    lr_decay_iters: int = 10000
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_iters: int = 1000
    eval_interval: int = 100
    save_interval: int = 1000
    eval_iters: int = 200
    log_interval: int = 10

# Attention Module
class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.size()
        
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0)
        
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y

# MLP Module
class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

# Transformer Block
class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT Model
class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx

# Data loading
def get_batch(data, config: TrainConfig, block_size: int, split: str):
    data = data[split]
    ix = torch.randint(len(data) - block_size, (config.batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

def get_lr(it, config: TrainConfig):
    if it < config.warmup_iters:
        return config.learning_rate * it / config.warmup_iters
    if it > config.lr_decay_iters:
        return config.learning_rate * 0.1
    decay_ratio = (it - config.warmup_iters) / (config.lr_decay_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.learning_rate * coeff

def main():
    # Initialize wandb
    wandb.init(project="shakespeare-gpt", name="124M-model")
    
    # Set memory efficiency
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Configurations
    model_config = ModelConfig()
    train_config = TrainConfig()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load and process data
    with open('input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    enc = tiktoken.get_encoding("gpt2")
    data = torch.tensor(enc.encode(text), dtype=torch.long)
    
    # Split data
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    data_dict = {'train': train_data, 'val': val_data}
    
    # Initialize model
    model = GPT(model_config)
    model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        weight_decay=train_config.weight_decay
    )
    
    # Training loop
    best_val_loss = float('inf')
    accumulated_loss = 0
    
    for iter in range(train_config.max_iters):
        lr = get_lr(iter, train_config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter % train_config.eval_interval == 0:
            losses = torch.zeros(train_config.eval_iters)
            model.eval()
            for k in range(train_config.eval_iters):
                X, Y = get_batch(data_dict, train_config, model_config.block_size, split='val')
                X, Y = X.to(device), Y.to(device)
                with torch.no_grad():
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            val_loss = losses.mean()
            
            wandb.log({
                'iter': iter,
                'val_loss': val_loss,
                'lr': lr
            })
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'iter': iter,
                    'best_val_loss': best_val_loss
                }, 'best_model.pt')
            
            model.train()
        
        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        accumulated_loss = 0
        
        for _ in range(train_config.gradient_accumulation_steps):
            X, Y = get_batch(data_dict, train_config, model_config.block_size, split='train')
            X, Y = X.to(device), Y.to(device)
            
            logits, loss = model(X, Y)
            loss = loss / train_config.gradient_accumulation_steps  # Scale loss
            loss.backward()
            accumulated_loss += loss.item() * train_config.gradient_accumulation_steps
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
        optimizer.step()
        
        if iter % train_config.log_interval == 0:
            wandb.log({
                'iter': iter,
                'train_loss': accumulated_loss,
                'lr': lr
            })
            
        if iter % train_config.save_interval == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'iter': iter,
                'best_val_loss': best_val_loss
            }, f'checkpoint_{iter}.pt')
            
            # Generate sample text
            context = "ROMEO: "
            x = torch.tensor([enc.encode(context)], dtype=torch.long, device=device)
            y = model.generate(x, max_new_tokens=100, temperature=0.7, top_k=50)[0]
            completion = enc.decode(y.tolist())
            wandb.log({
                'iter': iter,
                'sample_generation': wandb.Html(f"<pre>{completion}</pre>")
            })

if __name__ == '__main__':
    main() 