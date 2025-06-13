# Script created for modularity using the "gpt2-text-generation.ipynb" notebook as a reference
# ---------------------------------------

# Imports
import requests
import os

import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32 
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu' #device agnostic code
eval_iters = 200

# Read the data
with open('data/input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Tokenize the input text, create mapping from characters to integers
stoi = { ch:i for i, ch in enumerate(chars)}
itos = { i:ch for i, ch in enumerate(chars)}
encode = lambda x: [stoi[c] for c in x]  
decode = lambda y: ''.join([itos[i] for i in y])

# Create train and test splits
data = torch.tensor(encode(text), dtype=torch.int64)
n = int(0.9*len(data)) #90-10 split for train-val
train_data = data[:n]
val_data = data[n:]


# Load the data
def get_batch(split):
    """
    Generates a small batch of data of inputs x and targets y
    """
    data = train_data if split == 'train' else val_data
    ix = torch.randint(0, len(data) - block_size, (batch_size,))  #generate random positions to grab chunk out of  | torch.randint(low, high, (size:tuple))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  #offset by 1

    x, y = x.to(device), y.to(device)

    return x, y

# Loss function to average loss over multiple batches 
@torch.no_grad() # Context manager for telling PyTorch that anything under this would not be called during the backpropagation phase (loss.backward())
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Bigram Language Model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) 

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1) 
            idx = torch.cat((idx, idx_next), dim=1)
        return idx
    

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# Create optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # Evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    xb, yb = get_batch('train')

    # Evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
