import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = 'cpu'
eval_iters = 200
n_embed = 32
# --------------------

torch.manual_seed(1337)

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# create vocabulary and get size
chars = sorted(list(set(text)))
vocab_size = len(chars)

# simple character to integer encoding
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # take string, return list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # take list of integers, output a string

# format data
data = torch.tensor(encode(text), dtype =torch.long)
# split dataset into train/val split (90%/10%)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    data = train_data if split == 'train' else val_data
    # get batch_size number of starting values randomly from dataset
    # with at least block_size characters of runway
    ix = torch.randint(len(data) - block_size, (batch_size,))

    # stack the values so each sequence is a row vector
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

@torch.no_grad() # tell pytorch we're not going to call gradient to save memory
def estimate_loss():
    # at any time, do eval_iters random splits from train/val sets and
    # evaluate average performance - better than reporting every iteration 
    # or even every ith iteration
    out = {}
    model.eval() # turn model into 'evaluation mode' - does nothing now
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # turn model into 'train mode' - does nothing now
    return out

class Head(nn.Module):
    """ one head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)

        # not a trainable parameter
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    
    def forward(self, x):
        B,T,H = x.shape
        k = self.key(x) # B,T,H
        q = self.query(x) # B,T,H
        v = self.value(x) # B,T,H

        # compute attention scores ('affinities')
        wei = q @ k.transpose(-2, -1) * H**-0.5 # B,T,T
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # B,T,T
        wei = F.softmax(wei, dim=1)

        out = wei @ v # B,T,H
        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

    def forward(self, x):
        # concatenate attention heads' outputs into single vector along channel dimension
        return torch.cat([h(x) for h in self.heads], dim=-1)
    



class FeedForward(nn.Module):
    """ simple linear layer followed by non-linearity """
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed),
            nn.ReLU()
        )
    
    def forward(self, x):
        return self.net(x)
    
class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        # inits simple embedding table
        # each token gets embedding vector
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)

        # embedding for position as well as token embedding
        self.position_embedding_table = nn.Embedding(block_size, n_embed)

        # self-attention heads
        self.sa_heads = MultiHeadAttention(4, n_embed//4) # 4 heads of 8 dimensional self-attention

        # simple feed forward layer to allow network to 'think' on attention output before generating predictions
        self.ffwd = FeedForward(n_embed) 

        # final linear layer converts token embedding to vocab logits
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensor of integers
        B,T = idx.shape
        # B = batch_size
        # T = block_size
        # C = n_embed
        
        # get embeddings for tokens
        tok_emb = self.token_embedding_table(idx) # (B,T,C)

        # get embeddings for positions 
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)

        # combine embeddings and feed to self-attention head
        x = tok_emb + pos_emb

        x = self.sa_heads(x) # apply one head of self-attention (B,T,H)
        x = self.ffwd(x) # feed attention output through feed forward layer (B,T,H)

        # convert to logits
        logits = self.lm_head(x) # (B,T,vocab_size)
        
        if targets is None:
            loss = None
        else:

            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            # compute loss by comparing logit outputs to actual targets
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            
            # positional embeddings only go up to block_size, so have to crop if 
            # input has more than block_size pieces
            idx_cond = idx[:, -block_size:] # get up to last block_size idxs
            logits, loss = self(idx_cond)
            
            # get values for last element in context length
            logits = logits[:, -1, :] # becomes (B, C) 
            
            # get prob distribution for each batch across vocab values
            probs = F.softmax(logits, dim=-1) # (B, C)

            # sample from that distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B,1)

            # append sampled index value to idx tensor
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# Conduct training
for iter in range(max_iters):
    
    # predefined intervals, estimate the model 
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    
    # sample data
    xb, yb = get_batch('train')

    # evaluate loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from model
start = torch.zeros((1,1), dtype=torch.long, device=device) # initialize start of sequence
print(decode(m.generate(start, max_new_tokens=500)[0].tolist()))
