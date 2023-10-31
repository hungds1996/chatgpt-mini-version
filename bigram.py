import torch 
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 # number of independent sequences to be processed parallely 
block_size = 8 # maximum context length for predictions
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 32
#----------------------------

torch.manual_seed(20)

# !curl https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt --output "input.txt"
with open("input.txt", 'r', encoding='utf-8') as f:
    text = f.read()
    
print("length of dataset: ", len(text))

# make a list of all unique characters in the dataset
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create mapping from character to index and index to character
stoi = { ch: i for i, ch in enumerate(chars) }
itos = { i: ch for i, ch in enumerate(chars) }

encode = lambda input_string: [stoi[char] for char in input_string] # Encoder: take a string as input, output a list of integer
decode = lambda int_list: ''.join([itos[i] for i in int_list]) # Decoder: take a list of integer as input, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(text))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split: str) -> torch.Tensor:
    """Generate a small batch of data including inputs x and targets y

    Args:
        split (str): "train" or "val"
    """
    data = train_data if split == "train" else val_data
    
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: block_size+i+1] for i in ix])
    
    x, y = x.to(device), y.to(device)
    
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    
    # set model to evaluation phase
    model.eval()
    
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    
    # reset back to training phase
    model.train()
    return out

class BiagramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        self.lm_head = nn.Linear(n_embd, vocab_size) #language modeling head
        
    def forward(self, idx, targets=None):
        """Calculate logits and loss from inputs and targets
        B for batch size, T for block size
        
        Args:
            idx (_type_): (B, T) tensor of integers as input. 
            targets (_type_, optional): (B, T) tensor of interger as corresponding target value. Defaults to None.
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B, T, C= number of embedding)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C = number of embedding)
        x = tok_emb + pos_emb # (B, T, C)
        
        logits = self.lm_head(x) # form (B, T, n_embd) to (B, T, vocab_size)
        
        
        if targets == None:
            loss = None
        else:        
            # cross_entropy function of Pytorch take input with a specific shape so need to reshape logits and targets
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is the array of idices in current context
        for _ in range(max_new_tokens):
            # predictions
            logits, loss = self(idx)
            
            # get only the last time step
            logits = logits[:, -1, :] # shape (B, C)
            
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1) # (B, C)
            
            #sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx
    
model = BiagramLanguageModel()
m = model.to(device)

#Create pytorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for iter in range(max_iters):
    # for every interval evaluate and print loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
    # sample a batch of data
    xb, yb = get_batch('train')
    
    #evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    
context = torch.zeros((1,1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=15000)[0].tolist()))