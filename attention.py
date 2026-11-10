import torch
import torch.nn as nn
from torch.nn import functional as F
from corpus import Corpus
import time

device = 'cpu' 
dropout = 0.2


class attentionHead(nn.Module):

    def __init__(self, n_embd ,head_size):
        super().__init__()
        self.head_size = head_size 

        self.key = nn.Linear(n_embd, self.head_size)
        self.query = nn.Linear(n_embd, self.head_size)
        self.value = nn.Linear(n_embd, self.head_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        B, T, _ = x.shape

        #Self Attention
        key = self.key(x) #(B, T, head_size)
        query = self.query(x) #(B, T, head_size)
        #Now this matrix holds value of how tokens attends to each other
        wei = query @ key.transpose(-2, -1) # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(torch.tril(torch.ones((B,T,T), device=device)) == 0, float('-inf'))
        wei = wei / (self.head_size ** 0.5) # We need to scale or softmax will make high values sharpen to one-hot
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x) #(B, T, head_size)
        out = wei @ v #(B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return out


class multiheadAttention(nn.Module):
    def __init__(self, n_embd, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([attentionHead(n_embd, head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        #We assume B, T are the same, and stack each output along the dim - each stack will be able to get a pool of their own softmax to ask their own questions
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        #The reason why we need this is because if we are to combine back with x with addition, the information encoded from the self attention block must align. I.e, the number at 2nd head position encodes its relation with xyz noun, whereas the same position in x might mean something completely different. We didn't have this problem without residual connections because the layers could learn these behaviors by themselves, without conforming back to x. However, we now need a projection layer (one which learns how to map the information in the self attention layer) to help the self attention block conform back to x
        out = self.proj(out)
        out = self.dropout(out)
        return out

class feedforward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.n_embd = n_embd 

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd), #grow the size here for more computation to capture more detail
            nn.ReLU(),
        )

        self.proj = nn.Linear(4*n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.net(x)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class layerNorm3D(nn.Module):

    def __init__(self, dim, eps = 1e-5):
        """
        dim -> dimensions of the n_embds, so that gamma and beta know their size
        eps -> guard against dividing by zero if var is 0
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = torch.mean(x, -1, keepdim=True) #x is (B, T, C), so setting mean dim to -1 is going to mean to (B, T, 1)
        var = torch.var(x, -1, keepdim=True, unbiased=False)
        xhat = ((x - mean)/(torch.sqrt(var+self.eps)))
        out = self.gamma * xhat + self.beta
        return out


class Block(nn.Module):

    def __init__(self, n_embd, num_heads):
        super().__init__()
        #We want n_embd//head_size here because then, concatenated together, we get n_embd again, which we need for either residuals or the final linear layer
        head_size = n_embd//num_heads
        self.sa = multiheadAttention(n_embd, head_size, num_heads)
        self.feedforward = feedforward(n_embd)
        #Layer norm prevents inputs from getting exponentially blown up or killed
        self.layer_norm1 = layerNorm3D(n_embd)
        self.layer_norm2 = layerNorm3D(n_embd)

    def forward(self, x):
        #We prenorm here so the residual connections don't lose the clear gradient highway
        x = x + self.sa(self.layer_norm1(x))
        x = x + self.feedforward(self.layer_norm2(x))

        return x

class BigramLM(nn.Module):

    def __init__(self, num_heads, n_layer, vocab_size,block_size, n_embd):
        super().__init__()

        # a look up for for the logit row for each token in the vocab
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads=num_heads) for _ in range (n_layer)])
        self.layerNorm = layerNorm3D(n_embd)
        self.feedforward = feedforward(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    
    def forward(self, idx, targets=None):

        #idx -> (B,T)

        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B, T, C) -> (4, 8, C)
        positional_embeddings = self.positional_encoding_table(torch.arange(T, device=device)) # (T, C) -> (8, C)
        x = token_embeddings + positional_embeddings 

        #Now, each token has an embedding representation, but we have to add dataset dependent info to the embedding 

        x = self.blocks(x) #(B, T, n_embd) 
        x = self.layerNorm(x) 

        logits = self.lm_head(x) # (B, T, vocab_size) 

        if targets == None:
            loss=None

        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """
        idx: current context of characters in a batch in (B, T) format
        """

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:] # crop idx to the last block_size tokens, or else we keep on putting in longer idx so that pos encoding will overflow
            logits, loss = self.forward(idx_cond)
            logits = logits[:,-1, :] # We only care about the last character in the block size -> (B, C)
            probs = F.softmax(logits, dim=-1) 
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)


        return idx


def train_bigram(model,corpus, batch_size, block_size, lr, epochs, eval_iters, eval_interval):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range (epochs):

        if epoch % eval_interval == 0: 
            losses = estimate_loss(model,corpus, eval_iters, block_size, batch_size)
            print(f"Epoch {epoch} of {epochs}")
            print(f"training loss: {losses['train']}, eval loss: {losses['eval']}")


        xb, yb = corpus.get_batch('train', block_size, batch_size)
        xb = xb.to(device)
        yb = yb.to(device)

        logits, loss = model(xb, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Final training loss: {loss.item()}")

@torch.no_grad
def estimate_loss(model,corpus, eval_iters, block_size, batch_size):
    """
    Quickly hop into eval mode and just average the loss over the past eval_iters
    """

    model.eval()
    out = {} 
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for i in range (eval_iters):
            X, Y = corpus.get_batch(split, block_size, batch_size)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


if __name__ == "__main__":

    #hyperparameters
    batch_size = 64
    block_size = 256
    n_embd = 384
    n_head = 6
    n_layers = 6
    lr = 1e-3
    device='cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10000
    #How many interval per epoch we evaluate
    eval_interval = 300
    #How many iterations per evaluation do we average the loss over
    eval_iters = 200

    corpus = Corpus("input.txt")
    model = BigramLM(num_heads=n_head, n_layer=n_layers, vocab_size=corpus.tokenizer.vocab_size, block_size=block_size, n_embd=n_embd).to(device)
    # xb, yb = corpus.get_batch('train', block_size=block_size, batch_size=batch_size)
    # logits, loss = model(xb, yb)
    # print(f"Logits: {logits.shape}, Loss: {loss.item()}")


    start_time = time.time()
    train_bigram(model, corpus, batch_size=batch_size, block_size=block_size, lr=lr, epochs=epochs, eval_iters=eval_iters, eval_interval=eval_interval)
    print(f"Training time: {time.time() - start_time} seconds")

    # # Create a random starting index for the generation, because 0 is our line break
    idx = torch.zeros((1,1), dtype=torch.long, device=device)

    model.eval()
    raw_output = model.generate(idx, max_new_tokens=1000)
    output = corpus.tokenizer.decode(raw_output[0].tolist())
    print(f"output: {output}")





