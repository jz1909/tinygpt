import torch
import torch.nn as nn
from torch.nn import functional as F
from corpus import Corpus
import time

device = 'cpu' 


class attentionHead(nn.Module):

    def __init__(self, n_embd ,head_size):
        super().__init__()
        self.head_size = head_size 

        self.key = nn.Linear(n_embd, self.head_size)
        self.query = nn.Linear(n_embd, self.head_size)
        self.value = nn.Linear(n_embd, self.head_size)

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
        v = self.value(x) #(B, T, head_size)
        out = wei @ v #(B, T, T) @ (B, T, head_size) -> (B, T, head_size)

        return out


class multiheadAttention(nn.Module):
    def __init__(self, n_embd, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([attentionHead(n_embd, head_size) for _ in range(num_heads)])

    def forward(self, x):

        #We assume B, T are the same, and stack each output along the dim - each stack will be able to get a pool of their own softmax to ask their own questions
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        return out


class BigramLM(nn.Module):

    def __init__(self, vocab_size,block_size, n_embd):
        super().__init__()

        # a look up for for the logit row for each token in the vocab
        self.block_size = block_size
        self.n_embd = n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.positional_encoding_table = nn.Embedding(block_size, n_embd)
        #We want n_embd//4 here because then, concatenated together, we get n_embd again, which we need for either residuals or the final linear layer
        self.sa_heads = multiheadAttention(n_embd=n_embd, head_size=n_embd//4, num_heads=4) #(B, T, n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)


    
    def forward(self, idx, targets=None):

        #idx -> (B,T)

        B, T = idx.shape
        token_embeddings = self.token_embedding_table(idx) # (B, T, C) -> (4, 8, C)
        positional_embeddings = self.positional_encoding_table(torch.arange(T, device=device)) # (T, C) -> (8, C)
        x = token_embeddings + positional_embeddings 

        #Now, each token has an embedding representation, but we have to add dataset dependent info to the embedding 

        out = self.sa_heads(x) #(B, T, n_embd) 
        logits = self.lm_head(out) # (B, T, vocab_size) 

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


def train_bigram(model,corpus, batch_size, block_size, lr, epochs):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for _ in range (epochs):

        xb, yb = corpus.get_batch('train', block_size, batch_size)
        xb = xb.to(device)
        yb = yb.to(device)

        logits, loss = model(xb, yb)
        model.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print(f"Final loss: {loss.item()}")


if __name__ == "__main__":

    #hyperparameters
    batch_size = 32
    block_size = 8
    lr = 1e-3
    device='cuda' if torch.cuda.is_available() else 'cpu'
    epochs = 10000

    corpus = Corpus("input.txt")
    model = BigramLM(vocab_size=corpus.tokenizer.vocab_size, block_size=8, n_embd=32).to(device)
    # xb, yb = corpus.get_batch('train', block_size=block_size, batch_size=batch_size)
    # logits, loss = model(xb, yb)
    # print(f"Logits: {logits.shape}, Loss: {loss.item()}")


    start_time = time.time()
    train_bigram(model, corpus, batch_size=batch_size, block_size=block_size, lr=lr, epochs=epochs)
    print(f"Training time: {time.time() - start_time} seconds")

    # # Create a random starting index for the generation, because 0 is our line break
    idx = torch.zeros((1,1), dtype=torch.long, device=device)

    raw_output = model.generate(idx, max_new_tokens=100)
    output = corpus.tokenizer.decode(raw_output[0].tolist())
    print(f"output: {output}")





