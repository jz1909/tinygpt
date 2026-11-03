import torch
import torch.nn as nn
from torch.nn import functional as F
from corpus import Corpus
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BigramLM(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()

        # a look up for for the logit row for each token in the vocab
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    
    def forward(self, idx, targets=None):

        logits = self.token_embedding_table(idx) # (B, T, C) -> (4, 8, C)

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
            logits, loss = self.forward(idx)
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

    corpus = Corpus("input.txt")
    model = BigramLM(corpus.tokenizer.vocab_size).to(device)
    xb, yb = corpus.get_batch('train', block_size=8, batch_size=4)
    # logits, loss = model(xb, yb)
    # print(f"Logits: {logits.shape}, Loss: {loss.item()}")

    # # Create a random starting index for the generation, because 0 is our line break
    idx = torch.zeros((1,1), dtype=torch.long, device=device)

    start_time = time.time()
    train_bigram(model, corpus, batch_size=4, block_size=8, lr=1e-3, epochs=10000)
    print(f"Training time: {time.time() - start_time} seconds")

    raw_output = model.generate(idx, max_new_tokens=100)
    output = corpus.tokenizer.decode(raw_output[0].tolist())
    print(f"output: {output}")





