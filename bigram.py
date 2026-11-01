import torch
import torch.nn as nn
from torch.nn import functional as F

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

