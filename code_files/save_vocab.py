# save_vocab.py
import torch
from collections import Counter

with open("data/knowledge.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokens = text.split()  # basic whitespace tokenizer
vocab = Counter(tokens)
token2id = {token: idx for idx, (token, _) in enumerate(vocab.items())}
id2token = {idx: token for token, idx in token2id.items()}

# Save it
torch.save(token2id, "vocab.pt")
torch.save(id2token, "vocab_reverse.pt")  # optional
