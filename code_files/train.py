import torch
import torch.nn as nn
from tokenizer import Tokenizer
from model import TinyLLM

def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower()

def get_batches(data, seq_len=32):
    for i in range(0, len(data) - seq_len - 1):
        x = torch.tensor(data[i:i+seq_len])
        y = torch.tensor(data[i+1:i+seq_len+1])
        yield x, y

def train():
    text = load_text("data/knowledge.txt")
    tokenizer = Tokenizer(text)
    encoded = tokenizer.encode(text)

    model = TinyLLM(len(tokenizer.token2id))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(10):
        for x, y in get_batches(encoded):
            x = x.unsqueeze(1)
            y = y.unsqueeze(1)
            out = model(x)
            loss = loss_fn(out.view(-1, out.size(-1)), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "model.pt")
    torch.save(tokenizer.token2id, "vocab.pt")

if __name__ == "__main__":
    train()
