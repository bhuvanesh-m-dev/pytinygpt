import torch
import torch.nn as nn

class TinyLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, heads=2, layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, layers)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        x = self.transformer(x)
        return self.fc(x)
