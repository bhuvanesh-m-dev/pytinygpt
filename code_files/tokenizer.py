from collections import Counter

class Tokenizer:
    def __init__(self, text, vocab_size=5000):
        tokens = text.split()
        vocab = Counter(tokens).most_common(vocab_size)
        self.token2id = {w: i+1 for i, (w, _) in enumerate(vocab)}
        self.token2id["<UNK>"] = 0
        self.id2token = {i: w for w, i in self.token2id.items()}

    def encode(self, text):
        return [self.token2id.get(w, 0) for w in text.split()]

    def decode(self, ids):
        return ' '.join([self.id2token.get(i, "<UNK>") for i in ids])
