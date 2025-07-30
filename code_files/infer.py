import torch
from model import TinyLLM
from tokenizer import Tokenizer

def load_tokenizer():
    token2id = torch.load("vocab.pt")
    tok = Tokenizer("")
    tok.token2id = token2id
    tok.id2token = {i: w for w, i in token2id.items()}
    return tok

def generate(prompt, max_len=40):
    tokenizer = load_tokenizer()
    input_ids = tokenizer.encode(prompt.lower())
    model = TinyLLM(len(tokenizer.token2id))
    model.load_state_dict(torch.load("vocab.pt"))
    model.eval()

    x = torch.tensor(input_ids).unsqueeze(1)
    for _ in range(max_len):
        with torch.no_grad():
            out = model(x)
        next_id = out[-1].argmax().item()
        x = torch.cat([x, torch.tensor([[next_id]])], dim=0)
        if next_id == 0:
            break
    # Decode and format as a clean sentence
    output = tokenizer.decode(x.squeeze().tolist())
    output = output.replace(" <UNK>", "").strip()
    # Capitalize first letter and add period if missing
    if output and not output.endswith('.'):
        output += '.'
    output = output[0].upper() + output[1:] if output else output
    return output
