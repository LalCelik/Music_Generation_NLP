"""
generate.py  
the autoregressive text generation for the Transformer decoder
"""

import torch


@torch.no_grad()
def generate(
    model,
    vocab,
    start_string: str = "X",
    generation_length: int = 1000,
    temperature: float = 1.0,
    device=None,
) -> str:
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    generated = list(start_string)
    input_ids = torch.tensor(vocab.encode(start_string), dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(generation_length):
        logits = model(input_ids)
        next_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated.append(vocab.idx2char[next_id.item()])
        input_ids = torch.cat([input_ids, next_id], dim=1)

    return "".join(generated)
