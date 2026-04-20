"""
rnn_model.py — Vanilla RNN for character-level ABC music generation
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class VanillaRNN(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 64, hidden_size: int = 256, num_layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.vocab_size  = vocab_size
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.embedding   = nn.Embedding(vocab_size, embed_dim)
        self.rnn         = nn.RNN(embed_dim, hidden_size, num_layers, batch_first=True,
                                  dropout=dropout if num_layers > 1 else 0.0, nonlinearity="tanh")
        self.fc          = nn.Linear(hidden_size, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        emb    = self.embedding(x)
        out, h = self.rnn(emb, hidden)
        return self.fc(out)

    def init_hidden(self, batch_size: int, device: torch.device):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)


def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad=1.0) -> float:
    model.train()
    total_loss, hidden = 0.0, None
    pbar = tqdm(loader, desc="  train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        if hidden is None or hidden.size(1) != x.size(0):
            hidden = model.init_hidden(x.size(0), device)
        optimizer.zero_grad()
        logits, hidden = model(x, hidden)
        loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimizer.step()
        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, hidden = 0.0, None
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if hidden is None or hidden.size(1) != x.size(0):
            hidden = model.init_hidden(x.size(0), device)
        logits, hidden = model(x, hidden)
        total_loss += criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1)).item()
    avg = total_loss / len(loader)
    return avg, math.exp(avg)


@torch.no_grad()
def generate(model, vocab, start_string: str = "X", generation_length: int = 1000, temperature: float = 1.0, device=None) -> str:
    if device is None:
        device = next(model.parameters()).device
    model.eval()
    input_ids = torch.tensor(vocab.encode(start_string), dtype=torch.long, device=device).unsqueeze(0)
    hidden    = model.init_hidden(1, device)
    generated = list(start_string)
    if input_ids.size(1) > 1:
        _, hidden = model(input_ids[:, :-1], hidden)
    current = input_ids[:, -1:]
    for _ in range(generation_length):
        logits, hidden = model(current, hidden)
        probs   = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated.append(vocab.idx2char[next_id.item()])
        current = next_id
    return "".join(generated)


def run_training(train_loader, val_loader, vocab, embed_dim=64, hidden_size=256, num_layers=1,
                 dropout=0.0, lr=0.005, epochs=20, clip_grad=5.0, device_str="auto", save_path="rnn_checkpoint.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device_str == "auto" else torch.device(device_str)
    print(f"[rnn] device={device}")

    model     = VanillaRNN(vocab.size, embed_dim, hidden_size, num_layers, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    history   = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_val  = float("inf")

    for epoch in range(1, epochs + 1):
        tl       = train_one_epoch(model, train_loader, optimizer, criterion, device, clip_grad)
        vl, vp   = evaluate(model, val_loader, criterion, device)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_ppl"].append(vp)
        print(f"Epoch {epoch:3d}/{epochs}  train={tl:.4f}  val={vl:.4f}  ppl={vp:.2f}")
        if vl < best_val:
            best_val = vl
            torch.save({"epoch": epoch, "model_state": model.state_dict(),
                        "vocab_size": vocab.size, "embed_dim": embed_dim,
                        "hidden_size": hidden_size, "num_layers": num_layers}, save_path)

    print(f"[rnn] best val loss={best_val:.4f}  saved → {save_path}")
    return model, history


RNNModel = VanillaRNN



if __name__ == "__main__":
    import argparse
    from src.dataset import build_dataloaders

    p = argparse.ArgumentParser()
    p.add_argument("--data",        default="data/")
    p.add_argument("--seq_len",     type=int,   default=100)
    p.add_argument("--batch_size",  type=int,   default=64)
    p.add_argument("--embed_dim",   type=int,   default=64)
    p.add_argument("--hidden_size", type=int,   default=256)
    p.add_argument("--num_layers",  type=int,   default=1)
    p.add_argument("--epochs",      type=int,   default=20)
    p.add_argument("--lr",          type=float, default=0.005)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen_len",     type=int,   default=1000)
    p.add_argument("--save",        default="rnn_checkpoint.pt")
    args = p.parse_args()

    train_loader, val_loader, vocab = build_dataloaders(args.data, args.seq_len, args.batch_size)
    model, history = run_training(train_loader, val_loader, vocab, args.embed_dim, args.hidden_size,
                                  args.num_layers, lr=args.lr, epochs=args.epochs, save_path=args.save)

    sample = generate(model, vocab, start_string="X:1\n", generation_length=args.gen_len, temperature=args.temperature)
    print("\n" + "="*60 + "\n" + sample)
    out = args.save.replace(".pt", "_generated.abc")
    with open(out, "w") as f:
        f.write(sample)
    print(f"[rnn] generated ABC → {out}")
