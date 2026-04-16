"""
train.py the training loop for transformer decoder
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.transformer import TransformerDecoder


def train_one_epoch(model, loader, optimizer, criterion, device, clip_grad=1.0) -> float:
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc="  train", leave=False)
    for x, y in pbar:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
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
    total_loss = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1)).item()
    avg = total_loss / len(loader)
    return avg, math.exp(avg)


def run_training(
    train_loader,
    val_loader,
    vocab,
    d_model=128,
    nhead=4,
    num_layers=4,
    dim_feedforward=512,
    dropout=0.1,
    lr=3e-4,
    epochs=20,
    clip_grad=1.0,
    device_str="auto",
    save_path="transformer_checkpoint.pt",
):
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    print(f"[transformer] device={device}")

    model = TransformerDecoder(
        vocab_size=vocab.size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "val_ppl": []}
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device, clip_grad)
        vl, vp = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        history["val_ppl"].append(vp)
        print(f"Epoch {epoch:3d}/{epochs}  train={tl:.4f}  val={vl:.4f}  ppl={vp:.2f}")

        if vl < best_val:
            best_val = vl
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "vocab_size": vocab.size,
                    "d_model": d_model,
                    "nhead": nhead,
                    "num_layers": num_layers,
                    "dim_feedforward": dim_feedforward,
                },
                save_path,
            )

    print(f"[transformer] best val loss={best_val:.4f}  saved -> {save_path}")
    return model, history


if __name__ == "__main__":
    import argparse
    from src.dataset import build_dataloaders
    from src.generate import generate

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/")
    p.add_argument("--seq_len", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--d_model", type=int, default=128)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dim_feedforward", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--gen_len", type=int, default=1000)
    p.add_argument("--max_chars", type=int, default=None, help="Truncate dataset to first N chars for fast iteration")
    p.add_argument("--save", default="transformer_checkpoint.pt")
    args = p.parse_args()

    train_loader, val_loader, vocab = build_dataloaders(
        args.data, args.seq_len, args.batch_size, max_chars=args.max_chars
    )
    model, history = run_training(
        train_loader,
        val_loader,
        vocab,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        save_path=args.save,
    )

    sample = generate(
        model, vocab, start_string="X:1\n", generation_length=args.gen_len, temperature=args.temperature
    )
    print("\n" + "=" * 60 + "\n" + sample)
    out = args.save.replace(".pt", "_generated.abc")
    with open(out, "w") as f:
        f.write(sample)
    print(f"[transformer] generated ABC -> {out}")
