"""
evaluate_transformer.py the total evaluation for a trained transformer checkpoint
This loads a saved checkpoint, then runs validation metrics, then generates sample ABC text
From there, it produces all of comparison plots and stats by using the shared evaluation module
"""

import argparse
import torch
import torch.nn as nn

from src.dataset import build_dataloaders
from src.models.transformer import TransformerDecoder
from src.generate import generate
from src.evaluation import (
    perplexity_from_loss,
    pitch_distribution,
    duration_stats,
    step_stats,
    plot_pitch_distributions,
    print_comparison_table,
)


def load_model(checkpoint_path: str, device: torch.device) -> tuple:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = TransformerDecoder(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        nhead=ckpt["nhead"],
        num_layers=ckpt["num_layers"],
        dim_feedforward=ckpt["dim_feedforward"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    return model, ckpt


@torch.no_grad()
def compute_val_loss(model, loader, device) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total += criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1)).item()
    return total / len(loader)


def run_eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device}")

    _, val_loader, vocab = build_dataloaders(
        args.data, seq_len=args.seq_len, batch_size=args.batch_size
    )

    model, ckpt = load_model(args.checkpoint, device)
    print(f"[eval] loaded checkpoint from epoch {ckpt['epoch']}")

    val_loss = compute_val_loss(model, val_loader, device)
    val_ppl = perplexity_from_loss(val_loss)
    print(f"[eval] val_loss={val_loss:.4f}  perplexity={val_ppl:.2f}")

    training_text = open(args.data).read() if args.data.endswith((".abc", ".txt")) else ""

    generated_text = generate(
        model, vocab,
        start_string="X:1\n",
        generation_length=args.gen_len,
        temperature=args.temperature,
    )
    print(f"\n[eval] generated {len(generated_text)} chars")
    print("=" * 60)
    print(generated_text[:500])
    print("=" * 60)

    gen_dur = duration_stats(generated_text)
    gen_step = step_stats(generated_text)
    print(f"\n[eval] generated duration stats: {gen_dur}")
    print(f"[eval] generated step stats:     {gen_step}")

    if training_text:
        train_dur = duration_stats(training_text)
        train_step = step_stats(training_text)
        print_comparison_table(
            ["Training Data", "Transformer"],
            [0.0, val_loss],
            [0.0, val_ppl],
            [train_dur, gen_dur],
            [train_step, gen_step],
        )
        plot_pitch_distributions(
            {"Training Data": training_text, "Transformer": generated_text},
            save_path=args.plot_path,
        )
        print(f"[eval] pitch distribution plot saved -> {args.plot_path}")

    out = args.checkpoint.replace(".pt", "_generated.abc")
    with open(out, "w") as f:
        f.write(generated_text)
    print(f"[eval] generated ABC saved -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="transformer_checkpoint.pt")
    p.add_argument("--data", default="data/")
    p.add_argument("--seq_len", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--gen_len", type=int, default=1000)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--plot_path", default="pitch_distribution.png")
    run_eval(p.parse_args())
