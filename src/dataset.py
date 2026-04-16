"""
data_pipeline.py — shared ABC music data utilities
"""

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader


def load_abc_text(path: str) -> str:
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    texts = []
    for root, _, files in os.walk(path):
        for fname in sorted(files):
            if fname.endswith((".abc", ".txt")):
                with open(os.path.join(root, fname), "r", encoding="utf-8", errors="ignore") as f:
                    texts.append(f.read().strip())
    if not texts:
        raise FileNotFoundError(f"No .abc or .txt files found under: {path}")
    return "\n\n".join(texts)


def clean_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = [l.rstrip() for l in text.split("\n")]
    cleaned, blank_count = [], 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned.append(line)
        else:
            blank_count = 0
            cleaned.append(line)
    return "\n".join(cleaned)


def split_songs(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"\n{2,}", text.strip()) if s.strip()]


def filter_songs(songs: list[str], min_len: int = 50) -> list[str]:
    return [s for s in songs if len(s) >= min_len]


class Vocabulary:
    """
    Character-level vocabulary with a reserved PAD token at index 0.

    Attributes
    ----------
    size      : int              total vocab size (including PAD)
    pad_idx   : int              always 0; use for Transformer attention masks
    char2idx  : dict[str, int]   token -> index
    idx2char  : dict[int, str]   index -> token
    chars     : list[str]        all real (non-PAD) characters, sorted

    Note: batches from ABCDataset are fixed-length and contain no PAD tokens.
    pad_idx is exposed so the Transformer can use it for src_key_padding_mask.
    """

    PAD_TOKEN = "<PAD>"

    def __init__(self, text: str):
        self.pad_idx  = 0
        self.chars    = sorted(set(text))
        self.char2idx = {self.PAD_TOKEN: 0}
        self.char2idx.update({c: i + 1 for i, c in enumerate(self.chars)})
        self.idx2char = {i: c for c, i in self.char2idx.items()}
        self.size     = len(self.char2idx)

    def encode(self, text: str) -> list[int]:
        return [self.char2idx[c] for c in text if c in self.char2idx]

    def decode(self, indices) -> str:
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
        return "".join(
            self.idx2char[i]
            for i in indices
            if i != self.pad_idx and i in self.idx2char
        )

    def __repr__(self):
        return f"Vocabulary(size={self.size}, pad_idx={self.pad_idx})"


class ABCDataset(Dataset):
    """
    Sliding-window next-character prediction dataset.

    Each item is (x, y) where:
      x : LongTensor (seq_len,)   input token indices
      y : LongTensor (seq_len,)   targets shifted right by 1

    Batches from DataLoader are shape (batch_size, seq_len), dtype=torch.long.
    All windows are exactly seq_len tokens — no padding occurs.
    """

    def __init__(self, text: str, vocab: Vocabulary, seq_len: int = 100):
        self.vocab        = vocab
        self.seq_len      = seq_len
        self.data         = torch.tensor(vocab.encode(text), dtype=torch.long)
        self.n_sequences  = len(self.data) - seq_len
        if self.n_sequences <= 0:
            raise ValueError(f"Text too short ({len(self.data)} tokens) for seq_len={seq_len}.")

    def __len__(self) -> int:
        return self.n_sequences

    def __getitem__(self, idx: int):
        x = self.data[idx     : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


def build_dataloaders(
    path:        str,
    seq_len:     int   = 100,
    batch_size:  int   = 64,
    val_split:   float = 0.1,
    num_workers: int   = 0,
    seed:        int   = 42,
):
    """
    path -> (train_loader, val_loader, vocab)

    Vocab quick-reference:
      vocab.size          int
      vocab.pad_idx       int  (0)
      vocab.char2idx      dict[str, int]
      vocab.idx2char      dict[int, str]

    Batch shape: (batch_size, seq_len), dtype=torch.long
    Max sequence length: seq_len (fixed; no padding needed)
    """
    text  = clean_text(load_abc_text(path))
    vocab = Vocabulary(text)
    print(f"[data] {len(text):,} chars | {vocab}")

    full_ds = ABCDataset(text, vocab, seq_len=seq_len)
    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    gen     = torch.Generator().manual_seed(seed)
    train_ds, val_ds = torch.utils.data.random_split(full_ds, [n_train, n_val], generator=gen)

    kw = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return DataLoader(train_ds, shuffle=True, **kw), DataLoader(val_ds, shuffle=False, **kw), vocab


if __name__ == "__main__":
    SAMPLE = "X:1\nT:The Butterfly\nM:9/8\nL:1/8\nK:Emin\n|:B2E G2E F3|B2E G2E FED:|\n\nX:2\nT:Cooley's\nM:4/4\nL:1/8\nK:Edor\n|:eB BB dBAB|eB BB BAFE:|"
    vocab  = Vocabulary(SAMPLE)
    print(vocab)
    ds   = ABCDataset(SAMPLE, vocab, seq_len=10)
    x, y = ds[0]
    print(f"x={vocab.decode(x)!r}  y={vocab.decode(y)!r}")
    loader = DataLoader(ds, batch_size=4)
    xb, yb = next(iter(loader))
    print(f"batch x: {xb.shape}  y: {yb.shape}")
