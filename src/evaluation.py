"""
evaluation.py — shared metrics and visualisations for all three models
"""

import re
import math
import collections
import numpy as np
import matplotlib.pyplot as plt


_NOTE_RE     = re.compile(r"[\^_=]?[A-Ga-g]")
_DURATION_RE = re.compile(r"[\^_=]?[A-Ga-gz](\d*)")
_NOTE_TO_MIDI = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def perplexity_from_loss(avg_loss: float) -> float:
    return math.exp(avg_loss)


def _abc_note_to_semitone(note_char: str) -> int | None:
    if note_char.lower() == "z":
        return None
    semitone = _NOTE_TO_MIDI.get(note_char.upper())
    if semitone is None:
        return None
    return semitone + (12 if note_char.islower() else 0)


def extract_pitches(abc_text: str) -> list[int]:
    return [s for n in _NOTE_RE.findall(abc_text) if (s := _abc_note_to_semitone(n[-1])) is not None]


def extract_durations(abc_text: str) -> list[int]:
    return [int(d) if d else 1 for d in _DURATION_RE.findall(abc_text)]


def extract_steps(pitches: list[int]) -> list[int]:
    return [abs(pitches[i + 1] - pitches[i]) for i in range(len(pitches) - 1)]


def pitch_distribution(abc_text: str) -> dict[int, float]:
    pitches = extract_pitches(abc_text)
    if not pitches:
        return {}
    counter = collections.Counter(pitches)
    total   = sum(counter.values())
    return {k: v / total for k, v in sorted(counter.items())}


def duration_stats(abc_text: str) -> dict:
    arr = np.array(extract_durations(abc_text), dtype=float)
    return {} if arr.size == 0 else {"mean": float(arr.mean()), "std": float(arr.std()), "min": int(arr.min()), "max": int(arr.max())}


def step_stats(abc_text: str) -> dict:
    arr = np.array(extract_steps(extract_pitches(abc_text)), dtype=float)
    return {} if arr.size == 0 else {"mean": float(arr.mean()), "std": float(arr.std()), "min": int(arr.min()), "max": int(arr.max())}


def plot_loss_curves(histories: dict[str, dict], save_path: str | None = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, hist in histories.items():
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], label=name)
        axes[1].plot(epochs, hist["val_loss"],   label=name)
    for ax, title in zip(axes, ["Train Loss", "Validation Loss"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title(title)
        ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) if save_path else plt.show()


def plot_pitch_distributions(texts: dict[str, str], save_path: str | None = None):
    note_names = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B",
                  "c","c#","d","d#","e","f","f#","g","g#","a","a#","b"]
    fig, ax = plt.subplots(figsize=(14, 5))
    x, width = np.arange(24), 0.8 / len(texts)
    for i, (label, text) in enumerate(texts.items()):
        dist = pitch_distribution(text)
        ax.bar(x + i * width, [dist.get(s, 0.0) for s in range(24)], width=width, label=label, alpha=0.8)
    ax.set_xticks(x + width * (len(texts) - 1) / 2)
    ax.set_xticklabels(note_names, fontsize=8)
    ax.set_ylabel("Relative Frequency"); ax.set_title("Pitch Distribution: Training vs Generated")
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150) if save_path else plt.show()


def print_comparison_table(model_names, val_losses, val_ppls, dur_stats_list, step_stats_list):
    header = f"{'Model':<15} {'Val Loss':>10} {'Perplexity':>12} {'Dur Mean':>10} {'Dur Std':>8} {'Step Mean':>11} {'Step Std':>9}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))
    for name, vl, vp, ds, ss in zip(model_names, val_losses, val_ppls, dur_stats_list, step_stats_list):
        print(f"{name:<15} {vl:>10.4f} {vp:>12.2f} {ds.get('mean',0):>10.2f} {ds.get('std',0):>8.2f} {ss.get('mean',0):>11.2f} {ss.get('std',0):>9.2f}")
    print("=" * len(header))


if __name__ == "__main__":
    sample = "|:B2E G2E F3|B2E G2E FED|B2E G2E F3|B3 B3 B3:|"
    print("pitches  :", extract_pitches(sample))
    print("durations:", extract_durations(sample))
    print("dur stats:", duration_stats(sample))
    print("step stats:", step_stats(sample))
