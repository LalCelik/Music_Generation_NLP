<<<<<<< HEAD
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
=======
import math
import re
import matplotlib.pyplot as plt

# ABC notation has lower and higher octave notes
# it has G and g for example
# g is 12 semitones higher than G (an octave)
# so we need to add 12 for those higher notes

def perplexity(avg_loss):
    #perplexity is exp of avg cross entropy loss
    return math.exp(avg_loss)


NOTE_TO_SEMITONE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

def extract_pitches(abc_text):
    pitches = []
    for char in abc_text:
        if char.upper() in NOTE_TO_SEMITONE:
            semitone = NOTE_TO_SEMITONE[char.upper()] #capitalization

            if char.islower():
                semitone = semitone + 12
            pitches.append(semitone)

    return pitches

#steps are the intervals in semitones between consecutive notes
def extract_steps(abc_text):
    pitches = extract_pitches(abc_text)
    steps = []
    for i in range(len(pitches) - 1):
        step = abs(pitches[i + 1] - pitches[i])
        steps.append(step)
    return steps


#plot step interval distribution, compare training vs generated
def plot_step_distribution(training_text, generated_text, save_path):
    step_labels = list(range(13))

    train_steps = extract_steps(training_text)
    gen_steps = extract_steps(generated_text)

    train_counts = [0] * 13
    gen_counts = [0] * 13

    for s in train_steps:
        if s <= 12:
            train_counts[s] = train_counts[s] + 1

    for s in gen_steps:
        if s <= 12:
            gen_counts[s] = gen_counts[s] + 1

    #normalize to relative frequency
    train_total = sum(train_counts)
    gen_total = sum(gen_counts)

    for i in range(13):
        if train_total > 0:
            train_counts[i] = train_counts[i] / train_total
        if gen_total > 0:
            gen_counts[i] = gen_counts[i] / gen_total

    width = 0.4
    train_x = []
    gen_x = []
    for v in step_labels:
        train_x.append(v - width / 2)
        gen_x.append(v + width / 2)

    plt.figure(figsize=(10, 5))
    plt.bar(train_x, train_counts, width=width, label="Training", alpha=0.8)
    plt.bar(gen_x, gen_counts, width=width, label="Generated", alpha=0.8)
    plt.xticks(step_labels)
    plt.xlabel("Interval (semitones)")
    plt.ylabel("Relative Frequency")
    plt.title("Step dist: Training vs Generated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)


#plotting
#plot the number of times each pitch is in the music (ex. #times A is in the music)
#compare training and model generated to see if it is accurate
def plot_pitch_distribution(training_text, generated_text, save_path):
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
                  "c", "c#", "d", "d#", "e", "f", "f#", "g", "g#", "a", "a#", "b"]

    train_pitches = extract_pitches(training_text)
    gen_pitches = extract_pitches(generated_text)

    train_counts = [0] * 24
    gen_counts = [0] * 24

    for p in train_pitches:
        if p < 24:
            train_counts[p] = train_counts[p] + 1

    for p in gen_pitches:
        if p < 24:
            gen_counts[p] = gen_counts[p] + 1

    #normalize to relative frequency
    train_total = sum(train_counts)
    gen_total = sum(gen_counts)

    for i in range(24):
        if train_total > 0:
            train_counts[i] = train_counts[i] / train_total
        if gen_total > 0:
            gen_counts[i] = gen_counts[i] / gen_total

    x = list(range(24))

    width = 0.4

    plt.figure(figsize=(14, 5))
    train_x = []
    gen_x = []
    for v in x:
        train_x.append(v - width / 2)
        gen_x.append(v + width / 2)

    plt.bar(train_x, train_counts, width=width, label="Training", alpha=0.8)
    plt.bar(gen_x, gen_counts, width=width, label="Generated", alpha=0.8)
    plt.xticks(x, note_names, fontsize=8)
    plt.ylabel("Relative Frequency")
    plt.title("Pitch dist: Training vs Generated")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)

#plot the loss results
def plot_loss_curves(train_losses, val_losses, save_path, model_name="Model"):

    epochs = list(range(1, len(train_losses) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train loss")
    plt.plot(epochs, val_losses, label="Val loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(model_name + " Training and Validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
>>>>>>> 60b2d19 (a lot of files)
