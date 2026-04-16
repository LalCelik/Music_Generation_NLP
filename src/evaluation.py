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
