import glob
import kagglehub
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from music21 import converter

# Download latest version
#abc notation
path = kagglehub.dataset_download("juansebm/abc-notation-music-for-rnn")

print("Path to dataset files:", path)

abc_file = glob.glob(path + "/**/*.txt", recursive=True)[0]

with open(abc_file, "r", encoding="utf-8", errors="ignore") as f:
    raw = f.read()

tunes = []
for t in raw.split("X:"):
    if t.strip():
        tunes.append("X:" + t.strip())

# Preprocessing
# take metadata lines from each tune
# This code is based on/adapted from this example:
# https://www.geeksforgeeks.org/nlp/generating-music-using-abc-notation/
def preprocess(tune):
    lines = tune.strip().split("\n")
    clean = []
    for line in lines:
        if not line.startswith("X:") and not line.startswith("T:") and not line.startswith("S:") and not line.startswith("%"):
            clean.append(line)
    return "\n".join(clean)

cleaned_tunes = []
for t in tunes:
    result = preprocess(t)
    if result.strip():
        cleaned_tunes.append(result)

print("Total tunes: " + str(len(cleaned_tunes)))
print("Sample tune:\n" + cleaned_tunes[0])

# combine all the tunes into one string
joined = "\n\n".join(cleaned_tunes)
print("Total characters: " + str(len(joined)))

# build vocabulary of unique characters
# reserve index 0 for PAD token
vocab = sorted(set(joined))
char_index = {"<PAD>": 0}
index_char = {0: "<PAD>"}
for i, char in enumerate(vocab):
    char_index[char] = i + 1
    index_char[i + 1] = char

vocab_size = len(char_index)
print("Vocab size: " + str(vocab_size))

# wraped into object so it is compatible with the other models
class Vocab:
    def __init__(self, char_index, index_char):
        self.char2idx = char_index
        self.idx2char = index_char
        self.size = len(char_index)
        self.pad_idx = 0

vocab = Vocab(char_index, index_char)

# encode the full text as a list of integers
encoded = []
for char in joined:
    encoded.append(char_index[char])

print("Encoded length: " + str(len(encoded)))

# slice into fixed-length windows
seq_len = 100
inputs = []
targets = []

for i in range(len(encoded) - seq_len):
    input_seq = encoded[i : i + seq_len]
    target_seq = encoded[i + 1 : i + seq_len + 1]
    inputs.append(input_seq)
    targets.append(target_seq)

print("Total sequences: " + str(len(inputs)))

# PyTorch Dataset
class MusicDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        x = torch.tensor(self.inputs[index], dtype=torch.long)
        y = torch.tensor(self.targets[index], dtype=torch.long)
        return x, y

dataset = MusicDataset(inputs, targets)
print("Dataset size: " + str(len(dataset)))

# train / val / test split (80 / 10 / 10)
n_total = len(dataset)
n_train = int(n_total * 0.8)
n_val = int(n_total * 0.1)
n_test = n_total - n_train - n_val

train_set, val_set, test_set = random_split(dataset, [n_train, n_val, n_test])

print("Train size: " + str(len(train_set)))
print("Val size: " + str(len(val_set)))
print("Test size: " + str(len(test_set)))

# DataLoaders
batch_size = 64
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#play the first tune that works
for tune in tunes:
    try:
        s = converter.parse(tune, format="abc")
        s.show("midi")
        break
    except Exception:
        continue

