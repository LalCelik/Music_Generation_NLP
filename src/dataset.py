import glob
import kagglehub
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
vocab = sorted(set(joined))
char_index = {}
index_char = {}
for i, char in enumerate(vocab):
    char_index[char] = i
    index_char[i] = char

print("Vocab size: " + str(len(vocab)))

#play the first tune that works
for tune in tunes:
    try:
        s = converter.parse(tune, format="abc")
        s.show("midi")
        break
    except Exception:
        continue

