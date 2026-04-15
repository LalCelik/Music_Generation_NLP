import kagglehub
from music21 import converter

# Download latest version
#abc notation
path = kagglehub.dataset_download("juansebm/abc-notation-music-for-rnn")

print("Path to dataset files:", path)

#Play the file
s = converter.parse(path, format="abc")
s[0].show("midi")   # opens MIDI player

