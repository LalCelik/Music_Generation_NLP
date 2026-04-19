import torch
from music21 import converter

# Generation approach adapted from:
# https://www.geeksforgeeks.org/nlp/generating-music-using-abc-notation/
# https://github.com/MITDeepLearning/introtodeeplearning/blob/master/lab1/PT_Part2_Music_Generation.ipynb

def generate(model, vocab, start_string, generation_length, temperature):
    model.eval()

    #encode the seed string into token ids
    input_ids = vocab.encode(start_string)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    generated = list(start_string)
    k_count = start_string.count("K:")

    for i in range(generation_length):
        #forward pass
        with torch.no_grad():
            logits = model(input_tensor)

        #take last timestep and add temp
        next_logits = logits[0, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=0)

        #sample next character
        next_id = torch.multinomial(probs, num_samples=1).item()
        next_char = vocab.idx2char[next_id]

        #stop before writing a second K: (new tune starting)
        if next_char == "K" and generated[-1] == "\n" and k_count >= 1:
            break

        generated.append(next_char)

        #append to input for next step
        next_tensor = torch.tensor([[next_id]], dtype=torch.long)
        input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

    return "".join(generated)


def save_midi(output, save_path):
    #save raw abc if the saving doesn't work
    abc_path = save_path.replace(".mid", ".abc")
    with open(abc_path, "w") as f:
        f.write(output)
    print("ABC notation saved to " + abc_path)

    #trim to last complete bar so music21 doesn't fail becuase of incomplete endings
    last_bar = output.rfind("|")
    if last_bar != -1:
        clean = output[:last_bar + 1]
    else:
        clean = output

    try:
        s = converter.parse(clean, format="abc")
        s.write("midi", save_path)
        print("MIDI saved to " + save_path + " = open in GarageBand to hear it")
    except Exception as e:
        print("Could not save MIDI (" + str(e) + ") = use the .abc file instead")
