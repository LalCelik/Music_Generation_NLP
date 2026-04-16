import torch
import torch.nn as nn
from dataset import vocab
from models.lstm import LSTMModel

# Generation approach adapted from:
# https://www.geeksforgeeks.org/nlp/generating-music-using-abc-notation/
# https://github.com/MITDeepLearning/introtodeeplearning/blob/master/lab1/PT_Part2_Music_Generation.ipynb

def generate(model, vocab, start_string, generation_length, temperature):
    model.eval()

    #encode the seed string into token ids
    input_ids = vocab.encode(start_string)
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)

    generated = list(start_string)

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
        generated.append(next_char)

        #append to input for next step
        next_tensor = torch.tensor([[next_id]], dtype=torch.long)
        input_tensor = torch.cat([input_tensor, next_tensor], dim=1)

    return "".join(generated)


#load saved model and generate
model = LSTMModel(vocab.size, embed_size=64, hidden_size=256)
model.load_state_dict(torch.load("outputs/lstm_model.pt"))

seed = "M:4/4\nK:G\n|"
output = generate(model, vocab, seed, generation_length=200, temperature=1.0)
print("Generated tune:\n" + output)


# quick test
# model = LSTMModel(vocab.size, embed_size=64, hidden_size=256)
# optimizer = Adam(model.parameters(), lr=0.001)
# loss_fn = nn.CrossEntropyLoss()

# #quick training run
# model.train()

# for batch_num, (x, y) in enumerate(train_loader):
#     if batch_num >= 200:
#         break
#     optimizer.zero_grad()
#     logits = model(x)
#     loss = loss_fn(logits.view(-1, vocab.size), y.view(-1))
#     loss.backward()
#     optimizer.step()

# print("Training done")

# #generate from seed
# seed = "M:4/4\nK:G\n|"
# output = generate(model, vocab, seed, generation_length=200, temperature=1.0)
# print("Generated tune:\n" + output)
