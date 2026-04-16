import os
import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import train_loader, val_loader, vocab
from models.lstm import LSTMModel

save_path = "outputs/lstm_model.pt"

# hyperparams
embed_size = 64
hidden_size = 256
num_epochs = 10
max_batches = None
learning_rate = 0.001

# model, loss, optimizer
model = LSTMModel(vocab.size, embed_size, hidden_size)
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)

# training loop
for epoch in range(num_epochs):

    model.train()
    train_loss = 0

    #for x, y in train_loader:
    for batch_num, (x, y) in enumerate(train_loader):
        if max_batches is not None and batch_num >= max_batches:
            break
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits.view(-1, vocab.size), y.view(-1))
        loss.backward()
        optimizer.step()
        train_loss = train_loss + loss.item()

    if max_batches is not None:
        num_train_batches = min(max_batches, len(train_loader))
    else:
        num_train_batches = len(train_loader)
    train_loss = train_loss / num_train_batches

    model.eval()
    val_loss = 0

    for x, y in val_loader:
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab.size), y.view(-1))
            val_loss = val_loss + loss.item()

    val_loss = val_loss / len(val_loader)

    print("Epoch " + str(epoch + 1) + " | Train Loss: " + str(round(train_loss, 4)) + " | Val Loss: " + str(round(val_loss, 4)))

#save model after training
os.makedirs(os.path.dirname(save_path), exist_ok=True)
torch.save(model.state_dict(), save_path)
print("Saved to " + save_path)
