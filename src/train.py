import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import train_loader, val_loader, vocab
from models.lstm import LSTMModel

# hyperparameters
embed_size = 64
hidden_size = 256
num_epochs = 1 #add more when doing full bigger
max_batches = 50  # set to None to train on full dataset
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

    train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0

    for x, y in val_loader:
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab.size), y.view(-1))
            val_loss = val_loss + loss.item()

    val_loss = val_loss / len(val_loader)

    print("Epoch " + str(epoch + 1) + " | Train Loss: " + str(round(train_loss, 4)) + " | Val Loss: " + str(round(val_loss, 4)))
