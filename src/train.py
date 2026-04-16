import os
import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import train_loader, val_loader, vocab, joined
from models.lstm import LSTMModel
from evaluation import plot_loss_curves, perplexity, plot_pitch_distribution
from generate import generate

save_path = "outputs/lstm_model.pt"

# hyperparams
embed_size = 64
hidden_size = 256
num_epochs = 10
max_batches = None
learning_rate = 0.001

train_losses = []
val_losses = []

#early stopping
patience = 3
best_val_loss = float("inf")
epochs_no_improve = 0

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

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print("Epoch " + str(epoch + 1) + " | Train Loss: " + str(round(train_loss, 4)) + " | Val Loss: " + str(round(val_loss, 4)))

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), save_path)
        print("Val loss improved, model saved")
    else:
        epochs_no_improve = epochs_no_improve + 1
        print("No improvement for " + str(epochs_no_improve) + " epoch(s)")

    if epochs_no_improve >= patience:
        print("Early stopping at epoch " + str(epoch + 1))
        break

print("Saved to " + save_path)

#plot loss curves
plot_loss_curves(train_losses, val_losses, "outputs/loss_curves.png")
print("Loss curve saved to outputs/loss_curves.png")

#print final perplexity
print("Final perplexity: " + str(round(perplexity(best_val_loss), 4)))

#generate a sample and plot pitch distribution
seed = "M:4/4\nK:G\n|"
generated_text = generate(model, vocab, seed, generation_length=500, temperature=1.0)

plot_pitch_distribution(joined, generated_text, "outputs/pitch_distribution.png")
print("Distribution of pitch is saved to outputs/pitch_distribution.png ")
