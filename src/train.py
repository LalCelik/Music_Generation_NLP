import sys
import torch
import torch.nn as nn
from torch.optim import Adam
from dataset import train_loader, val_loader, test_loader, vocab, joined
from evaluation import plot_loss_curves, perplexity, plot_pitch_distribution, plot_step_distribution
from generate import generate, save_midi

# hyperparams
embed_size = 64
hidden_size = 256
num_epochs = 10
max_batches = None
learning_rate = 0.001

#early stopping
patience = 3

def train(model, train_loader, val_loader, num_epochs, patience, max_batches, learning_rate, save_path, log=print):

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    # model, loss, optimizer
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
        log("Epoch " + str(epoch + 1) + " | Train Loss: " + str(round(train_loss, 4)) + " | Val Loss: " + str(round(val_loss, 4)))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            log("Val loss improved, model saved")
        else:
            epochs_no_improve = epochs_no_improve + 1
            log("No improvement for " + str(epochs_no_improve) + " epoch(s)")

        if epochs_no_improve >= patience:
            log("Early stopping at epoch " + str(epoch + 1))
            break

    log("Saved to " + save_path)
    return train_losses, val_losses, best_val_loss

#run function
def run(model_name):

    #write to a log file as well as printing
    log_file = open("outputs/" + model_name + "_log.txt", "w")
    def log(msg):
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()


    if model_name == "lstm":
        from models.lstm import LSTMModel
        model = LSTMModel(vocab.size, embed_size, hidden_size)

    elif model_name == "rnn":
        from models.rnn import RNNModel
        model = RNNModel(vocab.size, embed_size, hidden_size)

    elif model_name == "transformer":
        from models.transformer import TransformerModel
        model = TransformerModel(vocab.size, embed_size, hidden_size)

    else:
        print("Unknown model: " + model_name + ". Choose from: lstm, rnn, transformer")
        return

    train_losses, val_losses, best_val_loss = train(
        model, train_loader, val_loader,
        num_epochs, patience, max_batches, learning_rate,
        save_path="outputs/" + model_name + "_model.pt",
        log=log
    )

    #plot loss curves
    plot_loss_curves(train_losses, val_losses, "outputs/" + model_name + "_loss_curves.png", model_name=model_name.upper())
    log("Loss curve saved to outputs/" + model_name + "_loss_curves.png")

    #print final val perplexity
    log("Val perplexity: " + str(round(perplexity(best_val_loss), 4)))

    #evaluate on test set using best saved model
    model.load_state_dict(torch.load("outputs/" + model_name + "_model.pt"))
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    test_loss = 0

    for x, y in test_loader:
        with torch.no_grad():
            logits = model(x)
            loss = loss_fn(logits.view(-1, vocab.size), y.view(-1))
            test_loss = test_loss + loss.item()

    test_loss = test_loss / len(test_loader)
    log("Test perplexity: " + str(round(perplexity(test_loss), 4)))

    #generate a sample and plot pitch distribution
    seed = "M:4/4\nK:G\n|"
    generated_text = generate(model, vocab, seed, generation_length=500, temperature=1.0)

    plot_pitch_distribution(joined, generated_text, "outputs/" + model_name + "_pitch_distribution.png")
    log("Distribution of pitch is saved to outputs/" + model_name + "_pitch_distribution.png")

    #plot step distribution
    plot_step_distribution(joined, generated_text, "outputs/" + model_name + "_step_distribution.png")
    log("Distribution of steps is saved to outputs/" + model_name + "_step_distribution.png")

    save_midi(generated_text, "outputs/" + model_name + "_generated.mid")
    log_file.close()


model_names = sys.argv[1:]

if len(model_names) == 0:
    print("Please pick models and run: python3 train.py lstm")
else:
    for name in model_names:
        run(name)
