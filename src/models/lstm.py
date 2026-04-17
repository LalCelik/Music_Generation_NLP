import torch
import torch.nn as nn

#recurrence (1 timestep)
class LSTMCell(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.forget_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_gate = nn.Linear(input_size + hidden_size, hidden_size)

        self.gate_gate = nn.Linear(input_size + hidden_size, hidden_size)
        self.output_gate = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, hidden, cell):
        combined = torch.cat([x, hidden], dim=1)

        f = torch.sigmoid(self.forget_gate(combined)) # f=sigmoid(W[h,x])
        i = torch.sigmoid(self.input_gate(combined)) # i=sigmoid(W[h,x])
        g = torch.tanh(self.gate_gate(combined)) # g=tanh(W[h,x])
        o = torch.sigmoid(self.output_gate(combined)) # o=sigmoid(W[h,x])

        # c=f*c+i*g
        # h=o*tanh(c)
        new_cell = f * cell + i * g
        new_hidden = o * torch.tanh(new_cell)
        

        return new_hidden, new_cell

#model that uses the cell
class LSTMModel(nn.Module):

    def __init__(self, vocab_size, embed_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm_cell = LSTMCell(embed_size, hidden_size)
        self.output_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        batch_size = x.size(0)
        seq_len = x.size(1)

        hidden = torch.zeros(batch_size, self.hidden_size)
        cell = torch.zeros(batch_size, self.hidden_size)

        embedded = self.embedding(x)

        outputs = []
        for t in range(seq_len):
            
            x_t = embedded[:, t, :]
            hidden, cell = self.lstm_cell(x_t, hidden, cell)
            outputs.append(hidden)

        outputs = torch.stack(outputs, dim=1)
        logits = self.output_head(outputs)
        return logits
