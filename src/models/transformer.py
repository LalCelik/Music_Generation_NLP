"""
transformer.py 
The transformer decoder for the character level ABC music generation
"""

import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TransformerModel(nn.Module):
    """
    the causal transformer decoder for the nexttoken prediction. The constructor signature
    matches the LSTMmodel /RNNModel: TransformerModel(vocab_size, embed_size, hidden_size)
    The forward method takes (batch, seq_len) and returns (batch, seq_len, vocab_size).
    """

    def __init__(self, vocab_size, embed_size, hidden_size, nhead=4, num_layers=4, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = embed_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_encoder = PositionalEncoding(embed_size, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=nhead,
            dim_feedforward=hidden_size,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(embed_size, vocab_size)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.embedding.weight, std=0.02)
        nn.init.normal_(self.fc_out.weight, std=0.02)
        nn.init.zeros_(self.fc_out.bias)

    @staticmethod
    def _causal_mask(seq_len, device):
        return nn.Transformer.generate_square_subsequent_mask(seq_len, device=device)

    def forward(self, x):
        """
        x : LongTensor (batch, seq_len)
        it returns : (batch, seq_len, vocab_size) logits
        """
        seq_len = x.size(1)
        mask = self._causal_mask(seq_len, x.device)

        emb = self.pos_encoder(self.embedding(x) * math.sqrt(self.d_model))
        out = self.transformer_decoder(emb, emb, tgt_mask=mask, memory_mask=mask)
        return self.fc_out(out)
