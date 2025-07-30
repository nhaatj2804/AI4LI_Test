import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_vocab_size = 14, embed_size = 256, hidden_size = 512):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
    
    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, hidden = self.gru(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)
        return outputs, hidden