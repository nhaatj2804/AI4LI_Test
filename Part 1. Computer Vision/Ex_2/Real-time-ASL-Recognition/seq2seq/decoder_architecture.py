import torch
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(self, target_vocab_size = 34, embed_size = 256, hidden_size = 512):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, embed_size)
        self.gru = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, target_vocab_size)
    
    def forward(self, x, hidden):
        x = x.unsqueeze(1)  # batch_size x 1
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        prediction = self.fc(output.squeeze(1))
        return prediction, hidden