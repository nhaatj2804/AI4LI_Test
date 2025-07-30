
import torch
import torch.nn as nn

class MLPModel(nn.Module):
    def __init__(self, input_size=21*3*2, hidden_size=64, num_classes=24):
        super(MLPModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
        
    def forward(self, x):
        return self.layers(x)