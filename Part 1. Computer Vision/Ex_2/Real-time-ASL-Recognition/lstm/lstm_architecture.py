import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self,input_size = 258,hidden_size = 160,output_size = 9):
        super(LSTMModel,self).__init__()
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers = 1,batch_first = True)
        self.dropout = nn.Dropout(0.6)
        self.fc1 = nn.Linear(hidden_size,hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size,output_size)
    
    def forward(self,x):
        x, _ = self.lstm(x)
        x = torch.max(x,dim = 1).values
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x 
