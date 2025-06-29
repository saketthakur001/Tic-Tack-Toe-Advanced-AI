import torch
import torch.nn as nn

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(25, 256) # Input: 5x5 board = 25 features, first hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128) # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, 25) # Output: 25 possible moves

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
