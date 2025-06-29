import torch
import torch.nn as nn

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.fc1 = nn.Linear(25, 128) # Input: 5x5 board = 25 features
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 25) # Output: 25 possible moves

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x