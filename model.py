import torch
import torch.nn as nn

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(25, 128), # features.0
            nn.ReLU(),
            nn.Linear(128, 128), # features.3
            nn.ReLU(),
            nn.Linear(128, 64), # features.6
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 64), # classifier.0
            nn.ReLU(),
            nn.Linear(64, 25) # classifier.3
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x