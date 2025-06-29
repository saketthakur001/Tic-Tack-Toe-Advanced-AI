import torch
import torch.nn as nn

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # features.0
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), # features.2
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), # features.4
            nn.ReLU(),
            nn.Conv2d(128, 512, kernel_size=5, padding=0) # This layer reduces to 1x1 spatial, outputting 512 channels
        )
        # After features, the output is (batch_size, 512, 1, 1), which flattens to 512
        self.classifier = nn.Sequential(
            nn.Linear(512, 512), # classifier.0
            nn.ReLU(),
            nn.Linear(512, 25) # classifier.2 (final output for 25 moves)
        )

    def forward(self, x):
        # Reshape input from (batch_size, 25) to (batch_size, 1, 5, 5) for CNN
        x = x.view(-1, 1, 5, 5)
        x = self.features(x)
        x = torch.flatten(x, 1) # Flatten the output of the features block
        x = self.classifier(x)
        return x
