import torch
import torch.nn as nn


class FeedForwardModel(nn.Module):
    def __init__(self):
        super(FeedForwardModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3072 + 38, 500),
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)
