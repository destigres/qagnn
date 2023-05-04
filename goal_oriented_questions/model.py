import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn
class QuestionPredictor(nn.Module):

    def __init__(self, input_size=32):
        super(QuestionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size,50)
        self.fc2 = nn.Linear(50,25)
        self.fc3 = nn.Linear(25,1)
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return self.fc3(x)