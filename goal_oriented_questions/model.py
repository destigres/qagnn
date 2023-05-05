import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn
class QuestionPredictor(nn.Module):

    def __init__(self, input_size=32):
        super(QuestionPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size,800)
        self.fc2 = nn.Linear(800,700)
        self.fc3 = nn.Linear(700,400)
        self.fc4 = nn.Linear(400,200)
        self.fc5 = nn.Linear(200,50)
        self.fc6 = nn.Linear(50,1)
    def forward(self, x):
        x =  F.relu(self.fc1(x))
        x =  F.relu(self.fc2(x))
        x =  F.relu(self.fc3(x))
        x =  F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        return x