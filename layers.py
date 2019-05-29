import torch
import torch.nn as nn
import torch.nn.functional as F



class Baseline(nn.Module):
    def __init__(self, input_size):
        super(Baseline, self).__init__()

        self.linear1 = nn.Linear(input_size, 500)
        self.linear2 = nn.Linear(500, 100)
        self.linear3 = nn.Linear(100, 2)

    def forward(self, input):
        input = input.view(input.shape[0], -1)
        out = F.relu(self.linear1(input))
        out = F.relu(self.linear2(out))
        out = F.log_softmax(self.linear3(out))
        return out


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 64,   )