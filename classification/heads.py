import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, input_size, output_size):
        super(Head, self).__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        x = F.relu(x)
        return x