import torch
import torch.nn as nn

class HAPTDNN(nn.Module):

    def __init__(self, input_shape = 562 , output_shape = 6):
        super(HAPTDNN, self).__init__()
        self.fc1 = torch.nn.Linear(input_shape, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 16)
        self.fc_f = torch.nn.Linear(16, output_shape)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc_f(x)  # No softmax here
        return x