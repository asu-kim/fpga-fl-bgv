import torch
import torch.nn as nn
from torch.quantization import QuantStub, DeQuantStub

class QuantLeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2) # 2 avg_pools in LeNet have same params
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.avg_pool(x)

        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        x = self.dequant(x)
        return x
