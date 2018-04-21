import torch.nn as nn


class FCN(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, batch):
        output = self.bn1(self.relu(self.conv1(batch)))
        output = self.conv2(output)
        return output
