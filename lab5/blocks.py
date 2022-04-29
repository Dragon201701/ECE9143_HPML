import torch
import torch.nn as nn
import torch.nn.functional as F

class Linear_BasicBlock(nn.Module):
    # Basic Linear layer with batch norm and relu.
    def __init__(self, inputSize, outputSize):
        super(Linear_BasicBlock, self).__init__()
        self.linear = nn.Linear(inputSize, outputSize)
        # self.bn = nn.BatchNorm2d(outputSize)
    
    def forward(self, x):
        out = F.relu(self.linear(x))
        return out