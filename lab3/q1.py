from operator import mod
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from utils import progress_bar
from myblocks import Linear_BasicBlock
import numpy as np
import math
import os

x = np.linspace(start=-math.sqrt(7), stop=math.sqrt(7), num=3000, endpoint=True)
x_train = np.array(x, dtype=np.float32).reshape(-1, 1)

y = abs(x)
# y = [a * math.sin(5*a) for a in x]
# y = [(a if a > 0 else 0) + 0.2 * math.sin(5 * a) for a in x]
y_train = np.array(y, dtype=np.float32).reshape(-1, 1)

my_dataset = TensorDataset(Tensor(x_train), Tensor(y_train))
my_dataloader = DataLoader(my_dataset, batch_size=64, shuffle=True, num_workers=2)

class P1(nn.Module):
    def __init__(self):
        super(P1, self).__init__()
        self.inputlayer = Linear_BasicBlock(1, 2)
        self.hiddenlayers = self.make_layer_(8)
        self.outputlayer = Linear_BasicBlock(2, 1)

    def make_layer_(self, numlayers):
        layers = []
        for _ in range(numlayers):
            layers.append(Linear_BasicBlock(2,2))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.inputlayer(x)
        out = self.hiddenlayers(out)
        out = self.outputlayer(out)
        return out

inputDim = 1        # takes variable 'x' 
outputDim = 1       # takes variable 'y'
learningRate = 0.01 
epochs = 1000
model = P1()

##### For GPU #######
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

criterion = torch.nn.MSELoss() 
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(my_dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(my_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        for param in model.parameters():
            print(param)
def test(epoch):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(my_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(my_dataloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    
    
if __name__ == '__main__':
    start_epoch = 0
    for epoch in range(start_epoch, start_epoch+200):
        train(epoch)
        test(epoch)
        scheduler.step()