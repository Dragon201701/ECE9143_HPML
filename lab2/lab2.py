import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from resnet import ResNet18
from utils import progress_bar

from time import perf_counter


# Training
def train(net, device, trainloader, criterion, optimizer, epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    best_train_acc = 0 # best train accuracy
    time_loading = 0
    time_training = 0
    trainloader_iter = iter(trainloader)
    # for batch_idx, (inputs, targets) in enumerate(trainloader):
    train_loss_total = 0
    for batch_idx in range(len(trainloader)):
        time_loading_start = perf_counter()
        inputs, targets = next(trainloader_iter)
        inputs, targets = inputs.to(device), targets.to(device)
        time_loading_end = perf_counter()
        time_loading += time_loading_end - time_loading_start

        time_training_start = perf_counter()
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        time_training_end = perf_counter()
        time_training += time_training_end - time_training_start
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        train_acc = 100.*correct/total
        best_train_acc = max(best_train_acc, train_acc)
        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    print('Avg train loss: ', train_loss/len(trainloader))
    print('Best train acc: %.3f%%' % best_train_acc)
    return time_loading, time_training
best_acc = 0  # best test accuracy

def test(net, device, testloader, criterion, epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

def get_optimizer(args, net):
#   if optim == 'adam':
#     optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
#   else:
#     # optim == 'sgd':
    print("Optmizer: ", args.optim)
    if args.optim == 'sgd':
        print('Using SGD')
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
    elif args.optim == 'sgd_nes':
        print('Using SGD with Nesterov')
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif args.optim == 'adagrad':
        print('Using Adagrad.')
        optimizer = torch.optim.Adagrad(net.parameters(), lr=args.lr, weight_decay=5e-4)
    elif args.optim == 'adadelta':
        print('Using adadelta')
        optimizer = torch.optim.Adadelta(net.parameters(), lr=args.lr, rho=0.9, eps=1e-06)
    elif args.optim == 'adam':
        print('Using Adam.')
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
    else:
        print('Using SGD')
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr,
                            momentum=0.9, weight_decay=5e-4)
    return optimizer

def main():

    parser = argparse.ArgumentParser(description='ECE9143 Lab2 ResNet18 using CIFAR10 dataset yl7897')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--optim', default='sgd')
    parser.add_argument('--num_workers', default=2, type=int, help='number of workers of train and test loader')
    parser.add_argument('--data_path', default='./data')
    parser.add_argument('--cpu_only', action='store_true', help='Use GPU by default without this option.')
    args = parser.parse_args()
    

    device = 'cuda' if torch.cuda.is_available() and not args.cpu_only else 'cpu'
    
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=args.num_workers)

    testset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=args.num_workers)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = ResNet18()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(args, net)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    time_loading_list = []
    time_training_list = []
    for epoch in range(0, 5):
        time_loading, time_training = train(net, device, trainloader, criterion, optimizer, epoch)
        time_loading_list.append(time_loading)
        time_training_list.append(time_training)
        test(net, device, testloader, criterion, epoch)
        scheduler.step()
    print('Loading time for each epoch: ', time_loading_list)
    print('Total loading time for all epoch: ', sum(time_loading_list), ' Avg for each epoch: ', sum(time_loading_list)/5)
    print('Training time for each epoch: ', time_training_list, 'Average training time: ', sum(time_training_list)/5)
    print("Running time for each epoch: ", [a + b for a, b in zip(time_loading_list, time_training_list)])
    print("Total running time:", sum(time_loading_list) + sum(time_training_list))
if __name__ == '__main__':
    main()