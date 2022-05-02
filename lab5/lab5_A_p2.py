import torch

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
from utils import getDevice, plot_losses, train, validate, training_loop, train_w_timing
from time import perf_counter
from args import args
import json
from resnet import ResNet50

from datetime import datetime

def main(args):
    print(args)
    print('==> Preparing data..')

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    transform_test = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    testset = torchvision.datasets.CIFAR10(
        root=args.data_path, train=False, download=True, transform=transform_test)
    validloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = getDevice()
    print('[INFO]: Device:', device)
    # Model
    print('==> Building model..')
    model = ResNet50()
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    valid_acc = 0
    train_time_tot = 0
    epoch = 0
    result_json = {}
    train_losses = []
    valid_losses = []
    now = datetime.now()
    while valid_acc < 0.92:
        train_loss, train_time = train_w_timing(model, criterion, optimizer, trainloader, device, scheduler=scheduler, printProgress=False)
        valid_loss, valid_acc = validate(model, criterion, validloader, device)
        train_time_tot += train_time
        epoch += 1
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
    print('Total training time: ', train_time_tot, 'Epoch: ', epoch)
    result_json['train_time_tot'] = train_time_tot
    result_json['epoch'] = epoch
    result_json['train_losses'] = train_losses
    result_json['valid_losses'] = valid_losses
    result_fname = 'start_' + now.strftime("%H_%M_%S")
    with open(result_fname, 'w') as outfile:
        json.dump(result_json, outfile)
if __name__ == '__main__':
    args = args().parse()
    main(args)