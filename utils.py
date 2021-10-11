#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchvision import datasets, transforms
from torch.autograd import Variable

def getData(name='cifar10', train_bs=128, test_bs=1000, data_dir='', arch='resnet', train_length=1):
    """
    Get the dataloader
    """
    if name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)

        train_length = int(len(trainset) * train_length)
        trainset, _ = torch.utils.data.random_split(trainset, [train_length, len(trainset) - train_length]) 

        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)
    if name == 'cifar10_without_dataaugmentation':
        transform_train = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])

        trainset = datasets.CIFAR10(root='../data',
                                    train=True,
                                    download=True,
                                    transform=transform_train)
        train_loader = torch.utils.data.DataLoader(trainset,
                                                   batch_size=train_bs,
                                                   shuffle=True)

        testset = datasets.CIFAR10(root='../data',
                                   train=False,
                                   download=False,
                                   transform=transform_test)
        test_loader = torch.utils.data.DataLoader(testset,
                                                  batch_size=test_bs,
                                                  shuffle=False)

    if name == 'imagenet':

        train_resolution = 224
        if arch == "inceptionv3":
            train_resolution = 299

        test_resolution = (256, 224)
        if arch == 'inceptionv3':
            test_resolution = (342, 299)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])

        train_dataset = datasets.ImageFolder(
            os.path.join(data_dir, 'train'),
            transforms.Compose([
                transforms.RandomResizedCrop(train_resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        
        train_length = int(len(train_dataset) * train_length)
        train_dataset, _ = torch.utils.data.random_split(train_dataset, [train_length, len(train_dataset) - train_length])             

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=train_bs, shuffle=True,
            num_workers=8, pin_memory=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(data_dir, 'val'), transforms.Compose([
                transforms.Resize(test_resolution[0]),
                transforms.CenterCrop(test_resolution[1]),
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=test_bs, shuffle=False,
            num_workers=8, pin_memory=True)


    return train_loader, test_loader


def test(model, test_loader, cuda=True):
    """
    Get the test performance
    """
    model.eval()
    correct = 0
    total_num = 0
    for data, target in test_loader:
        if cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        pred = output.data.max(
            1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()
        total_num += len(data)
    print('testing_correct: ', correct / total_num, '\n')
    return correct / total_num
