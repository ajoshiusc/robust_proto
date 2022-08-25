'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
import pdb

from models import *
from utils import progress_bar
from prototypical_loss import prototypical_loss as loss_fn
from prototypical_batch_sampler import PrototypicalBatchSampler
from datasets.cifar10 import IMBALANCECIFAR10, CIFAR10_LT
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument("--loss", required=False, default='PRO', choices=['CE', 'PRO'])
parser.add_argument('--n_support', default=24, type=int, help='number of support samples')
parser.add_argument('--n_query', default=24, type=int, help='number of query samples in each batch')
parser.add_argument('--batch_size', default=480, type=int, help='batch size')
parser.add_argument('--classes', default=10, type=int, help='number of classes')
parser.add_argument('--epochs', default=80, type=int, help='number of training epochs')
parser.add_argument('--start_epoch', default=0, type=int, help='start epoch')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = args.start_epoch  # start from epoch 0 or last checkpoint epoch

def init_sampler(labels, mode, total_len):
    if 'train' in mode:
        classes_per_it = args.classes
        num_samples = args.n_support + args.n_query
    else:
        classes_per_it = args.classes
        num_samples = args.n_support + args.n_query

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=int(total_len/num_samples))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# trainset = torchvision.datasets.CIFAR10(
#     root='./data', train=True, download=True, transform=transform_train)
# trainset = IMBALANCECIFAR10(root='./data', imb_type='exp', imb_factor=0.01, rand_number=0, train=True, download=True, transform=transform_train)

LT_dataset = CIFAR10_LT(distributed=False, root='./data', imb_type='exp', imb_factor=0.1, batch_size=args.batch_size, num_works=2)
trainloader = LT_dataset.train_balance
# train_sampler = init_sampler(trainset.targets, 'train', len(trainset))
# trainloader = torch.utils.data.DataLoader(trainset, batch_sampler=train_sampler)
# trainloader = torch.utils.data.DataLoader(
#     trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

# testset = torchvision.datasets.CIFAR10(
#     root='./data', train=False, download=True, transform=transform_test)

# test_sampler = init_sampler(testset.targets, 'val', len(testset))
# testloader = torch.utils.data.DataLoader(testset, batch_sampler=test_sampler)
# testloader = torch.utils.data.DataLoader(
#     testset, batch_size=100, shuffle=False, num_workers=2)
testloader = LT_dataset.eval

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
net = SimpleDLA()
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


def calc_cls_acc(predicted, targets):
    acc_cls = []
    for i in range(args.classes):
        corrects = predicted[targets.eq(i).nonzero()[args.n_support:]] == i
        acc_cls.append(corrects.float().mean().item())
    return np.array(acc_cls)


# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    ##==================WarmUp====================
    # if epoch < 10:
    #     args.loss = 'CE'
    # else:
    #     args.loss = 'PRO'
        
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        ##==========
        args.n_support = inputs.shape[0] // 20
        ##==========
        optimizer.zero_grad()
        outputs = net(inputs)
        if args.loss=='CE':
            loss = criterion(outputs, targets)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            acc = calc_cls_acc(predicted, targets)
            correct += predicted.eq(targets).sum().item()
            acc_mean = np.mean(acc)
            
        else:
            loss, acc = loss_fn(outputs, targets, args.n_support)
            loss = loss.to(device)
            acc_mean = acc.mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f'
                     % (train_loss/(batch_idx+1), acc_mean))
        # print(acc)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            if args.loss == 'CE':
                loss = criterion(outputs, targets)
                _, predicted = outputs.max(1)
                total += targets.size(0)

                acc_test = calc_cls_acc(predicted, targets)

                correct += predicted.eq(targets).sum().item()
                # acc_test_mean = 1.0 * correct/total
                acc_test_mean = np.mean(acc_test)
            else:
                loss, acc_test = loss_fn(outputs, targets, args.n_support)
                acc_test_mean = acc_test.mean()
                # loss = loss.to(device)
            test_loss += loss.item()
            #_, predicted = outputs.max(1)

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f'
                         % (test_loss/(batch_idx+1), acc_test_mean))
            print("Accuracy per class", acc_test)
            
    # Save checkpoint.
    acc = acc_test.mean()
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

    print("Best test accuracy: %.3f" % best_acc)

for epoch in range(start_epoch, start_epoch + args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
