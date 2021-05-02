import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

import models

def save_model(model):
    state = {
        'acc' : args.acc,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    file_name = '.'.join([args.model, args.dataset, 'pth.tar'])
    torch.save(state, os.path.join('saved_models/', file_name))
    print('save model : {}'.format(file_name))
    

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data))
    return

def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / len(test_loader.dataset)

    if acc > args.acc and not args.evaluate:
        args.acc = acc
        save_model(model)

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%), Best Acc : {}'.format(
        test_loss * args.batch_size, correct, len(test_loader.dataset),
        100. * float(correct) / len(test_loader.dataset), args.acc))
    return

if __name__=='__main__':
    # argument
    parser = argparse.ArgumentParser(description='Pytorch Example')
    parser.add_argument('--dataset', action='store', default='MNIST',
            help='dataset: MNIST |')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
            help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
            help='input batch size for testing (default: 128)')
    parser.add_argument('--model', action='store', default='LeNet5',
            help='dataset: LeNet5 |')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
            help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
            help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
            metavar='W', help='weight decay (default: 5e-4)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
            help='number of epochs to train (default: 200)')
    parser.add_argument('--log-interval', type=int, default=100, metavar='N',
            help='how many batches to wait before logging training status')
    parser.add_argument('--evaluate', action="store_true",
            help='evaluate model')
    parser.add_argument('--pretrained', action='store_true',
            help='pretrained')

    args = parser.parse_args()

    ## accuracy state
    args.acc = 0.0

    ## cuda flag
    args.cuda =  torch.cuda.is_available()
    print(args.cuda)

    ## control random seed [torch, cuda, cuDnn]
    torch.manual_seed(1)

    if args.cuda:
        torch.cuda.manual_seed(1)
        ## There is a problem that the computation processing speed is reduced
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False


    ## load dataset
    if args.dataset == 'MNIST':
        transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,)) ])
        train_data = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('data', train=False, download=True, transform=transform)

        kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, **kwargs)

        num_classes = 10
    

    ## load model
    if args.model == 'LeNet5':
        model = models.LeNet5()
        args.momentum = 0
        args.weight_decay = 0
    
    if args.cuda:
        model.cuda()
    
    ## optimizer & criterion
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()


    ## load pretrained model
    if args.pretrained:
        file_name = '.'.join([args.model, args.dataset, 'pth.tar'])
        pretrained_model = torch.load('saved_models/'+file_name)
        args.acc = pretrained_model['acc']
        model.load_state_dict(pretrained_model['model_state_dict'])
        optimizer.load_state_dict(pretrained_model['optimizer_state_dict'])

    # print the number of model parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print('Total parameter number:', params, '\n')


    ## test
    if args.evaluate:
        test()
        exit()

    ## train
    for epoch in range(1, args.epochs + 1):
        train(epoch)
        test()