#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
from random import random

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
import numpy as np
from torchvision import datasets, transforms
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.sampling import mnist_iid, mnist_noniid, cifar_iid
from utils.options20 import args_parser
from models.Update import LocalUpdate
from models.Nets import MLP, CNNMnist, CNNCifar, CNNCifarRes, CNNCifarRes18
from utils.averagingSGD import quan_average_gradients_Imperfect_CSI, quan_average_gradients_AWGN, \
    quan_average_gradients_Fading, quan_average_gradients_EF

from models.test import test_img

import time
import logging

# 创建一个logger
logger = logging.getLogger('mytest')
logger.setLevel(logging.DEBUG)

# 创建一个handler，用于写入日志文件

# fh = logging.FileHandler('logger_{:.4f}.log'.format(time.time()))
fh = logging.FileHandler('20_Fading_SGD_0.25_{:.4f}.log'.format(time.time()))

fh.setLevel(logging.DEBUG)

# 再创建一个handler，用于输出到控制台
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# 定义handler的输出格式
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)

# 给logger添加handler
logger.addHandler(fh)
logger.addHandler(ch)

# logger.info('fed_learn_mnist_cnn_100_iid_v2')
logger.info('fed_learn_cifar_cnn')

if __name__ == '__main__':
    # parse args
    args = args_parser()
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    # import pdb
    # pdb.set_trace()
    # print(torch.__version__)

    # load dataset and split users
    if args.dataset == 'mnist':
        trans_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        dataset_train = datasets.MNIST('../data/mnist/', train=True, download=True, transform=trans_mnist)
        dataset_test = datasets.MNIST('../data/mnist/', train=False, download=True, transform=trans_mnist)
        # sample users
        if args.iid:
            dict_users = mnist_iid(dataset_train, args.num_users)
        else:
            dict_users = mnist_noniid(dataset_train, args.num_users)
    elif args.dataset == 'cifar':
        trans_cifar = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset_train = datasets.CIFAR10('../data/cifar', train=True, download=True, transform=trans_cifar)
        dataset_test = datasets.CIFAR10('../data/cifar', train=False, download=True, transform=trans_cifar)
        if args.iid:
            dict_users = cifar_iid(dataset_train, args.num_users)
        else:
            exit('Error: only consider IID setting in CIFAR10')
    else:
        exit('Error: unrecognized dataset')
    img_size = dataset_train[0][0].shape

    # build model
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'resnet18' and args.dataset == 'cifar':
        net_glob = CNNCifarRes18(args=args).to(args.device)
    elif args.model == 'mlp':
        len_in = 1
        for x in img_size:
            len_in *= x
        net_glob = MLP(dim_in=len_in, dim_hidden=200, dim_out=args.num_classes).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)
    net_glob.train()

    w_glob = net_glob.state_dict()

    # training
    cv_loss, cv_acc = [], []
    val_loss_pre, counter = 0, 0
    net_best = None
    best_loss = None
    val_acc_list, net_list = [], []

    # 记录日志
    logger.info(args)
    optimizer = torch.optim.SGD(net_glob.parameters(), lr=args.lr, momentum=args.momentum)
    # optimizer = torch.optim.Adam(net_glob.parameters(), lr=args.lr)
    if args.lr_scheduler:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 200], gamma=0.1)

    net_total_params = sum(p.numel() for p in net_glob.parameters())
    print('| net_total_params:', net_total_params)

    if args.dataset == 'mnist':
        log_probs_dummy = net_glob(torch.ones(1, 1, 28, 28).to(args.device))
    else:
        log_probs_dummy = net_glob(torch.ones(1, 3, 32, 32).to(args.device))
    loss_dummy = F.cross_entropy(log_probs_dummy, torch.ones(1, ).cuda().long())
    loss_dummy.backward()
    optimizer.zero_grad()

    for iter in range(1, args.epochs + 1):
        w_locals, loss_locals = [], []
        buffer_locals = []
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        for idx in idxs_users:
            local = LocalUpdate(args=args, dataset=dataset_train, idxs=dict_users[idx])
            w, buffer, loss = local.train(net=copy.deepcopy(net_glob).to(args.device))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
            buffer_locals.append(copy.deepcopy(buffer))
            # del w, buffer, loss
        # update global weights
        if args.mode == 'EF':
            w_glob = quan_average_gradients_EF(w_locals, args.snr)
        elif args.mode == 'AWGN':
            w_glob = quan_average_gradients_AWGN(w_locals, args.snr)
        elif args.mode == 'P_CSI':
            w_glob = quan_average_gradients_Fading(w_locals, snr=args.snr, g_th=args.thd)
        elif args.mode == 'NP_CSI':
            w_glob = quan_average_gradients_Imperfect_CSI(w_locals, snr=args.snr, delta=args.delta, g_th=args.thd)
        else:
            raise NotImplementedError

        for key, value in net_glob.named_parameters():
            value.grad.data = w_glob[key].data.detach()


        def average_buffer(w, layer):
            w_avg = copy.deepcopy(w[0][layer])
            for k in w_avg.keys():
                for i in range(1, len(w)):
                    w_avg[k] += w[i][layer][k]
                w_avg[k] = torch.div(w_avg[k], len(w))
            return w_avg


        for (key, module) in net_glob.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                buffer_avg = average_buffer(buffer_locals, key)
                module._buffers['running_mean'].data = buffer_avg['running_mean'].data
                module._buffers['running_var'].data = buffer_avg['running_var'].data
                module._buffers['num_batches_tracked'].data = buffer_avg['num_batches_tracked'].data

        optimizer.step()


        if args.lr_scheduler:
            scheduler.step()

        loss_avg = sum(loss_locals) / len(loss_locals)

        logger.info('Epoch: {}'.format(iter))
        logger.info('Train loss: {:.4f}'.format(loss_avg))

        del w_locals, loss_locals, buffer_locals
        # testing
        if iter % 1 == 0:
            acc_train, loss_train = test_img(net_glob, dataset_train, args)
            acc_test, loss_test = test_img(net_glob, dataset_test, args)

            logger.info("average train acc: {:.2f}%".format(acc_train))
            logger.info("average train loss: {:.4f}".format(loss_train))

            logger.info("average test acc: {:.2f}%".format(acc_test))
            logger.info("average test loss: {:.4f}".format(loss_test))
