import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import random
import sys
import copy
import math
import copy
import wandb
import numpy as np
import time
import os
from model import *
from dataloader import *
from random import sample
from test import *
import argparse

parser = argparse.ArgumentParser(description='Set up')
parser.add_argument('--lr', type=float, default = 0.005)
parser.add_argument('--epochs', type=int, default = 500)
parser.add_argument('--hidden_size', type=int, default = 256)
parser.add_argument('--feq', type=float, default = 2.0)
parser.add_argument('--Mode', type=str, default = 'train')
args = parser.parse_args()
print(args)

if args.Mode == 'train':
    wandb.init(project='CVAE')
    wandb.save('/home/austin/nctu_hw/DL/DL_hw5/model.py')
    config = wandb.config
    config.hidden_size = args.hidden_size
    config.epochs = args.epochs
    config.learning_rate = args.lr

    model = CVAE(28, 4, args.hidden_size)
    model = model.cuda()
    wandb.watch(model)
    input_batch, cond_batch = Dataloader('/home/austin/nctu_hw/DL/DL_hw5/lab5_dataset/train.txt','train')
    Batch = []

    for i in range(len(input_batch)):
        input_batch[i] = torch.LongTensor(input_batch[i]).cuda()
        cond_batch[i]= torch.LongTensor([cond_batch[i]]).cuda()
        Batch.append([input_batch[i], cond_batch[i]]) # size : 4908

    criterion = torch.nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    BLeU = Bleu()
    test = Test()

    def gen_teach_ratio(Epoch):
        return 1.0 - (Epoch/args.epochs)
    def sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))
    def gen_KLD_ratio(mode, Epoch):
        if mode == 'mon':
            return (Epoch/args.epochs)*0.25
        else:
            period = args.epochs//args.feq
            Epoch %= period
            ratio = sigmoid((Epoch - period // 2.0) / (period // 10)) / 2.0
            return ratio*0.5


    for epoch in range(args.epochs):
        print('Epoch : ', epoch+1 ,'...',end='\r')
        random.shuffle(Batch)
        Loss = []
        KLD_avg = []

        teach_ratio = gen_teach_ratio(epoch)
        KL_ratio = gen_KLD_ratio('cyc', epoch)

        for sample in Batch:
            loss, KLD = model(sample[0], sample[1], teach_ratio, KL_ratio, criterion, optimizer)
            Loss.append(loss)
            KLD_avg.append(KLD)
        
        CE_loss = sum(Loss)/len(Loss) - (sum(KLD_avg)/len(KLD_avg))*KL_ratio
        model.eval()
        test.bleU_test(model)
        model.train()
        score = BLeU.get_score()

        wandb.log({"teach_ratio": teach_ratio})
        wandb.log({"BLeU": score})
        wandb.log({"KL_ratio": KL_ratio})
        wandb.log({"KLD": sum(KLD_avg)/len(KLD_avg)})
        wandb.log({"loss": sum(Loss)/len(Loss)})
        wandb.log({"CE_loss": CE_loss})
        file = open('/home/austin/nctu_hw/DL/DL_hw5/Record.txt', "a+")
        file.write(str(teach_ratio)+'/'+str(KL_ratio)+'/'+str(sum(Loss)/len(Loss))+'/'+str(CE_loss)+'/'+str(sum(KLD_avg)/len(KLD_avg))+'/'+str(score)+'\n')

        print('Epoch : ',epoch+1,' Loss : ',sum(Loss)/len(Loss),' CE_Loss :', CE_loss, 'KLD_loss : ',sum(KLD_avg)/len(KLD_avg), ' Bleu : ', score)
        if score >= 0.7:
            name = '/home/austin/nctu_hw/DL/DL_hw5/weight/CVAE_'+str(epoch+1)+'_'+str(sum(Loss)/len(Loss))+'_'+str(score)+'.pth'
            torch.save(model.state_dict(), name)
            wandb.save(name)
else:
    model = CVAE(28, 4, args.hidden_size)
    model.load_state_dict(torch.load('/home/austin/nctu_hw/DL/DL_hw5/Result/weight/CVAE_45_0.38194182919856_0.8323583241361134.pth'))
    model = model.cuda()
    model.eval()
    score = 0
    BLeU = Bleu()
    test = Test()
    test.gaussian_test(model)
    for i in range(1000):
        print(i,end='\r')
        test.bleU_test(model)
        score = BLeU.get_score()
        if score >= 0.8:
            print('Bleu score : ', score)
            break

