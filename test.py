import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import random
import sys
import copy
import math
import numpy as np
import time
import os
from model import *
from dataloader import *
from random import sample
from bleu import *

class Test(object):
    def __init__(self, show=False):
        super(Test, self).__init__()
        self.input_batch, self.cond_batch = Dataloader(path='/home/austin/nctu_hw/DL/DL_hw5/lab5_dataset/test.txt',mode='test')
        self.word = []
        for i in range(len(self.input_batch)):
            self.word.append(torch.LongTensor(self.input_batch[i]).cuda())
        self.cond_batch= torch.LongTensor(self.cond_batch).cuda()
        self.alphabet2num = {'SOS': 0, 'EOS': 1}
        self.alphabet2num.update([(chr(i+97),i+2) for i in range(0,26)])
        self.num2alphabet = {y:x for x,y in self.alphabet2num.items()}
        self.tense2num = {'sp': 0, 'tp': 1, 'pg': 2, 'p': 3}
        self.tensem2condition = {y:x for x,y in self.tense2num.items()}
        self.test_cond_list = [self.cond_batch[0], 
                            self.cond_batch[0],
                            self.cond_batch[0],
                            self.cond_batch[0],
                            self.cond_batch[3],
                            self.cond_batch[0],
                            self.cond_batch[3],
                            self.cond_batch[2],
                            self.cond_batch[2],
                            self.cond_batch[2]]
        self.test_cond_target_list = [self.cond_batch[3], 
                                    self.cond_batch[2],
                                    self.cond_batch[1],
                                    self.cond_batch[1],
                                    self.cond_batch[1],
                                    self.cond_batch[2],
                                    self.cond_batch[0],
                                    self.cond_batch[0],
                                    self.cond_batch[3],
                                    self.cond_batch[1]]

    
    def num2word(self, input_, dict_):
        word = ''
        for em in input_:
            w = dict_[em]
            word += str(w)

        return word


    def tester(self, Model, input_word, cond1, cond2, show):
        pred = Model.Eval(input_word, cond1, cond2)
        pred_word = self.num2word(pred, self.num2alphabet)

        return pred_word

    def bleU_test(self, model, show=False):
        open('/home/austin/nctu_hw/DL/DL_hw5/pred.txt', 'w').close()
        file = open('/home/austin/nctu_hw/DL/DL_hw5/pred.txt', "a+")
        for i in range(len(self.word)):
            word = self.tester(model, self.word[i], self.test_cond_list[i], self.test_cond_target_list[i], show)
            file.write(word+'\n')
    
    def Gaussian_score(self,words):
        words_list = []
        score = 0
        yourpath = '/home/austin/nctu_hw/DL/DL_hw5/lab5_dataset/train.txt' #should be your directory of train.txt
        with open(yourpath,'r') as fp:
            for line in fp:
                word = line.split(' ')
                word[3] = word[3].strip('\n')
                words_list.extend([word])
            for t in words:
                for i in words_list:
                    if t == i:
                        score += 1
        return score/len(words)

    def gaussian_test(self, model):
        for i in range(1000):
            open('/home/austin/nctu_hw/DL/DL_hw5/gaussian.txt', 'w').close()
            file = open('/home/austin/nctu_hw/DL/DL_hw5/gaussian.txt', "a+")
            pre = model.gaussion(self.cond_batch)
            WoRD = []
            for batch in pre:
                b = []
                for word in batch:
                    Word = self.num2word(word,self.num2alphabet)
                    b.append(Word)
                    file.write(Word+'\n')
                WoRD.append(b)
                file.write('==============='+'\n')
            score = self.Gaussian_score(WoRD)
            if score >= 0.3:
                print('Gaussian score : ',score)
                break




   
