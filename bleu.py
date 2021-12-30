from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import time
import math
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
from os import system
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu


class Bleu(object):
    def __init__(self):
        super(Bleu, self).__init__()
        self.o = 0

    def compute_bleu(self, output, reference):
        cc = SmoothingFunction()
        if len(reference) == 3:
            weights = (0.33,0.33,0.33)
        else:
            weights = (0.25,0.25,0.25,0.25)
        return sentence_bleu([reference], output,weights=weights,smoothing_function=cc.method1)
        
    def get_score(self):
        file = open('/home/austin/nctu_hw/DL/DL_hw5/lab5_dataset/test.txt', "rt")
        line = file.readlines()
        reference = []
        for s in line:
            split = s.split('\n')[0].split(' ')
            reference.append(split[1])
        
        file = open('/home/austin/nctu_hw/DL/DL_hw5/pred.txt', "rt")
        line = file.readlines()
        output = []
        for s in line:
            split = s.split('\n')[0].split(' ')[0]
            output.append(split)
        
        score = 0
        for i in range(len(reference)):
            score += self.compute_bleu(output[i], reference[i])

        return score/len(reference)

