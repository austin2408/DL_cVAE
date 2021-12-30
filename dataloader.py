import random
import sys
import copy
import math
import copy
import numpy as np
import time
import os

def Dataloader(path,mode='train'):
    file = open(path, "rt")
    line = file.readlines()
    
    tense2num = {'sp': 0, 'tp': 1, 'pg': 2, 'p': 3}
    condition_list = list(tense2num.values())

    alphabet2num = {'SOS': 0, 'EOS': 1}
    alphabet2num.update([(chr(i+97),i+2) for i in range(0,26)])

    input_batch = []
    cond_batch = []

    if mode == 'test':
      for i in range(4):
          cond_batch.append(condition_list[i])

    for s in line:
      # read word each line
      word = s.split('\n')[0].split(' ')
      if mode == 'train':
          tmp = [word[0], word[1], word[2], word[3]]
      else:
          tmp = [word[0]]
      
      # encode word to numbers
      for n in range(len(tmp)):
          encode = []
          for id in tmp[n]:
            encode.append(alphabet2num[id])
            
          encode.append(1)
          input_batch.append(encode)
          
          if mode == 'train':
              cond_batch.append(condition_list[n])
    
    return input_batch, cond_batch