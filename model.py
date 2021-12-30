import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
import sys
import math
import random
import numpy as np
import time
from torch.nn import init
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CVAE(nn.Module):
    class EncoderRNN(nn.Module):
        def __init__(self, input_size, input_cond_size, hidden_size):
            super(CVAE.EncoderRNN, self).__init__()
            self.hidden_size = hidden_size
            self.embedding_N = nn.Embedding(input_size, hidden_size)
            self.rnn_N = nn.LSTM(hidden_size, hidden_size)
            for weight in self.rnn_N.parameters():
                if len(weight.size()) > 1:
                    init.orthogonal_(weight.data)

        def initial(self):
            h = torch.zeros(1,1,self.hidden_size-8, device=device)
            c = torch.zeros(1,1,self.hidden_size, device=device)
            
            return h ,c

        def forward(self, input, hidden, cell):
            embedded = self.embedding_N(input).view(1, 1, -1)
            embedded = embedded.permute(1,0,2)
            
            output, (h, c) = self.rnn_N(embedded, (hidden, cell))

            return output, h, c

    class DecoderRNN(nn.Module):
        def __init__(self, input_size, input_cond_size, hidden_size):
            super(CVAE.DecoderRNN, self).__init__()
            self.hidden_size = hidden_size
            self.embedding_D = nn.Embedding(input_size, hidden_size)
            self.rnn_D = nn.LSTM(hidden_size, hidden_size)
            for weight in self.rnn_D.parameters():
                if len(weight.size()) > 1:
                    init.orthogonal_(weight.data)

        def initial(self):
            h = torch.zeros(1,1,self.hidden_size, device=device)
            c = torch.zeros(1,1,self.hidden_size, device=device)
            
            return h, c

        def forward(self,input, hidden, cell):
            embedded = self.embedding_D(input).view(1, 1, -1)
            embedded = embedded.permute(1,0,2)

            output, (h, c) = self.rnn_D(embedded, (hidden, cell))

            return output, (h, c)

    def __init__(self, input_size, input_cond_size, hidden_size):
        super(CVAE,self).__init__()
        self.input_size = input_size
        self.encoder = self.EncoderRNN(input_size, input_cond_size, hidden_size).cuda()
        self.decoder = self.DecoderRNN(input_size, input_cond_size, hidden_size).cuda()
        self.embedding_cond = nn.Embedding(input_cond_size, 8)
        self.fc_meam_h = nn.Linear(hidden_size, 32)
        self.fc_logvar_h = nn.Linear(hidden_size, 32)
        self.fc_st_D = nn.Linear(40, hidden_size)
        self.fc_out = nn.Linear(hidden_size, input_size)


    def reparameterize(self, mean, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        z = eps.mul(std).add_(mean)

        return z


    def forward(self, input_word, cond, teach_ratio, KLD_ratio, criterion, optimizer):
        loss = 0
        optimizer.zero_grad()
        
        # concatenate conditional to hidden unit 0
        Cond = self.embedding_cond(cond)
        hidden_n, cell_n = self.encoder.initial()
        hidden_n = torch.cat((hidden_n, Cond.view(1, 1, -1)),2)
        
        # Encoder
        for alphabet in input_word:
            _, hidden_n, cell_n = self.encoder(alphabet, hidden_n, cell_n)
        
        # get mean and variance
        h_t = torch.squeeze(hidden_n)
        mean_h = self.fc_meam_h(h_t)
        logvar_h = self.fc_logvar_h(h_t)

        KLD_loss = -0.5 * torch.sum(1 + logvar_h - mean_h.pow(2) - logvar_h.exp())
        
        # sample
        latent_h = self.reparameterize(mean_h, logvar_h)

        pred = None
        out_fc = None

        _, cell_d = self.decoder.initial()
        decoder_input = torch.tensor([[[0]]]).cuda()
        # concatenate conditional to latent vector
        hidden_d = latent_h.view(1, -1)
        hidden_d = torch.cat((hidden_d, Cond),1)
        hidden_d = self.fc_st_D(hidden_d).view(1, 1, -1)
        
        # Decoder
        for idx in range(input_word.shape[0]):
            use_teacher_forcing = True if random.random() < teach_ratio else False
            if (use_teacher_forcing) and (idx !=0):
                output_d, (hidden_d, cell_d) = self.decoder(input_word[idx - 1], hidden_d, cell_d)
            else:
                output_d, (hidden_d, cell_d) = self.decoder(decoder_input, hidden_d, cell_d)
            
            
            out_fc = self.fc_out(output_d) # classification
            out_alphabet = torch.argmax(out_fc).item() # return to alphabet code
            decoder_input = torch.tensor([[[out_alphabet]]]).cuda()

            if idx == 0:
                pred = out_fc.view(1,self.input_size)
            else:
                pred = torch.cat((pred, out_fc.view(1,self.input_size)),0)


        loss = criterion(pred, input_word.long())
        loss += KLD_loss*KLD_ratio
        loss.backward()
        optimizer.step()

        return loss.item(), KLD_loss.item()

    def Eval(self, input_word, cond1, cond2):
        with torch.no_grad():
            hidden_n, cell_n = self.encoder.initial()
            Cond1 = self.embedding_cond(cond1).view(1, 1, -1)
            hidden_n = torch.cat((hidden_n, Cond1),2)

            for alphabet in input_word:
                _, hidden_n, cell_n = self.encoder(alphabet, hidden_n, cell_n)
            
            h_t = torch.squeeze(hidden_n)
            mean_h = self.fc_meam_h(h_t)
            logvar_h = self.fc_logvar_h(h_t)
            
            latent_h = self.reparameterize(mean_h, logvar_h)
            
            pred_word = []
            out_fc = None

            _, cell_d = self.decoder.initial()
            decoder_input = torch.tensor([[[0]]]).cuda()
            Cond2 = self.embedding_cond(cond2).view(1, 1, -1)
            hidden_d = latent_h.view(1, 1, -1)
            hidden_d = torch.cat((hidden_d, Cond2),2)
            hidden_d = self.fc_st_D(hidden_d).view(1, 1, -1)

            for i in range(input_word.shape[0]):
                output_d, (hidden_d, cell_d) = self.decoder(decoder_input, hidden_d, cell_d)
                out_fc =  self.fc_out(output_d)
                out_alphabet = torch.argmax(out_fc).item()
                if out_alphabet == 1:
                    break
                decoder_input = torch.tensor([[[out_alphabet]]]).cuda()
                pred_word.append(out_alphabet)
            
        return pred_word

    def gaussion(self, Cond):
        Result = []
        with torch.no_grad():
            for i in range(100):
                print(i,end='\r')
                latent = torch.randn_like(torch.zeros(1, 1, 32)).cuda()
                pred_batch = []

                for cond in Cond:
                    word = []
                    _, cell_d = self.decoder.initial()
                    out_fc = None
                    decoder_input = torch.tensor([[[0]]]).cuda()
                    Cond_ = self.embedding_cond(cond.long()).view(1, 1, -1)

                    hidden_d = latent.view(1, 1, -1)
                    hidden_d = torch.cat((hidden_d, Cond_),2)
                    hidden_d = self.fc_st_D(hidden_d).view(1, 1, -1)

                    for i in range(100):
                        output_d, (hidden_d, cell_d) = self.decoder(decoder_input, hidden_d, cell_d)
                        out_fc =  self.fc_out(output_d)
                        out_alphabet = torch.argmax(out_fc).item()
                        if out_alphabet == 1:
                            break
                        decoder_input = torch.tensor([[[out_alphabet]]]).cuda()
                        word.append(out_alphabet)
                    pred_batch.append(word)

                Result.append(pred_batch)

        return Result







        

        

