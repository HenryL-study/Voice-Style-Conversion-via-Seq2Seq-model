import numpy as np
import torch.nn as nn
import torch
from torch.nn import Sequential
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence
from torch.autograd import Variable
from LockedDropout import LockedDropout
from random import randint
from bisect import bisect_left
import random

class Speller(nn.Module):
    def __init__(self, context_size = 128, rnn_hidden_size = 256, class_size = 40, useLockDrop = False):
        super().__init__()
        self.class_size = class_size
        self.rnn_hidden_size = rnn_hidden_size
        self.layers = 3
        # self.embedding = nn.Embedding(class_size, rnn_hidden_size)
        self.outfc = nn.Linear(rnn_hidden_size + context_size, 256)
        self.a = nn.LeakyReLU(negative_slope=0.2)
        self.out = nn.Linear(256, class_size)
        # self.out.weight = self.embedding.weight.t()
        # print("Output weight size: ", self.out.weight.size())
        self.arnn = nn.LSTMCell(class_size + context_size, rnn_hidden_size)
        self.rnn1 = nn.LSTMCell(rnn_hidden_size, rnn_hidden_size)
        self.rnn2 = nn.LSTMCell(rnn_hidden_size, rnn_hidden_size)
    
    def forward(self, listener_state, listener_len, listener_last_state, max_iters, attention, batch_size = 32, trancripts=None, has_trans = False, greedy_sample = True, teacher_force_rate=0.9, blank_symbol = 0):
        '''
        max_iters       : time_steps
        trancripts      : Batch_size * seq_len (for each = [x x x x x [<blank_symbol>] ])
        '''
        beam_size = 16
        train_flag = False
        if has_trans:
            max_iters = trancripts.size()[1]+1
            train_flag = True
        raw_predict_output = []
        predict_seq = []
        attention_scores = []
        speller_state = self.get_initial(batch_size, listener_last_state)
        current_word = torch.zeros((batch_size, self.class_size), dtype = torch.float).cuda() # <start>
        if not greedy_sample: 
            predict_seq = [[[torch.zeros((batch_size,), dtype = torch.long).cuda()], self.get_initial(batch_size, listener_last_state), 1] for _ in range(beam_size)]
        for step in range(max_iters):
            #print(step)
            if not greedy_sample:
                # ERROR
                raise NotImplementedError
            else:
                current_embed = current_word # self.embedding(current_word)
                # print(current_embed.size())
                _, predict_output, speller_state, att_score = self.run_once(attention, listener_state, listener_len, speller_state, current_embed, batch_size)
                # record score
                raw_predict_output.append(predict_output)
                attention_scores.append(att_score)
                if train_flag and random.random() < teacher_force_rate and step+1<max_iters:
                    current_word = trancripts[:, step, :]
                else:
                    current_word = predict_output #torch.argmax(predict_output, dim=1)

                predict_seq.append(current_word)
        
        if not greedy_sample:
            predict_seq = predict_seq[0][0]
        # print(predict_seq)
        # exit(0)
        return torch.stack(raw_predict_output, dim=1), torch.stack(predict_seq, dim=1), attention_scores  # B * L * C, B * L * C, L * B * S

    def run_once(self, attention, listener_state, listener_len, speller_state, speller_last_output, batch_size):
        '''
        listener_state      : Time * Batch_size * length
        listener_len        : Batch_size
        speller_state       : [h_(t-1), c_(t-1)]
        speller_last_output : Batch_size * rnn_hidden_size = h_(t-1)
        '''
        
        speller_hidden_state, speller_cell_state = speller_state[0], speller_state[1]
        context, att_score = attention(listener_state, listener_len, speller_hidden_state[-1], batch_size)
        rnn_input = torch.cat([speller_last_output, context], dim=1)
        new_speller_hidden_state, new_speller_cell_state = [0,0,0], [0,0,0]
        # print("In run_once: ", rnn_input.type(), speller_hidden_state[0].type(), speller_cell_state[0].type())
        #----------------------------------BEGIN----------------------------------------------------------
        new_speller_hidden_state[0], new_speller_cell_state[0] = self.arnn(
            rnn_input, 
            (speller_hidden_state[0], speller_cell_state[0]))
        #----------------------------------LSTM 1----------------------------------------------------------
        new_speller_hidden_state[1], new_speller_cell_state[1] = self.rnn1(
            new_speller_hidden_state[0], 
            (speller_hidden_state[1], speller_cell_state[1]))
        #----------------------------------LSTM 2----------------------------------------------------------
        new_speller_hidden_state[2], new_speller_cell_state[2] = self.rnn2(
            new_speller_hidden_state[1], 
            (speller_hidden_state[2], speller_cell_state[2]))

        speller_output = new_speller_hidden_state[-1]
        speller_output = torch.cat([speller_output, context], dim=1)
        speller_output = self.a(self.outfc(speller_output))
        predict_output = self.out(speller_output)
        return speller_output, predict_output, (new_speller_hidden_state, new_speller_cell_state), att_score

    def get_initial(self, batch_size = 32, listener_last_state = None):
        speller_hidden_state, speller_cell_state = [], []
        start = 0
        if type(listener_last_state) != type(None):
            tmp = (listener_last_state[0][0] + listener_last_state[0][1])/2
            speller_hidden_state.append(tmp)
            tmp = (listener_last_state[1][0] + listener_last_state[1][1])/2
            speller_cell_state.append(tmp)
            start = 1
        for _ in range(start, self.layers):
            speller_hidden_state.append(Variable(torch.randn(batch_size, self.rnn_hidden_size).cuda()))
            speller_cell_state.append(Variable(torch.randn(batch_size, self.rnn_hidden_size).cuda()))
        # for i in range(self.layers):
        #     speller_hidden_state[i]
        #     speller_cell_state[i].cuda()
        return (speller_hidden_state, speller_cell_state)
