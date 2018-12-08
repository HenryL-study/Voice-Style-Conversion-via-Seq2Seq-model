import numpy as np
import torch.nn as nn
import torch
from torch.nn import Sequential
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence
from torch.autograd import Variable

class Attention(nn.Module):
    def __init__(self, attention_size=128, listener_size=256, speller_size=256):
        super().__init__()
        self.score = nn.Linear(listener_size, attention_size)
        self.value = nn.Linear(listener_size, attention_size)
        self.project = nn.Linear(speller_size, attention_size)
        self.a = nn.ReLU(inplace=True)

    def forward(self, listener_state, listener_len, speller_state, batch_size, batch_first = False):
        '''
        listener_state  : Time * Batch_size * length
        listener_len    : Batch_size
        speller_state   : State_size 
        '''
        # print("In attention: ", listener_state.size())
        if not batch_first:
            listener_state = listener_state.permute(1,0,2).contiguous()
        # print("In attention: ", listener_state.size(), self.score.weight.size())
        score = self.a(self.score(listener_state)).permute(0,2,1) # B * attention_size * L
        context = self.a(self.value(listener_state)) # B * L * attention_size
        decode_project = self.a(self.project(speller_state)) # B * attention_size
        # decode_project = self.a(self.project(speller_state)) # B * attention_size
        
        # Calculate softmax
        score = torch.bmm(decode_project.unsqueeze(1), score).squeeze(1) # B * L
        mask = torch.zeros(score.size()).cuda()
        for i,l in enumerate(listener_len):
            mask[i][l:] = -100
        score += mask
        score = F.softmax(score, dim = 1)
        context = torch.bmm(score.unsqueeze(1), context).squeeze(1) # B * attention_size

        return context, score

class FakeAttention(nn.Module):
    def __init__(self, attention_size=128, listener_size=256, speller_size=256):
        super().__init__()
        self.attention_size =attention_size
        # self.batch_size = batch_size

    def forward(self, listener_state, listener_len, speller_state, batch_size, batch_first = False):
        '''
        listener_state  : Time * Batch_size * length
        listener_len    : Batch_size
        speller_state   : State_size 
        '''
        context = torch.randn((batch_size, self.attention_size)).cuda()
        score = torch.randn((batch_size, 10)).cuda()

        return context, score