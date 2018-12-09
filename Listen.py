import numpy as np
import torch.nn as nn
import torch
from torch.nn import Sequential
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pack_sequence
from torch.autograd import Variable
from LockedDropout import LockedDropout

class Listener(nn.Module):
    def __init__(self, rnn_hidden_size = 256, dim = 200, useLockDrop = False, class_size = 64969):
        super().__init__()
                
        # self.lockdrop1 = LockedDropout(0.2)
        self.lockdrop2 = LockedDropout(0.2)
        self.lockdrop3 = LockedDropout(0.3)
        self.useLockDrop = useLockDrop
        self.bn = nn.BatchNorm1d(dim)
        self.embed = nn.Embedding(class_size, dim)

        self.LSTM1 = nn.LSTM(input_size=dim, hidden_size=rnn_hidden_size, num_layers=1, bidirectional=True)
        # self.mid_embed1 = getMidcnn()
        self.LSTM2 = nn.LSTM(input_size=rnn_hidden_size * 4, hidden_size=rnn_hidden_size, num_layers=1, bidirectional=True)
        # self.mid_embed2 = getMidcnn()
        self.LSTM3 = nn.LSTM(input_size=rnn_hidden_size * 4, hidden_size=rnn_hidden_size, num_layers=1, bidirectional=True)
        # self.mid_embed3 = getMidcnn()
        # self.LSTM4 = nn.LSTM(input_size=rnn_hidden_size * 4, hidden_size=rnn_hidden_size, num_layers=1, bidirectional=True)
    
    def forward(self, features):
        '''
        features : B*L
        '''
        features = features.unsqueeze_(2)  # B*L*1
        features = self.embed(features)    # B*L*200
        # batch_size = len(features)
        lens = [len(s) for s in features] # lens of all lines (already sorted)

        #----------------------------------BEGIN----------------------------------------------------------
        packed_input = pack_sequence(features)
        padded_input, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_input)
        padded_input = padded_input.permute(1, 2, 0).contiguous()
        norm_input = self.bn(padded_input)
        norm_input = norm_input.permute(2, 0, 1).contiguous()
        packed_input = pack_padded_sequence(norm_input, lens)
        #----------------------------------LSTM 1----------------------------------------------------------
        output_packed,_ = self.LSTM1(packed_input)   #  (seq_len, batch, num_directions * hidden_size)
        output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed) # unpacked output (padded)
        #----------------------------------LSTM 2----------------------------------------------------------
        seq_padded = output_padded.permute(1, 0, 2).contiguous()
        n, l, d = seq_padded.size()
        l = l - l%2
        seq_cropped = seq_padded[:,:l,:]
        seq_cropped = seq_cropped.view(n, l//2, d*2)
        if self.useLockDrop:
            seq_cropped = self.lockdrop2(seq_cropped)
        seq_cropped = seq_cropped.permute(1, 0, 2).contiguous()
        lens = [l//2 for l in lens]
        packed_input = pack_padded_sequence(seq_cropped, lens)
        output_packed,_ = self.LSTM2(packed_input)
        output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed) # unpacked output (padded)
        #----------------------------------MID 2----------------------------------------------------------
        seq_padded = output_padded.permute(1, 0, 2).contiguous()
        n, l, d = seq_padded.size()
        l = l - l%2
        seq_cropped = seq_padded[:,:l,:]
        seq_cropped = seq_cropped.view(n, l//2, d*2)
        if self.useLockDrop:
            seq_cropped = self.lockdrop2(seq_cropped)
        seq_cropped = seq_cropped.permute(1, 0, 2).contiguous()
        lens = [l//2 for l in lens]
        packed_input = pack_padded_sequence(seq_cropped, lens)
        output_packed, last_state = self.LSTM3(packed_input)
        output_padded, _ = torch.nn.utils.rnn.pad_packed_sequence(output_packed) # unpacked output (padded)
        #----------------------------------BACK EMBED----------------------------------------------------------
        output_padded = output_padded.permute(1, 0, 2).contiguous()
        output_padded = self.lockdrop3(output_padded)
        output_padded = output_padded.permute(1, 0, 2).contiguous()
        #----------------------------------END------------------------------------------------------------
        if False:
            print("output_padded size: ", output_padded.size())

        return output_padded, lens, last_state
