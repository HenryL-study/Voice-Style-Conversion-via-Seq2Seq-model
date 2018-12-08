import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

TRAIN = 0
DEV = 1
TEST = 2
names = ["train.npy", "dev.npy", "test.npy"]
label_names = ["idx_train_transcripts.npy", "idx_dev_transcripts.npy"]

def collate_lines(data):
    inputs,targets = [d[0] for d in data], [d[1] for d in data]
    lens = [seq.shape[0] for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets = pad_sequence([targets[i] for i in seq_order], batch_first=True)
    return inputs,targets

def collate_lines_test(data):
    inputs,targets = [d[0] for d in data], [d[1] for d in data]
    lens = [seq.shape[0] for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    return inputs,targets

class UttDataset(Dataset):
    def __init__(self, data_path = 'data', data_type = TRAIN):
        
        data = np.load(data_path + "/" + names[data_type], encoding='bytes')
        if data_type != TEST:
            label = np.load(data_path + "/" + label_names[data_type], encoding='bytes')
            end_sys = np.array([0])
            self.labels = [torch.tensor(np.append(l, end_sys), dtype=torch.long) for l in label]
        else:
            self.labels = None
        self.lines=[torch.tensor(l, dtype=torch.float) for l in data]
        
        print("Loaded ", names[data_type])
        print("Total utterances: ", len(self.lines))
        # print("Total trascripts: ", len(self.labels))
        

    def __getitem__(self,i):
        utt = self.lines[i].to('cuda')
        # utt = torch.transpose(utt, 0, 1)
        if self.labels != None:
            label = self.labels[i].to('cuda')
        else:
            label = None
        return utt, label

    def __len__(self):
        return len(self.lines)

def getDataloader(batch_size = 64, dev2train = True):
    if dev2train:
        train_dataset = UttDataset('data', DEV)
    else:
        train_dataset = UttDataset('data', TRAIN)
    dev_dataset = UttDataset('data', DEV)
    test_dataset = UttDataset('data', TEST)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn = collate_lines)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=16, collate_fn = collate_lines)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn = collate_lines_test)

    return train_loader, dev_loader, test_loader


# a,b,c = getDataloader()

# for batch_num, (inputs, targets) in enumerate(b):
#     print("batch_num: ", batch_num)
#     print("inputs lenth: ", len(inputs))
#     print("targets lenth: ", len(targets))
#     break