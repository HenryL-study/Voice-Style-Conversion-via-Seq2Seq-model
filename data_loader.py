import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
import torch
from torch.nn.utils.rnn import pad_sequence

TRAIN = 0
DEV = 1
TEST = 2
names = ["train_data_1.npy", "valid_data_1.npy", "test_data_1.npy"]
label_names = ["train_data_2.npy", "valid_data_2.npy"]

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
    def __init__(self, data_path = 'data/generate/', data_type = TRAIN):
        end_sys = np.array([[0 for _ in range(40)]])
        data = np.load(data_path + "/" + names[data_type], encoding='bytes')
        if data_type != TEST:
            label = np.load(data_path + "/" + label_names[data_type], encoding='bytes')
            self.labels = [torch.tensor(np.concatenate((l, end_sys), axis=0), dtype=torch.float) for l in label]
        else:
            self.labels = None
        self.lines=[torch.tensor(np.concatenate((l, end_sys), axis=0), dtype=torch.float) for l in data]
        
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
        train_dataset = UttDataset('data/generate/', DEV)
    else:
        train_dataset = UttDataset('data/generate/', TRAIN)
    dev_dataset = UttDataset('data/generate/', DEV)
    test_dataset = UttDataset('data/generate/', TEST)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, collate_fn = collate_lines)
    dev_loader = DataLoader(dev_dataset, shuffle=False, batch_size=4, collate_fn = collate_lines)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn = collate_lines_test)

    return train_loader, dev_loader, test_loader


# a,b,c = getDataloader(dev2train = False)

# for batch_num, (inputs, targets) in enumerate(b):
#     print("batch_num: ", batch_num)
#     print("inputs lenth: ", len(inputs))
#     print("targets lenth: ", len(targets))
#     break
