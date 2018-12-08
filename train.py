import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from data_loader import getDataloader
from cIdx import chr2idx, idx2chr
from torch.nn.utils.rnn import pad_sequence
from Listen import Listener
from Attention import Attention, FakeAttention
from Spell import Speller

BATCH_SIZE = 16
EPOCH = 2
MODEL_PATH = "./models"
LOAD_MODEL = [False, False, False]
SAVE_MODEL = [True, True, True]
DEV2TRAIN = False
PRE_TRAIN = True
PRE_EPOCH = 10
LOAD_PRE_TRAIN = False
SAVE_PRE_TRAIN = True
label_map = idx2chr
listener_size = 256
speller_size = 256
attention_size = 128

class ER:
    def __init__(self):
        self.ce = torch.nn.CrossEntropyLoss(reduction = 'elementwise_mean')

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

    def forward(self, prediction, target):
        # bs = target.size()[0]
        label_lens = [len(s) for s in target]
        logits = [p[:label_lens[i],:] for i,p in enumerate(prediction)]
        logits = torch.cat(logits, dim=0)
        target = target.view(-1)
        output = self.ce(logits, target)
        output = torch.exp(output)

        return output

def weight_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
    if isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    # if isinstance(m, torch.nn.LSTM):
    #     torch.nn.init.xavier_uniform_(m.weight.data)

train_loader, dev_loader, test_loader = getDataloader(batch_size=BATCH_SIZE, dev2train = DEV2TRAIN)
print("-----------------Loading Finished-------------------------")
encoder = Listener(
    rnn_hidden_size = listener_size, 
    dim             = 40, 
    useLockDrop     = True
)
attention = Attention(
    attention_size  = attention_size, 
    listener_size   = listener_size * 2, 
    speller_size    = speller_size
)
decoder = Speller(
    context_size    = attention_size, 
    rnn_hidden_size = speller_size, 
    class_size      = 33, 
    useLockDrop     = False
)

LAS = [encoder, attention, decoder]
for m in LAS:
    m.apply(weight_init)
    m.cuda()
if any(LOAD_MODEL):
    print("Loading model...")
    for i,m in enumerate(LAS):
        if LOAD_MODEL[i]:
            m.load_state_dict(torch.load(MODEL_PATH + '/params-' + str(i) + '-30.pkl'))

if PRE_TRAIN:
    print("Start pre-training...")
    # test_l_loss = preTestLloss()
    if LOAD_PRE_TRAIN:
        print("Loading pre model...")
        decoder.load_state_dict(torch.load(MODEL_PATH + '/params-pre-10.pkl'))
    fake_att = FakeAttention()
    optimizer = torch.optim.Adam(decoder.parameters(), lr = 0.00005, weight_decay=1.2e-6)
    for i in range(PRE_EPOCH):
        total_loss = torch.tensor([0.0])
        num_idx = 0
        loss_func = ER()
        for batch_num, (inputs, targets) in enumerate(train_loader):
            # print("batch_num: ", batch_num)
            # print("inputs lenth: ", len(inputs))
            # print("targets lenth: ", len(targets))
            optimizer.zero_grad()

            # output, predict, scores
            output, _, _ = decoder(
                listener_state = None, 
                listener_len = None, 
                listener_last_state = None,
                max_iters = None, 
                attention = fake_att, 
                batch_size = targets.size()[0], 
                trancripts= targets, 
                has_trans = True,
                greedy_sample = True, 
                teacher_force_rate=0.8, 
                blank_symbol = 0
            )

            loss = loss_func(output, targets)
            loss.backward()
            # torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
            optimizer.step()

            total_loss += loss.cpu()
            num_idx += len(inputs)
            if batch_num%5 == 0:
                print(batch_num, "th loss: ", total_loss.item()/num_idx)
            # if batch_num != 0 and batch_num%200 == 0:
            #     test_l_loss(i, batch_num, decoder)
        
        print(i, "th epoch loss: ", total_loss.item()/num_idx)
    if SAVE_PRE_TRAIN:
        print("Saving pre model...")
        torch.save(decoder.state_dict(), MODEL_PATH + '/params-pre-10.pkl')
# Training---------------------------------------------------------------------------------
parameters = [p for model in LAS for p in model.parameters()]
optimizer = torch.optim.Adam(parameters, lr = 0.00005, weight_decay=1.2e-6)
# optimizer = torch.optim.RMSprop(parameters, lr = 0.00005, weight_decay=1.2e-6)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, threshold=0.01, verbose=True)
loss_func = ER()
# test_l_loss = testLloss()
for i in range(EPOCH):
    total_loss = torch.tensor([0.0])
    num_idx = 0
    for batch_num, (inputs, targets) in enumerate(train_loader):
        # print("batch_num: ", batch_num)
        # print("inputs lenth: ", len(inputs))
        # print("targets lenth: ", len(targets))
        optimizer.zero_grad()
        
        output_padded, encode_lens, last_state = encoder(inputs)
        output, _, scores = decoder(
                listener_state = output_padded, 
                listener_len = encode_lens, 
                listener_last_state = last_state,
                max_iters = None, 
                attention = attention, 
                batch_size = targets.size()[0],
                trancripts= targets, 
                has_trans = True,
                greedy_sample = True, 
                teacher_force_rate=0.8, 
                blank_symbol = 0
            )

        loss = loss_func(output, targets)
        loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 0.01)
        optimizer.step()

        total_loss += loss.cpu()
        num_idx += len(inputs)
        # if batch_num % 200 == 0:
        #     fig1 = []
        #     for s in scores:
        #         fig1.append(s[0,:])
        #     fig1 = torch.stack(fig1, 0).cpu().detach().t().numpy()
        #     np.save("att" + str(batch_num) + ".npy", fig1)
        if batch_num%200 == 0:
            print(batch_num, "th loss: ", total_loss.item()/num_idx)
        # if batch_num != 0 and batch_num%200 == 0:
        #     del output, _, loss, scores
        #     test_l_loss(i, batch_num, encoder, attention, decoder)

    print(i, "th epoch loss: ", total_loss.item()/num_idx)
    # test_l_loss(i, batch_num, encoder, attention, decoder)

if any(SAVE_MODEL):
    print("Saving model...")
    for i,m in enumerate(LAS):
        if SAVE_MODEL[i]:
            torch.save(m.state_dict(), MODEL_PATH + '/params-' + str(i) + '-32.pkl')

for m in LAS:
    m.eval()
idx = 0
p_outputs = []
for batch_idx, (inputs, _) in enumerate(tqdm(test_loader)):
    output_padded, encode_lens, last_state = encoder(inputs)
    _, output, _ = decoder(
            listener_state = output_padded, 
            listener_len = encode_lens, 
            listener_last_state = last_state,
            max_iters = 200, 
            attention = attention, 
            batch_size = len(inputs),
            trancripts= None, 
            has_trans = False,
            greedy_sample = False, 
            teacher_force_rate=0.8, 
            blank_symbol = 0
        )
    
    output = output.detach().cpu().numpy()
    p_outputs.append(output)
    # for i in range(len(inputs)):
    #     chrs = "".join(label_map[o] for o in output[i])
    #     chrs = chrs.strip()
    #     predict.write(str(idx)+','+ chrs +'\n')
    #     idx += 1
    #if idx > 1:
    #    break
p_outputs = np.array(p_outputs)
np.save("predict_data", p_outputs)
print("Total predict: ", idx)
