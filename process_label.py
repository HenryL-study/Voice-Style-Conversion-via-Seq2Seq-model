import numpy as np
from cIdx import chr2idx

label_names = ["train_transcripts.npy", "dev_transcripts.npy"]
data_path = 'data'

for labeln in label_names:
    new_label = []
    label = np.load(data_path + "/" + labeln)
    for line in label:
        tmp = []
        for w in line:
            for c in w:
                c = chr(c)
                tmp.append(chr2idx[c])
            tmp.append(chr2idx[' '])
        new_label.append(np.array(tmp))
    
    new_label = np.array(new_label)
    np.save(data_path + "/idx_" + labeln, new_label)

