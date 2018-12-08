import numpy as np
import torch
import matplotlib.pyplot as plt

att = np.load("att1000.npy")

f = plt.figure(figsize=(8, 8.5))
ax = f.add_subplot(1, 1, 1)
i = ax.imshow(att, interpolation='nearest', cmap='gray')

print(att)