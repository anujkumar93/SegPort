import torch
import numpy as np
from train import train
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# OPTIONS
EPOCHS = 20
BATCH_SIZE = 3
LR = 1e-4
REG = 1e-5


model, losses = train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG)
torch.save(model.state_dict(), 'model.pth')  # only saves parameters
# load it back using torch.load('model.pth')
# use model.eval() at 'test' time to make sure dropout is in 'test' mode

# LOSS CURVE
plt.plot(np.arange(losses.shape[0]) + 1, losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig('loss_curve.png')
