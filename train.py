import torch
import torch.nn as nn
import numpy as np
from FCN import FCN
from DataLoader import DataLoader


# DEFAULT OPTIONS
EPOCHS = 20
BATCH_SIZE = 1
LR = 1e-4
REG = 1e-5


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG):
    model = FCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    loss_function = nn.BCEWithLogitsLoss()

    dlo = DataLoader(batch_size)
    training_set_size = dlo.get_training_set_size()
    iters_per_epoch = int(np.ceil(training_set_size / batch_size))
    losses = np.zeros([epochs * iters_per_epoch,])

    for e in range(epochs):
        model.train()  # to make sure dropout is in 'train' mode; use model.eval at 'test' time
        print('Epoch:', e)
        dlo.shuffle()
        for i in range(iters_per_epoch):
            model.zero_grad()
            x, y = dlo.generate_batch()
            x = torch.autograd.Variable(torch.FloatTensor(x))
            y = torch.autograd.Variable(torch.FloatTensor(y))
            output = model(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            print(loss.data[0])
            losses[i + e * iters_per_epoch] = loss.data[0]
            del x, y, output, loss

    return model, losses
