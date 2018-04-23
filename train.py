import torch
import torch.nn as nn
from FCN import FCN
from DataLoader import DataLoader


# DEFAULT OPTIONS
EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-4
REG = 1e-4


def train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG):
    model = FCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    loss_function = nn.BCEWithLogitsLoss()

    dlo = DataLoader(batch_size)
    training_set_size = dlo.get_training_set_size()

    for e in range(epochs):
        print('Epoch:', e)
        dlo.shuffle()
        i = 0
        while i < training_set_size:
            model.zero_grad()
            i += batch_size
            x, y = dlo.generate_batch()
            x_var = torch.autograd.Variable(torch.FloatTensor(x))
            y_var = torch.autograd.Variable(torch.FloatTensor(y))
            output = model(x_var)
            loss = loss_function(output, y_var)
            loss.backward()
            optimizer.step()
            print(loss.data[0])

    return model
