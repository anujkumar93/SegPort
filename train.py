import torch
import torch.nn as nn
from FCN import FCN


# OPTIONS
epochs = 5
batch_size = 10
lr = 1e-4
reg = 1e-4


# TRAINING...

model = FCN(2)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
loss_function = nn.BCEWithLogitsLoss()

dlo = DataLoader(batch_size)
training_set_size = dlo.get_training_set_size()

for e in range(epochs):
    dlo.shuffle_training_set()
    i = 0
    while i < training_set_size:
        model.zero_grad()
        i += batch_size
        x, y = dlo.get_next_batch()
        x_var = torch.autograd.Variable(torch.FloatTensor(x))
        y_var = torch.autograd.Variable(torch.FloatTensor(y))
        output = model(x_var)
        loss = loss_function(output, y_var)
        loss.backward()
        optimizer.step()
        print(loss.data[0])