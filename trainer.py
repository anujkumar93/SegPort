import torch
import torch.nn as nn
import numpy as np
import sys
from FCN import FCN
from DataLoader import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# DEFAULT OPTIONS
EPOCHS = 20
BATCH_SIZE = 1
LR = 1e-4
REG = 1e-5


def train(save_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG, checkpoint_interval=5):
    model = FCN()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    loss_function = nn.BCEWithLogitsLoss()

    dlo = DataLoader(batch_size)
    training_set_size = dlo.get_training_set_size()
    iters_per_epoch = int(np.ceil(training_set_size / batch_size))
    losses = np.zeros([epochs * iters_per_epoch,])

    for e in range(epochs):
        model.train()  # to make sure components are in 'train' mode; use model.eval() at 'test' time
        dlo.shuffle_training_set()
        for i in range(iters_per_epoch):
            model.zero_grad()
            x, y = dlo.get_next_training_batch()
            x = torch.autograd.Variable(torch.FloatTensor(x))
            y = torch.autograd.Variable(torch.FloatTensor(y))
            output = model(x)
            loss = loss_function(output, y)
            loss.backward()
            optimizer.step()
            losses[i + e * iters_per_epoch] = loss.data[0]
            print('Epoch: {:03}    Iter: {:04}    Loss: {}'.format(e, i, loss.data[0]))
            sys.stdout.flush()
            del x, y, output, loss

        # after every checkpoint_interval epochs: save checkpoint model, save loss curve, display test error
        if (e + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), save_dir+'/model_after_epoch_'+str(e)+'.pth')

            # save loss curve so far
            plt.plot(np.arange(losses.shape[0]) + 1, losses)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.tight_layout()
            plt.savefig(save_dir+'/loss_curve_after_epoch_'+str(e)+'.png')
            plt.close()

            # display test error
            _, test_acc = test(model, dlo)
            print('Test accuracy: {0:.3f}%'.format(test_acc*100))

    return model, losses, dlo


def test(model, dlo):
    softmax = nn.Softmax2d()
    model.eval()  # to make sure components are in 'test' mode; use model.train() at 'train' time
    test_set_size = dlo.get_test_set_size()
    dlo.reset_test_batch_counter()
    accuracy = 0
    predictions = []
    for i in range(int(np.ceil(test_set_size / dlo.get_batch_size()))):
        model.zero_grad()
        x, y = dlo.get_next_test_batch()
        x = torch.autograd.Variable(torch.FloatTensor(x))
        y = torch.autograd.Variable(torch.FloatTensor(y))
        output = model(x)
        output = softmax(output)
        _, preds = torch.max(output, 1)
        predictions.append(preds)
        accuracy += len(y) * (preds.data.double() == torch.max(y, 1)[1].data.double()).double().mean()

    return torch.cat(predictions).data.numpy(), accuracy/test_set_size
