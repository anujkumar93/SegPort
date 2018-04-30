import torch
import torch.nn as nn
import numpy as np
import sys
from FCN import FCN
from DataLoader import DataLoader
import time
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# DEFAULT OPTIONS
EPOCHS = 20
BATCH_SIZE = 1
LR = 1e-4
REG = 1e-5


def train(save_dir, model=None, optimizer=None,
          epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG,
          checkpoint_interval=5, use_6_channels=True, debug=False):
    if model is None:
        model = FCN(use_6_channels=use_6_channels)
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=reg)
    loss_function = nn.BCEWithLogitsLoss()

    dlo = DataLoader(batch_size, use_6_channels=use_6_channels, debug=debug)
    training_set_size = dlo.get_training_set_size()
    iters_per_epoch = int(np.ceil(training_set_size / batch_size))
    losses = np.zeros([epochs * iters_per_epoch,])

    start_time = time.time()
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
            losses[i + e * iters_per_epoch] = loss.item()
            print('{:3}%   Time: {:21}  Epoch: {:4}  Iter: {:4}  Loss: {}'.format(
                  int((i+1 + iters_per_epoch * e) / (iters_per_epoch * epochs) * 100),
                  time_since(start_time, (i + 1 + iters_per_epoch * e) / (iters_per_epoch * epochs)),
                  str(e + 1), str(i + 1), loss.item()))
            sys.stdout.flush()
            del x, y, output, loss

        # after every checkpoint_interval epochs: save checkpoint model, save loss curve, display test error
        if (e + 1) % checkpoint_interval == 0:
            torch.save(model.state_dict(), save_dir+'/model_after_epoch_'+str(e+1)+'.pth')
            torch.save(optimizer.state_dict(), save_dir+'/optimizer_after_epoch_'+str(e+1)+'.pth')
            np.save(save_dir+'/losses_after_epoch_'+str(e+1), losses)

            # save loss curve so far
            plt.plot(np.arange(losses.shape[0]) + 1, losses)
            plt.xlabel('Iterations')
            plt.ylabel('Loss')
            plt.tight_layout()
            plt.savefig(save_dir+'/loss_curve_after_epoch_'+str(e+1)+'.png')
            plt.close()

            # display test error
            _, test_acc, test_iou = test(model, dlo)
            print('Test accuracies:')
            print('Per-pixel classification: {0:.3f}%'.format(test_acc * 100))
            print('Intersection-over-Union:  {0:.3f}%'.format(test_iou * 100))

    return model, optimizer, losses, dlo


def test(model, dlo):
    softmax = nn.Softmax2d()
    model.eval()  # to make sure components are in 'test' mode; use model.train() at 'train' time
    test_set_size = dlo.get_test_set_size()
    dlo.reset_test_batch_counter()
    per_pixel_accuracy = 0
    iou_accuracy = 0
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
        preds = preds.data.double()
        labels = torch.max(y, 1)[1].data.double()
        per_pixel_accuracy += len(y) * (preds == labels).double().mean()
        # NOTE: FOLLOWING CODE FOR INTERSECTION-OVER-UNION IS VALID ONLY FOR BINARY CLASSIFICATION
        intersection = preds * labels
        union = ((preds + labels) > 0.5).double()
        iou_accuracy += len(y) * torch.sum(intersection)/torch.sum(union)

    return torch.cat(predictions).data.numpy(), per_pixel_accuracy/test_set_size, iou_accuracy/test_set_size


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (-%s)' % (as_minutes(s), as_minutes(rs))