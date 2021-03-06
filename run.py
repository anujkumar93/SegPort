import torch
import numpy as np
import trainer
import datetime
from scipy.io import loadmat
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from FCN import FCN
from DataLoader import DataLoader


# FOR CLOUD: DO NOT FORGET TO SET NUM THREADS
# torch.set_num_threads(16)

# OPTIONS
DEBUG = False
CONTINUE_TRAINING = False
TEST_ONLY = False
USE_6_CHANNELS = True
USE_CROSS_ENTROPY_LOSS = True
MODEL_PATH = 'model.pth'  # if continuing training or only testing
OPTIMIZER_PATH = 'optimizer.pth'  # if continuing training or only testing
EPOCHS = 2
BATCH_SIZE = 2
LR = 1e-4
REG = 1e-6
CHECKPOINT_INTERVAL = 5  # number of epochs between checkpoints (save model and loss curve)
NUM_TEST_SAMPLES = 30  # for generating test samples at the end


# GENERATE SAVE DIRECTORY PATH
timestamp = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())  # for save_dir
if not os.path.exists('output'):
    os.makedirs('output')
save_dir = 'output/' + timestamp + '_' + str(EPOCHS) + 'epochs_' + str(REG) + 'reg'
if DEBUG:
    save_dir += '_debug'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


if TEST_ONLY:
    model = FCN(use_6_channels=USE_6_CHANNELS)
    model.load_state_dict(torch.load(MODEL_PATH))
    dlo = DataLoader(BATCH_SIZE, use_6_channels=USE_6_CHANNELS, debug=DEBUG)
else:
    # TRAIN AND SAVE MODEL AND OPTIMIZER
    if CONTINUE_TRAINING:
        model = FCN(use_6_channels=USE_6_CHANNELS)
        model.load_state_dict(torch.load(MODEL_PATH))

        if os.path.exists(OPTIMIZER_PATH):
            optimizer = torch.optim.Adam(model.parameters())
            optimizer.load_state_dict(torch.load(OPTIMIZER_PATH))
        else:
            optimizer = None

        model, optimizer, losses, dlo = trainer.train(save_dir, model=model, optimizer=optimizer,
                                                      epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG,
                                                      checkpoint_interval=CHECKPOINT_INTERVAL,
                                                      use_6_channels=USE_6_CHANNELS, debug=DEBUG,
                                                      use_cross_entropy_loss=USE_CROSS_ENTROPY_LOSS)
    else:
        model, optimizer, losses, dlo = trainer.train(save_dir,
                                                      epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG,
                                                      checkpoint_interval=CHECKPOINT_INTERVAL,
                                                      use_6_channels=USE_6_CHANNELS, debug=DEBUG,
                                                      use_cross_entropy_loss=USE_CROSS_ENTROPY_LOSS)
    torch.save(model.state_dict(), save_dir+'/final_model.pth')  # only saves parameters
    torch.save(optimizer.state_dict(), save_dir+'/final_optimizer.pth')
    np.save(save_dir+'/final_losses', losses)


    # LOSS CURVE
    plt.plot(np.arange(losses.shape[0]) + 1, losses)
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.ylim(ymin=0)
    plt.tight_layout()
    plt.savefig(save_dir+'/final_loss_curve.png')
    plt.close()


# TEST THE MODEL
test_preds, test_acc, test_iou = trainer.test(model, dlo)
print('Final test accuracies:')
print('Per-pixel classification: {0:.3f}%'.format(test_acc * 100))
print('Intersection-over-Union:  {0:.3f}%'.format(test_iou * 100))


# GENERATE TEST SAMPLES AND PREDICTIONS
test_data, test_set_size = dlo.get_test_data_file_names(), dlo.get_test_set_size()
if NUM_TEST_SAMPLES > test_set_size:
    NUM_TEST_SAMPLES = test_set_size
sampled_indices = np.random.choice(range(test_set_size), NUM_TEST_SAMPLES, replace=False)
for i in range(NUM_TEST_SAMPLES):
    plt.subplot(131)
    fig = plt.imshow(mpimg.imread(dlo.images_folder + str(test_data[sampled_indices[i]])[:-4] + '.jpg'))
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title('Input')
    plt.tight_layout()
    plt.subplot(132)
    fig = plt.imshow(loadmat(dlo.label_folder + str(test_data[sampled_indices[i]][:-4]) + '_mask')['mask'], cmap='gray')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title('Annotation')
    plt.tight_layout()
    plt.subplot(133)
    fig = plt.imshow(test_preds[sampled_indices[i], :, :], cmap='gray')
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.title('Prediction')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    plt.savefig(save_dir + '/' + str(test_data[sampled_indices[i]])[:-4] + '.png', bbox_inches='tight', pad_inches=0.1)
    plt.close()
