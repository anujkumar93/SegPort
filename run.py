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


# OPTIONS
EPOCHS = 1
BATCH_SIZE = 1
LR = 1e-4
REG = 1e-5
CHECKPOINT_INTERVAL = 5
OUTPUT_TEST_IMAGES = 3


timestamp = '{:%Y%m%d_%H%M%S}'.format(datetime.datetime.now())  # for save_dir
if not os.path.exists('output'):
    os.makedirs('output')
if not os.path.exists('output/' + timestamp):
    os.makedirs('output/' + timestamp)
save_dir = 'output/' + timestamp

model, losses, dlo = trainer.train(save_dir, epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG, checkpoint_interval=CHECKPOINT_INTERVAL)
torch.save(model.state_dict(), save_dir+'/final_model.pth')  # only saves parameters
# load it back using torch.load('model.pth')
# use model.eval() at 'test' time to make sure components are in 'test' mode

# LOSS CURVE
plt.plot(np.arange(losses.shape[0]) + 1, losses)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.tight_layout()
plt.savefig(save_dir+'/final_loss_curve.png')
plt.close()

test_preds, test_acc = trainer.test(model, dlo)
print('Final test accuracy: {0:.3f}%'.format(test_acc * 100))

# plotting sample test images and their masks
test_data, test_set_size = dlo.test_data, dlo.test_set_size
sampled_indxs = np.random.choice(range(test_set_size), OUTPUT_TEST_IMAGES, replace=False)
for i in range(OUTPUT_TEST_IMAGES):
    plt.subplot(131)
    fig = plt.imshow(mpimg.imread(dlo.data_folder + str(test_data[sampled_indxs[i]])))
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.subplot(132)
    fig = plt.imshow(loadmat(dlo.label_folder + str(test_data[sampled_indxs[i]][:-4]) + '_mask')['mask'])
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.subplot(133)
    fig = plt.imshow(test_preds[sampled_indxs[i], :, :])
    plt.axis('off')
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.savefig(str(test_data[sampled_indxs[i]]), bbox_inches='tight', pad_inches=0)
    plt.close()
