import torch
import numpy as np
import trainer
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


# OPTIONS
EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-4
REG = 1e-5
CHECKPOINT_INTERVAL = 5


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
print('Final test accuracy: {0:.3f}%'.format(test_acc*100))

# TODO: generate few test samples (show image, true mask, predicted mask together)
