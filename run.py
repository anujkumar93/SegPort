from train import train


# OPTIONS
EPOCHS = 5
BATCH_SIZE = 1
LR = 1e-4
REG = 1e-4


model = train(epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, reg=REG)
