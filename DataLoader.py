import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

class DataLoader:
    def __init__(self,batch_size,test_split=0.1):
        self.data_folder="../data/images_data_crop/"
        self.label_folder="../data/images_mask_crop/"
        self.file_list=os.listdir(self.data_folder)
        self.train_split=1-test_split
        self.total_samples=len(self.file_list)
        self.train_data=self.file_list[:int(self.train_split*self.total_samples)]
        self.test_data=self.file_list[int(self.train_split*self.total_samples):]
        self.batch_counter=0
        self.batch_size=batch_size
        assert self.batch_size==int(self.batch_size)

    def shuffle(self):
        self.train_data=np.random.permutation(self.train_data)

    def generateBatch(self):
        X,y=[],[]
        for i in range(self.batch_counter*self.batch_size,min(int(self.train_split*self.total_samples), \
                       (self.batch_counter+1) * self.batch_size)):
            X.append(plt.imread(self.data_folder+str(self.train_data[i])))
            y.append(loadmat(self.label_folder+str(self.train_data[i][:-4])+'_mask')['mask'])
        self.batch_counter+=1
        return X,y

# obj=DataLoader(25)
# obj.shuffle()
# X,y=obj.generateBatch()
# X=np.array(X)
# y=np.array(y)
# print(X.shape, y.shape)