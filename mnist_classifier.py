'''
    Machine Learning Block Implementation Practice
    with Pytorch Lightning

    Author : Youngung Han (2023)
'''

# most of the case, you just change the component loading part
# all other parts are almost same

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

import numpy as np
from argparse import ArgumentParser

class MNISTDataset(Dataset):
    ''' 
        A single example : python object -> Tensor (convertor)
        return an example for batch construction. 
    '''
    def __init__(self, data):
        self.image_data = np.load(data['image'])
        self.label_data = np.load(data['label'])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image, label = self.image_data[index], self.label_data[index]
        def normalize_image(image):
            ''' Preprocessing '''
            # 2D -> 1D
            image = image.reshape(784).astype('float32') # -1
            # 0 ~ 255 -> 0 ~ 1.0
            image /= 255
        normalize_image(image)
        sample = [image, label]
        return sample

    
class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, 
                 batch_size: int = 32):
        super().__init__()
        self.batch_size = batch_size  
        self.train_dataset, self.valid_dataset, self.test_dataset = None, None, None
        
        ## NOTE
        ## --- Pytorch lightning provides '*.check prepare(), *.setup()'
        
        def set_dataset():   
            ''' numpy object to custom DATASET '''
            dataset = {
                'train' : {
                    'image': f'./mnist/data/train.image.npy',
                    'label': f'./mnist/data/train.label.npy'
                },
                'test' : {
                    'image': f'./mnist/data/test.image.npy',
                    'label': f'./mnist/data/test.label.npy'
                },
            }   
            self.train_dataset, self.test_dataset = MNISTDataset(dataset['train']), MNISTDataset(dataset['test'])
            N = len(self.train_dataset)
            self.train_dataset, self.valid_dataset = random_split(self.train_dataset, [int(N*0.8), N-int(N*0.2)])
        set_dataset()
        
        def train_dataloader(self):
            return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        def val_dataloader(self):
            return DataLoader(self.valid_dataset, batch_size=self.batch_size)
        def test_dataloader(self):  
            return DataLoader(self.test_dataset, batch_size=self.batch_size)
            
        
    
    
def main():
    pl.seed_everything(1234)
    
    # ----------------------------------------
    # args
    # ----------------------------------------
    parser = ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=200)
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MLP_MNIST_Classifier.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # ----------------------------------------
    # data
    # ----------------------------------------
    model = MLP_MNIST_Classifier(args)
    
    
    
    
if __name__ == '__main__':
    main()