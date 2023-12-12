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
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
