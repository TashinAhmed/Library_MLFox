import torch
import torchvision

import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import random_split, DataLoader


dataset = MNIST(root='/data', download=True, transform=transforms.ToTensor())
test_dataset = MNIST(root='/data', download=False, transform=transforms.ToTensor())
train_ds, val_ds = random_split(dataset, [50000, 10000])

batch_size = 128

train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)
test_loader = DataLoader(test_dataset, batch_size=256)

input_size = 28*28
num_classes = 10
