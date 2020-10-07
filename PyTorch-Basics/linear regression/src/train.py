import dataset
import model

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.utils.data import TensorDataset,  DataLoader
from torch.optim import SGD

def fit(epochs, model, loss_function, optimizer, train_dl):
    for epoch in range(epochs):
        for xb, yb in train_dl:
            prediction = model(xb)
            loss = loss_function(prediction, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        
        if (epoch+1)%10 == 0:
            print("epoch [{}/{}], loss {:.4f}".format(epoch+1, epochs, loss.item()))

fit(100, model.model, model.loss_function, model.optimizer, dataset.train_dl)
