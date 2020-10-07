import dataset

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.utils.data import TensorDataset,  DataLoader
from torch.optim import SGD


model = nn.Linear(3, 2)
loss_function = F.mse_loss
optimizer = SGD(model.parameters(), lr=1e-5)
