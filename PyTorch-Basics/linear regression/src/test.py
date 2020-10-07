import dataset
import model
import train

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from torch.utils.data import TensorDataset,  DataLoader
from torch.optim import SGD


handmade_validation = model.model(dataset.inputs)
print(handmade_validation)
print(dataset.targets)

test_prediction = model.model(dataset.test_data)
print(dataset.test_prediction)