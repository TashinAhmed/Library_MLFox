import dataset

import torch

def accuracy(outputs, labels):
    _, predicitions = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(predicitions == labels).item() / len(predicitions))

