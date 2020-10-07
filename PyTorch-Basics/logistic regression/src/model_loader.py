import dataset
import utils
import models
import evals
import train
import test

from torch.utils.data import random_split, DataLoader

'''load model & test''' 
model2 = models.MNISTModel()
model2.load_state_dict(torch.load('mnist-logistic.pth'))
model2.state_dict()

test_loader = DataLoader(dataset.test_dataset, batch_size=256)
result = evals.evaluate(model2, test_loader)