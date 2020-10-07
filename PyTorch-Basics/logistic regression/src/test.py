import dataset
import utils
import models
import evals
import train

import torch
import matplotlib.pyplot as plt

def predict_image(img, model):
    xb = img.unsqueeze(0)
    yb = model(xb)
    _, preds  = torch.max(yb, dim=1)
    return preds[0].item()


for i in range(5):
    img, label = dataset.test_dataset[i]
#     plt.imshow(img[0], cmap='gray')
    print('Label:', label, ', Predicted:', predict_image(img, models.model))
    
    
result = evals.evaluate(models.model, dataset.test_loader)
# model.state_dict()

