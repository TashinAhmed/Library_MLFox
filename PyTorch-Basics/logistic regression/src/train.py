import dataset
import utils
import models
import evals

import torch
import matplotlib.pyplot as plt


def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
    history = []
    optimizer = opt_func(models.model.parameters(), lr)
    for epoch in range(epochs):
        # Training Phase 
        for batch in train_loader:
            loss = models.model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        # Validation phase
        result = evals.evaluate(models.model, dataset.val_loader)
        models.model.epoch_end(epoch, result)
        history.append(result)
    return history

result0 = evals.evaluate(models.model, dataset.val_loader)

history1 = fit(5, 0.001, models.model, dataset.train_loader, dataset.val_loader)
history2 = fit(5, 0.001, models.model, dataset.train_loader, dataset.val_loader)
history3 = fit(5, 0.001, models.model, dataset.train_loader, dataset.val_loader)
history4 = fit(5, 0.001, models.model, dataset.train_loader, dataset.val_loader)

# Replace these values with your results
history = [result0] + history1 + history2 + history3 + history4
accuracies = [result['val_acc'] for result in history]
plt.plot(accuracies, '-x')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.title('Accuracy vs. No. of epochs')

torch.save(models.model.state_dict(), 'mnist-logistic.pth')
