import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics

#Model Defintion
class Model(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, 40)
        self.lstm = nn.LSTM(input_size=40, hidden_size=25, num_layers=1, batch_first=True)
        self.linear = nn.Linear(25, 1)
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        # produce output
        x = self.linear(x)
        return torch.sigmoid(x)

def train_eval(X1, Y, loader, test_X1, test_Y, input_dim, iterations, n_epochs):
  for i in range(iterations):
    #Training
    model = Model(input_dim)
    loss_fn = nn.BCELoss()
    optimiser = optim.Adam(model.parameters(), lr=0.01)

    training_acc = []
    training_loss = []
    for epoch in range(1, n_epochs + 1):
      for X_batch, y_batch in loader:
          output_train = model(X_batch) # forwards pass
          loss_train = loss_fn(output_train, y_batch) # calculate loss
          optimiser.zero_grad() # set gradients to zero
          loss_train.backward() # backwards pass
          optimiser.step() # update model parameters

      train_prediction = model(X1)
      training_accuracy = torch.sum(train_prediction.round() == Y).item()/len(X1)
      training_acc.append(training_accuracy)
      training_loss.append(loss_train.item())
      print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}, Accuracy {training_accuracy}")
      prediction = model(test_X1)
      y_pred = prediction.round()
      print("Test Accuracy: ", torch.sum(y_pred == test_Y).item()/len(test_X1))

    plt.plot(list(range(0, n_epochs)), training_acc, 'g', label='Training Accuracy')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    plt.plot(list(range(0, n_epochs)), training_loss, 'g', label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print("Finished Training")

    prediction = model(test_X1)
    y_pred = prediction.round()
    print("Test Accuracy: ", torch.sum(y_pred == test_Y).item()/len(test_X1))

    y_pred = model(test_X1)
    y_pred = y_pred.detach().numpy()
    fpr, tpr, thresholds = roc_curve(test_Y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='Tokens')
    display.plot()
    plt.show()

    return model

def eval(model, test_X1, test_Y):
    prediction = model(test_X1)
    y_pred = prediction.round()
    print("Test Accuracy: ", torch.sum(y_pred == test_Y).item()/len(test_X1))

    y_pred = model(test_X1)
    y_pred = y_pred.detach().numpy()
    fpr, tpr, thresholds = roc_curve(test_Y, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='')
    display.plot()
    plt.show()