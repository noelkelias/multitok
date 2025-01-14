import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics

#Model Defintion
class LSTMModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, 100)
        self.lstm = nn.LSTM(input_size=100, hidden_size=100, num_layers=1, batch_first=True)
        self.linear = nn.Linear(50, 1)
    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        # produce output
        x = self.linear(x)
        return torch.sigmoid(x)

class SimpleTransformer(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, seq_len, output_dim=1):
        super(SimpleTransformer, self).__init__()
        
        input_dim = 50257  # Vocabulary size (adjust based on your dataset)
        embed_dim = 100   
        num_heads = 2      # Number of attention heads
        seq_len = 512       # Length of the input sequence
        output_dim = 1   
 
        # Input embedding
        self.embedding = nn.Embedding(input_dim, embed_dim)
        
        # Simple self-attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        
        # Simple feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(embed_dim)
        self.layer_norm2 = nn.LayerNorm(embed_dim)
        
        # Output classifier
        self.fc = nn.Linear(embed_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
        # Positional Encoding
        self.positional_encoding = self._generate_positional_encoding(seq_len, embed_dim)

    # Basic positional encoding (sinusoidal)
    def _generate_positional_encoding(self, seq_len, embed_dim):
        position = torch.arange(seq_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * -(torch.log(torch.tensor(10000.0)) / embed_dim))
        pe = torch.zeros(seq_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def forward(self, x):
        # Embedding input and adding positional encoding
        x = self.embedding(x) + self.positional_encoding[:, :x.size(1), :]

        # Self-attention 
        x = x.permute(1, 0, 2)  # Convert (batch_size, seq_len, embed_dim) to (seq_len, batch_size, embed_dim)
        attn_output, _ = self.attention(x, x, x)
        
        # Add residual connection and normalize
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward network
        ff_output = self.ffn(x)
        
        # Add residual connection and normalize
        x = self.layer_norm2(x + ff_output)

        # Use only the last time step for classification 
        x = x[-1, :, :] 

        # Final output layer and sigmoid for binary classification
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

def train_eval(X1, Y, loader, test_X1, test_Y, input_dim, iterations, n_epochs):
  for i in range(iterations):
    #Training
    model = SimpleTransformer(input_dim)
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