import random
import torch
from torch.utils.data import DataLoader

#Create own embeddings with available unique words.
def random_tokens():
  exp1_dataX= []
  exp_dataY= []

  for i in range (10000):
    exp1_dataX.append(random.sample(range(0, 100000), 150))
    exp_dataY.append([random.randint(0,1)])

  X1 = torch.tensor(exp1_dataX)
  Y = torch.tensor(exp_dataY)

  Y = Y.to(torch.float32)

  loader = DataLoader(list(zip(X1, Y)), shuffle=True, batch_size=1000)

  #Evalutation
  exp1_testX = []
  exp_testY = []

  for i in range (10000):
    exp1_testX.append(random.sample(range(0, 100000), 150))
    exp_testY.append([random.randint(0,1)])

  test_X1 = torch.tensor(exp1_testX)
  test_Y = torch.tensor(exp_testY)

  return X1, Y, loader, test_X1, test_Y, 100000