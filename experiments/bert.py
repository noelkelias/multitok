from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import torch


def bert_tokens(train_sentences, train_labels, test_sentences, test_labels):
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

  #Create own embeddings with available unique words.
  exp1_dataX= []
  exp_dataY= []

  #Create tokens
  max_len = 0
  for i in range (len(train_sentences)):
    sentence_tokens = tokenizer.encode(train_sentences[i], max_length = tokenizer.model_max_length, truncation=True)

    max_len = max(max_len, len(sentence_tokens))


    exp1_dataX.append(sentence_tokens)
    exp_dataY.append([train_labels[i]])

  #padding
  X_padded = [val + [0]*(max_len - len(val)) for val in exp1_dataX]

  X1 = torch.tensor(X_padded)
  Y = torch.tensor(exp_dataY)

  Y = Y.to(torch.float32)

  loader = DataLoader(list(zip(X1, Y)), shuffle=True, batch_size=1000)

  print("Finished Preprocessing, ", "Max Length:", max_len)

  #Evalutation
  exp1_testX = []
  exp_testY = []

  #Create tokens
  for i in range (len(test_sentences)):
    sentence_tokens = tokenizer.encode(test_sentences[i], max_length = tokenizer.model_max_length, truncation=True)
    if len(sentence_tokens) <= max_len:
      exp1_testX.append(sentence_tokens)
      exp_testY.append([test_labels[i]])

  #padding
  testX_padded = [val + [0]*(max_len - len(val)) for val in exp1_testX]

  test_X1 = torch.tensor(testX_padded)
  test_Y = torch.tensor(exp_testY)

  return X1, Y, loader, test_X1, test_Y, 30522