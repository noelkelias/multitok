import pandas as pd
from torch.utils.data import DataLoader
import random
import torch

#MultiTok Tokenizer
def multitok_word_encode(data, dict_values, freeze_dict, window=None):
  if window == None or window ==0:
    window=len(data)

  code = []

  index =0
  count = len(dict_values)

  while index < len(data):
    new_code = ""

    # find word
    for j in range(index + 1, min(len(data)+1, index + 1 + window)):
        word = " ".join(map(str,data[index:j]))

        #Index of longest string in dictionary
        if word in dict_values:
          new_code = str(dict_values[word])

        #Adding new val to dictionary
        if word not in dict_values:
            if not freeze_dict:
              dict_values[word] = count
              count += 1
            index += len(data[index:j]) -1
            break

        #Reached last word
        if j >= len(data) or j>= index + window:
          index += len(data[index:j])
          break

    code.append(int(new_code))

  return code

def multitok_freq_tokens(train_sentences, train_labels, test_sentences, test_labels, input_window, output_window):
  #Create own embeddings with available unique words.
  exp1_dataX= []
  exp_dataY= []

  #Build unique dictionary
  num_words = 0
  count = 0
  raw_dict_input = {}
  for sentence in train_sentences:
    for word in sentence.split():
      num_words += 1
      if word not in raw_dict_input:
        raw_dict_input[word] = count
        count += 1

  for sentence in test_sentences:
    for word in sentence.split():
      if word not in raw_dict_input:
        raw_dict_input[word] = count
        count += 1

  words = list(raw_dict_input.keys())
  random.shuffle(words)

  dict_input = dict(zip(words, raw_dict_input.values()))

  #Create tokens
  max_len = 0
  for i in range (len(train_sentences)):
    vals = multitok_word_encode(train_sentences[i].split(), dict_input, False, input_window)
    exp1_dataX.append(vals)
    exp_dataY.append([train_labels[i]])


  token_freq = {}
  for entry in exp1_dataX:
    for token in entry:
      if token in token_freq:
        token_freq[token] += 1
      else:
        token_freq[token] = 1

  df = pd.DataFrame(token_freq.items(), columns=['Token', 'Frequency'])
  print(df['Frequency'].describe())
  print(df['Frequency'].value_counts(normalize=True))

  new_dict = {}
  inv_dict = dict(zip(dict_input.values(), dict_input.keys()))
  print("Original Dict size: " + str(len(inv_dict)))

  num_removed = 0
  for token in inv_dict.keys():
    if token in token_freq and token_freq[token] < 2:
      new_token = []
      for tok in inv_dict[token].split():
        new_token.append(dict_input[tok])
      new_dict[token] = new_token

      if len(inv_dict[token].split()) > 1:
        del dict_input[inv_dict[token]]
        num_removed += 1

    else:
      new_dict[token] = [token]
  print("Number of Removed tokens: " + str(num_removed))

  #update dictionary
  smaller_dict = {}
  count = 1
  for val in dict_input.values():
    smaller_dict[val] = count
    count+=1
  print("New Dict size: " + str(len(smaller_dict)))


  new_exp1_dataX = []
  compressed_words = 0
  for sentence in exp1_dataX:
    new_sentence = []
    for word in sentence:
      new_sentence += new_dict[word]

    compressed_words += len(new_sentence)
    max_len = max(max_len, len(new_sentence))
    new_exp1_dataX.append(new_sentence)

  for i in range(len(new_exp1_dataX)):
    for j in range(len(new_exp1_dataX[i])):
      new_exp1_dataX[i][j] = smaller_dict[new_exp1_dataX[i][j]]


  #padding
  X_padded = [val + [0]*(max_len - len(val)) for val in new_exp1_dataX]

  X1 = torch.tensor(X_padded)
  Y = torch.tensor(exp_dataY)

  Y = Y.to(torch.float32)

  loader = DataLoader(list(zip(X1, Y)), shuffle=True, batch_size=1000)

  print("Finished Preprocessing, ", "Max Length:", max_len)
  print("Compression Ratio: ", compressed_words/num_words, compressed_words, num_words)

  #Evalutation
  exp1_testX = []
  exp_testY = []

  #Create tokens
  for i in range (len(test_sentences)):
    sentence_tokens = multitok_word_encode(test_sentences[i].split(), dict_input, True, output_window)

    new_sentence_tokens = []
    for word in sentence_tokens:
      new_sentence_tokens += new_dict[word]

    if len(new_sentence_tokens) <= max_len:
      exp1_testX.append(new_sentence_tokens)
      exp_testY.append([test_labels[i]])

  for i in range(len(exp1_testX)):
    for j in range(len(exp1_testX[i])):
      exp1_testX[i][j] = smaller_dict[exp1_testX[i][j]]

  #padding
  testX_padded = [val + [0]*(max_len - len(val)) for val in exp1_testX]

  test_X1 = torch.tensor(testX_padded)
  test_Y = torch.tensor(exp_testY)

  return X1, Y, loader, test_X1, test_Y, len(dict_input)
