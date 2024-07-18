from torch.utils.data import DataLoader
import random
import torch

#LZW Tokenizer
def lwx_word_encode(data, dict_values, freeze_dict, window=None):
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

def lzw_tokens(train_sentences, train_labels, test_sentences, test_labels, input_window, output_window):
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
  compressed_words = 0
  for i in range (len(train_sentences)):
    vals = lwx_word_encode(train_sentences[i].split(), dict_input, False, input_window)
    compressed_words += len(vals)
    max_len = max(max_len, len(vals))
    exp1_dataX.append(vals)
    exp_dataY.append([train_labels[i]])


  #padding
  X_padded = [val + [0]*(max_len - len(val)) for val in exp1_dataX]

  X1 = torch.tensor(X_padded)
  Y = torch.tensor(exp_dataY)
  Y = Y.to(torch.float32)

  loader = DataLoader(list(zip(X1, Y)), shuffle=True, batch_size=1000)

  print("Finished Preprocessing, ", "Max Length:", max_len)
  print("Compression Ratio: ", num_words/compressed_words ,  num_words, compressed_words)

  #Evalutation
  exp1_testX = []
  exp_testY = []

  #Create tokens
  for i in range (len(test_sentences)):
    sentence_tokens = lwx_word_encode(test_sentences[i].split(), dict_input, True, output_window)
    if len(sentence_tokens) <= max_len:
      exp1_testX.append(sentence_tokens)
      exp_testY.append([test_labels[i]])

  #padding
  testX_padded = [val + [0]*(max_len - len(val)) for val in exp1_testX]

  test_X1 = torch.tensor(testX_padded)
  test_Y = torch.tensor(exp_testY)

  return X1, Y, loader, test_X1, test_Y, len(dict_input)