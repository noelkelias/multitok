from torch.utils.data import DataLoader
from transformers import AutoTokenizer
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

def bert_multitok_tokens(train_sentences, train_labels, test_sentences, test_labels, input_window, output_window):
  tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

  #MultiTok on Bert embeddings?
  exp1_dataX= []
  exp1_dataY= []

  #generate dictionary
  dict_input = {}
  for i in range (0, 30523):
    dict_input[str(i)] = i

  num_words = 0
  compressed_words = 0
  max_len = 0
  for i in range(len(train_sentences)):
    sentence_tokens = tokenizer.encode(train_sentences[i], max_length = tokenizer.model_max_length, truncation=True)
    vals = multitok_word_encode(sentence_tokens, dict_input, False, input_window)

    num_words += len(sentence_tokens)
    compressed_words += len(vals)

    max_len = max(max_len, len(vals))

    exp1_dataX.append(vals)
    exp1_dataY.append([train_labels[i]])

  print(len(dict_input))

  #padding
  X_padded = [val + [0]*(max_len - len(val)) for val in exp1_dataX]

  X1 = torch.tensor(X_padded)
  Y = torch.tensor(exp1_dataY)

  # X1 = X1.to(torch.float32)
  Y = Y.to(torch.float32)

  loader = DataLoader(list(zip(X1, Y)), shuffle=True, batch_size=1000)

  print("Finished Preprocessing, ", "Max Length:", max_len)
  print("Compression Ratio: ", num_words/compressed_words ,  num_words, compressed_words)

  #Evalutation
  exp1_testX = []
  exp1_testY = []

  #Create tokens
  for i in range (len(test_sentences)):
    bert_tokens = tokenizer.encode(test_sentences[i], max_length = tokenizer.model_max_length, truncation=True)
    sentence_tokens = multitok_word_encode(bert_tokens, dict_input, True, output_window)
    # sentence_tokens = bert_tokens
    if len(sentence_tokens) <= max_len:
      exp1_testX.append(sentence_tokens)
      exp1_testY.append([test_labels[i]])

  #padding
  testX_padded = [val + [0]*(max_len - len(val)) for val in exp1_testX]

  test_X1 = torch.tensor(testX_padded)
  test_Y = torch.tensor(exp1_testY)

  return X1, Y, loader, test_X1, test_Y, len(dict_input)
