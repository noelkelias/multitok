import pandas as pd
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, BertModel
import random
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import metrics

from LZW import *
from BERT import *
from LZW_BERT import *
from LZW_freq import *
from random import *
from model import *

#SSt
# dataset = load_dataset("glue", 'sst2')
# train_sentences = dataset['train']['sentence']
# train_labels = dataset['train']['label']

# test_sentences = dataset['validation']['sentence']
# test_labels = dataset['validation']['label']

#IMDB
dataset = load_dataset("imdb")
train_sentences = dataset['train']['text']
train_labels = dataset['train']['label']

test_sentences = dataset['test']['text']
test_labels = dataset['test']['label']

#Spam
# dataset = load_dataset("sms_spam")
# train_sentences = dataset['train']['sms'][0:5000]
# train_labels = dataset['train']['label'][0:5000]

# test_sentences = dataset['train']['sms'][5000:]
# test_labels = dataset['train']['label'][5000:]

a,b,c,d,e,f =  bert_tokens(train_sentences, train_labels, test_sentences, test_labels)
train_eval(a,b,c,d,e,f, 1, 10)