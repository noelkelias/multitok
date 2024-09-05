# MultiTok: Variable-Length Tokenization for Efficient LLM Training Adapted from LZW Compression

MultiTok provides a novel
variable-length tokenizer in the sense that each token can
represent a variable number of sub-words. The advantages
of MultiTok are it (i) dynamically compresses the necessary
training data by close to 33% (ii) allows the LLM model
to close to three times faster training and (iii) maintains
performance comparable to the current state-of-the-art for tokenization [BERT](https://arxiv.org/pdf/1810.04805). Specifically, MultiTok can be used to compress repetitive words or phrases within the training data without significantly harming the model performance. We hope that Multitok can mark the beginning of using information-theoretic approaches to provide efficient, secure, and robust LLM systems.

This is an implementation for our paper "MultiTok: Variable-Length Tokenization for
Efficient LLM Training Adapted from LZW
Compression" submitted to ICASSP 2025. 

```
.
├── README.md
├── experiments
│   ├── bert.py
│   ├── bert_multitok.py
│   ├── main.py
│   ├── model.py
│   ├── multitok.py
│   ├── multitok_freq.py
│   └── random.py
├── misc
│   └── pos_emb.py
└── requirements.txt
```

## Installation
Clone the repository

```shell
git clone https://github.com/noelkelias/multitok.git
```

Install the requirements
```shell
cd multitok
pip install -r requirements.txt
```

## Datasets
Our experiments focus on three mainstream text classification datasets:

| Name |  Description |
| --- | --- |
| [IMDB](https://huggingface.co/datasets/stanfordnlp/imdb) | The IMDb Movie Reviews dataset is a binary sentiment analysis dataset consisting of 50,000 reviews from the Internet Movie Database (IMDb) labeled as positive or negative.  |
| [sst2](https://huggingface.co/datasets/stanfordnlp/sst2) | Comprises sentences from movie reviews, annotated for sentiment (positive/negative) |
| [AG-News](https://huggingface.co/datasets/fancyzhx/ag_news) | AG is a collection of news articles annoted for their topic  |

## Usage
