# MultiTok: Variable-Length Tokenization for Efficient LLMs Adapted from LZW Compression

MultiTok provides a novel
variable-length tokenizer in the sense that each token can
represent a variable number of sub-words. We demonstrate that MultiTok can be
utilized as both a stand-alone tokenizer and an add-on to existing tokenizers.
The advantages of MultiTok are it (i) dynamically compresses the necessary
training data by close to 33% (ii) allows the LLM model
to close to three times faster training and (iii) maintains
performance comparable to the current state-of-the-art tokenizers [BERT](https://arxiv.org/pdf/1810.04805) and [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Specifically, MultiTok can be used to compress repetitive words or phrases within the training data without significantly harming the model performance. We hope that Multitok can mark the beginning of using information-theoretic approaches to provide efficient, secure, and robust LLM systems.

This is an implementation for our paper "MultiTok: Variable-Length Tokenization for
Efficient LLMs Adapted from LZW
Compression". The full paper can be accessed at [https://arxiv.org/abs/2410.21548](https://arxiv.org/abs/2410.21548).

```
.
├── README.md
├── experiments
│   ├── bert.py
│   ├── bert_multitok.py
│   ├── gpt2.py
│   ├── gpt2_multitok.py
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
The experiments can be found in the `experiments` folder. Specifically, we point users to the MultiTok tokenization implmentation found in [multitok.py](experiments/multitok.py). We demonstrate applying MultiTok tokenization on BERT and GPT-2 tokens in [bert_multitok.py](experiments/bert_multitok.py) and [gpt2_multitok.py](experiments/gpt2_multitok.py). Additionally, we include a frequency analysis component in [multitok_freq.py](experiments/multitok_freq.py) that can be optionally added to MultiTok for improved performance. Code for a few sample tests end-to-end tests with these tokenization schemes trained on a basic transformer [model](experiments/model.py) in [main.py](experiments/main.py).

We encourage users to modify the different parameters and experiment with varying datasets to utilize MultiTok in their own pipelines.

## Citation
The full paper can be accessed at [https://arxiv.org/abs/2410.21548](https://arxiv.org/abs/2410.21548).

```
@unpublished{elias25,
    author = {Noel Elias, Homa Esfahanizadeh, H. Kaan Kale, Sriram Vishwanath, and Muriel Medard},
    title = {MultiTok: Variable-Length Tokenization for Efficient LLMs Adapted from LZW Compression},
    note = {Manuscript submitted for publication}, 
    year = {2025}
}
```
