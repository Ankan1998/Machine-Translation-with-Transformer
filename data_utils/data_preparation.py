import torch
from torchtext.datasets import Multi30k, IWSLT, WMT14
from torchtext.data import Field, BucketIterator

import spacy
import numpy as np
import os
import random
import math
import time

SEED = 9999
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
spacy_de = spacy.load('de_core_news_sm')
spacy_en = spacy.load('en_core_web_sm')


def tokenize_de(text):
    # Reversing sentences to get long term benefit
    return [tok.text for tok in spacy_de.tokenizer(text)][::-1]


def tokenize_en(text):
    return [tok.text for tok in spacy_en.tokenizer(text)]


src = Field(tokenize=tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

trg = Field(tokenize=tokenize_en,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

class DataCreator:

    def data_loader(self, data_src='data_utils/.data', datasets='multi30k'):
        start = time.time()
        if datasets.lower() == "multi30k":
            train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                                fields=(src, trg),
                                                                root=data_src)

        elif datasets.lower() == "wmt14":
            train_data, valid_data, test_data = WMT14.splits(exts=('.de', '.en'),
                                                             fields=(src, trg),
                                                             root=data_src)

        elif datasets.lower() == "iwslt":
            train_data, valid_data, test_data = IWSLT.splits(exts=('.de', '.en'),
                                                             fields=(src, trg),
                                                             root=data_src)
        else:
            train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'),
                                                                fields=(src, trg),
                                                                root=data_src)
        end = time.time()
        print(f"Time taken to download data : {end - start}")

        print(f"Number of training examples: {len(train_data.examples)}")
        print(f"Number of validation examples: {len(valid_data.examples)}")
        print(f"Number of testing examples: {len(test_data.examples)}")

        return train_data, valid_data, test_data


    def vocab_builder(self, training_data, min_freq=2):
        src.build_vocab(training_data, min_freq=min_freq)
        trg.build_vocab(training_data, min_freq=min_freq)

        print(f"Unique tokens in source (de) vocabulary: {len(src.vocab)}")
        print(f"Unique tokens in target (en) vocabulary: {len(trg.vocab)}")

        return src, trg


    def data_iterator(self, train_data, valid_data, test_data, batch_size=64):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
            (train_data, valid_data, test_data),
            batch_size=batch_size,
            device=DEVICE)

        return train_iterator, valid_iterator, test_iterator


if __name__ == "__main__":
    data_c = DataCreator()
    train_data, val_data, test_data = data_c.data_loader('.data')

    source, target = data_c.vocab_builder(train_data)
    #print(target.vocab.stoi[target.pad_token])
    #train_itr, val_itr, test_itr = data_c.data_iterator(train_data, val_data, test_data)

