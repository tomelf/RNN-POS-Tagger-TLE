import data_loader
import numpy as np
import pandas as pd
import pickle
import os
import nltk
import re
import timeit

from torch.autograd import Variable
import torch

from sklearn import preprocessing, svm, metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.externals import joblib
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import cross_val_score

from util.classification.lstm_pos_tagger import LSTMPOSTagger

def main():
    meta_list, data_list = data_loader.load_data(load_train=True, load_dev=True, load_test=True)

    train_meta, train_meta_corrected, \
    dev_meta, dev_meta_corrected, \
    test_meta, test_meta_corrected = meta_list

    train_data, train_data_corrected, \
    dev_data, dev_data_corrected, \
    test_data, test_data_corrected = data_list

    EMBEDDING_DIM = 300
    HIDDEN_DIM = 200
    X_train = [[d["form"].tolist(), d["upostag"].tolist()] for d in train_data]
    X_dev = [[d["form"].tolist(), d["upostag"].tolist()] for d in dev_data]
    X_test = [[d["form"].tolist(), d["upostag"].tolist()] for d in test_data]

    word_to_ix = dict()
    tag_to_ix = dict()
    for i in range(len(X_train)):
        sent = X_train[i][0]
        tags = X_train[i][1]
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)

    EMBEDDING_DIM = len(word_to_ix.keys())
    HIDDEN_DIM = int(len(word_to_ix.keys())/2)

    model = LSTMPOSTagger(EMBEDDING_DIM, HIDDEN_DIM, word_to_ix, tag_to_ix)
    model.cuda()
    model.set_train_data(X_train)
    # model.set_dev_data(X_dev)
    model.train(epoch=10)

    preds = []
    actuals = []
    tag_to_ix_list = sorted(tag_to_ix.items(), key=lambda k: k[1])

    print("Evaluate test accuracy")
    for i in range(len(X_test)):
        actual = model.prepare_sequence(X_train[i][1], tag_to_ix)
        _, pred = torch.max(model.test(X_train[i][0]), 1)

        preds += pred.data.cpu().numpy().tolist()
        actuals += actual.data.cpu().numpy().tolist()

    print("Accuracy: {}".format(metrics.accuracy_score(actuals, preds)))
    print("Predicted tags: {}".format(len(preds)))
    print("# of POS tags: {}".format(len(tag_to_ix_list)))
    print("POS tags: {}".format(tag_to_ix_list))

if __name__ == "__main__":
    main()
