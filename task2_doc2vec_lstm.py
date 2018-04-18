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
    meta_list, data_list = data_loader.load_data(load_train=True, load_dev=False, load_test=False)

    train_meta, train_meta_corrected = meta_list
    train_data, train_data_corrected = data_list
    
    all_meta = train_meta
    all_data = train_data
    languages = all_meta["native_language"].unique()
    
    all_data_df = pd.DataFrame(train_data)
    
    all_data = [[d["form"].tolist(), d["upostag"].tolist()] for d in all_data]
    word_to_ix = dict()
    tag_to_ix = dict()
    for i in range(len(all_data)):
        sent = all_data[i][0]
        tags = all_data[i][1]
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
        for tag in tags:
            if tag not in tag_to_ix:
                tag_to_ix[tag] = len(tag_to_ix)
    if "_" not in word_to_ix:
        word_to_ix["_"] = len(word_to_ix.keys())
    if "_" not in tag_to_ix:
        tag_to_ix["_"] = len(tag_to_ix.keys())

    row_names = languages
    col_names = languages
    result = []
    for idx, train_language in enumerate(languages):
        result1 = []
        print("Train RNN model on {} native learners".format(train_language))
        idxes = (train_meta[train_meta["native_language"]==train_language].index-1).tolist()
        X_train = all_data_df.iloc[idxes,:].values
        X_train = [[d[0]["form"].tolist(), d[0]["upostag"].tolist()] for d in X_train]
        print("# of train data: {}".format(len(X_train)))
        EMBEDDING_DIM = 300
        HIDDEN_DIM = 200
        model = LSTMPOSTagger(EMBEDDING_DIM, HIDDEN_DIM, tag_to_ix, bidirectional=True)
        model.cuda()
        model.set_train_data(X_train)
        model.train(epoch=500, lr=0.5)

        for language in languages:
            idxes = (train_meta[train_meta["native_language"]==language].index-1).tolist()
            X_test = all_data_df.iloc[idxes,:].values
            X_test = [[d[0]["form"].tolist(), d[0]["upostag"].tolist()] for d in X_test]
            preds = []
            actuals = []
            tag_to_ix_list = sorted(tag_to_ix.items(), key=lambda k: k[1])
            print("Evaluate test accuracy")
            for i in range(len(X_test)):
                actual = model.prepare_sequence(X_test[i][1], tag_to_ix)
                _, pred = torch.max(model.test(X_test[i][0]), 1)
                preds += pred.data.cpu().numpy().tolist()
                actuals += actual.data.cpu().numpy().tolist()
            print("Test Language: {}".format(language))
            print("# of test data: {}".format(len(X_test)))
            print("Accuracy: {}\n".format(metrics.accuracy_score(actuals, preds)))
            result1.append(metrics.accuracy_score(actuals, preds))
        result.append(result1)
        result_df = pd.DataFrame(result, index=row_names[0:idx+1], columns=col_names)
        result_df.to_csv("task2_result.csv")

if __name__ == "__main__":
    main()
