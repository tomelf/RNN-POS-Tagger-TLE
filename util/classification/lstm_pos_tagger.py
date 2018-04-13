import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd

from gensim.models import KeyedVectors
from sklearn import metrics

class LSTMPOSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocabs, tagset):
        super(LSTMPOSTagger, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocabs = vocabs
        self.tagset = tagset

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        self.hidden2tag = nn.Linear(hidden_dim, len(tagset))
        self.hidden = self.init_hidden()

    def google_w2v_embedding(self, word):
        if not hasattr(self, 'w2v'):
            print("Loading word2vec model: GoogleNews-vectors-negative300.bin")
            self.w2v = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        return self.w2v[word].tolist() if word in self.w2v else self.w2v["_"].tolist()

    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())

    def forward(self, sentence):
        lstm_out, self.hidden = self.lstm(
            sentence.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.softmax(tag_space, dim=1)
        # tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

    def prepare_sequence(self, seq, to_ix):
        # idxs = [self.to_one_hot(to_ix[w]) for w in seq if w in to_ix]
        idxs = [to_ix[w] for w in seq if w in to_ix]
        tensor = torch.LongTensor(idxs).cuda()
        return autograd.Variable(tensor).cuda()

    def prepare_sequence_w2v(self, seq):
        idxs = [self.google_w2v_embedding(w) for w in seq]
        tensor = torch.FloatTensor(idxs).cuda()
        return autograd.Variable(tensor).cuda()

    def set_train_data(self, X_train):
        self.X_train = X_train

    def set_dev_data(self, X_dev):
        self.X_dev = X_dev

    def train(self, epoch=300):
        word_to_ix = self.vocabs
        tag_to_ix = self.tagset

        # loss_function = nn.MSELoss().cuda()
        # loss_function = nn.NLLLoss().cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(self.parameters(), lr=0.1)

        self.train_loss = []
        self.dev_loss = []
        self.train_accuracy = []
        self.dev_accuracy = []

        print("Start model training")
        for e in range(epoch):
            for i in range(len(self.X_train)):
                sentence = self.X_train[i][0]
                tags = self.X_train[i][1]

                self.zero_grad()
                self.hidden = self.init_hidden()

                sentence_in = self.prepare_sequence_w2v(sentence)
                targets = self.prepare_sequence(tags, tag_to_ix)

                tag_scores = self(sentence_in)

                loss = loss_function(tag_scores, targets)
                loss.backward()
                optimizer.step()

            if (e+1)%1 == 0:
                print("Finish training epochs {}".format(e+1))

            if hasattr(self, 'X_dev'):
                model = self

                print("Evalute train loss and accuracy")
                actuals = []
                preds = []
                running_loss = 0.0
                for i in range(len(self.X_train)):
                    actual = model.prepare_sequence(self.X_train[i][1], tag_to_ix)
                    pred = model.test(self.X_train[i][0])

                    loss = loss_function(pred, actual)
                    running_loss += loss.data[0]

                    _, pred = torch.max(pred, 1)
                    preds += pred.data.cpu().numpy().tolist()
                    actuals += actual.data.cpu().numpy().tolist()
                self.train_loss.append(running_loss)
                self.train_accuracy.append(metrics.accuracy_score(actuals, preds))

                print("Evalute dev loss and accuracy")
                actuals = []
                preds = []
                running_loss = 0.0
                for i in range(len(self.X_dev)):
                    actual = model.prepare_sequence(self.X_dev[i][1], tag_to_ix)
                    pred = model.test(self.X_dev[i][0])

                    loss = loss_function(pred, actual)
                    running_loss += loss.data[0]

                    _, pred = torch.max(pred, 1)
                    preds += pred.data.cpu().numpy().tolist()
                    actuals += actual.data.cpu().numpy().tolist()
                self.dev_loss.append(running_loss)
                self.dev_accuracy.append(metrics.accuracy_score(actuals, preds))

                pd.DataFrame({
                    "train_loss": self.train_loss,
                    "train_accuracy": self.train_accuracy,
                    "dev_loss": self.dev_loss,
                    "dev_accuracy": self.dev_accuracy,
                }).to_csv("training_stats.csv")

    def test(self, sent_vec):
        # sent = self.prepare_sequence(sent_vec, self.vocabs)
        sent = self.prepare_sequence_w2v(sent_vec)
        return self(sent)

    def get_training_loss(self):
        return self.training_loss

    def to_one_hot(self, target):
        # oneHot encoding
        return [1 if i==target else 0 for i in range(len(self.tagset))]
