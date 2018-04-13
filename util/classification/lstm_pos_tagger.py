import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from gensim.models import KeyedVectors

class LSTMPOSTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocabs, tagset):
        super(LSTMPOSTagger, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocabs = vocabs
        self.tagset = tagset

        # self.word_embeddings = nn.Embedding(len(vocabs), embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, len(tagset))
        self.hidden = self.init_hidden()

    def google_w2v_embedding(self, word):
        if not hasattr(self, 'w2v'):
            print("Loading word2vec model: GoogleNews-vectors-negative300.bin")
            self.w2v = KeyedVectors.load_word2vec_format('word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        return self.w2v[word].tolist() if word in self.w2v else self.w2v["_"].tolist()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda(),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).cuda())

    def forward(self, sentence):
        # embeds = self.word_embeddings(sentence)
        embeds = sentence
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
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

    def train(self, epoch=300):
        word_to_ix = self.vocabs
        tag_to_ix = self.tagset

        # loss_function = nn.MSELoss().cuda()
        # loss_function = nn.NLLLoss().cuda()
        loss_function = nn.CrossEntropyLoss().cuda()
        optimizer = optim.SGD(self.parameters(), lr=0.1)

        print("Start model training")
        for e in range(epoch):  # again, normally you would NOT do 300 epochs, it is toy data
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

    def test(self, sent_vec):
        # sent = self.prepare_sequence(sent_vec, self.vocabs)
        sent = self.prepare_sequence_w2v(sent_vec)
        return self(sent)

    def to_one_hot(self, target):
        # oneHot encoding
        return [1 if i==target else 0 for i in range(len(self.tagset))]
