# -*- coding: utf-8 -*-
from __future__ import division
from keras.layers import Masking
from keras.layers.core import Activation, Dense, Dropout, RepeatVector, SpatialDropout1D
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU, LSTM
from keras.layers.wrappers import TimeDistributed, Bidirectional
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.utils import np_utils
import collections
import matplotlib.pyplot as plt
import numpy as np
from numpy import zeros
from numpy import asarray
import keras.backend as K
import os
from keras.callbacks import Callback
from sklearn.metrics import precision_recall_fscore_support

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

EMBEDDING = '../glove.6B/glove.6B.100d.txt'
DATA_DIR = "../../data"

class Metrics(Callback):

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        val_predict = (np.asarray(self.model.predict(self.validation_data[0]))).round()
        val_targ = self.validation_data[1]
        print "===call back===="
        print val_predict,val_targ
        # _val_f1 = f1_score(val_targ, val_predict)
        # _val_recall = recall_score(val_targ, val_predict)
        # _val_precision = precision_score(val_targ, val_predict)
        # self.val_f1s.append(_val_f1)
        # self.val_recalls.append(_val_recall)
        # self.val_precisions.append(_val_precision)
        # print " — val_f1: % f — val_precision: % f — val_recall % f" % (_val_f1, _val_precision, _val_recall)
        return


def explore_data(datadir, datafiles):
    counter = collections.Counter()
    maxlen = 0
    for datafile in datafiles:
        fdata = open(os.path.join(datadir, datafile), "rb")
        for line in fdata:
            words = line.strip().split()
            if len(words) > maxlen:
                maxlen = len(words)
            for word in words:
                counter[word] += 1
        fdata.close()
    return maxlen, counter


def create_embedding(s_word2id):

    embeddings_index = dict()

    with open(EMBEDDING,'r+') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs

        # print('Loaded %s word vectors.' % len(embeddings_index))

    # create a weight matrix for words in training docs
    embedding_matrix = zeros((len(s_word2id), 100))

    print embedding_matrix

    for word, i in s_word2id.items():
        # print word,i
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    # print embedding_matrix
    return embedding_matrix


def build_tensor(filename, numrecs, word2index, maxlen,
                 make_categorical=False):
    print "===file name==="
    print filename
    fin = open(filename, "r+")
    lines = fin.readlines()
    data = np.empty((len(lines),), dtype=list)

    i = 0
    for line in lines:
        wids = []
        for word in line.strip().split():
            if word2index.has_key(word):
                wids.append(word2index[word])
            else:
                wids.append(word2index["UNK"])
        if make_categorical:
            data[i] = np_utils.to_categorical(
                wids, num_classes=len(word2index))
        else:
            data[i] = wids
        i += 1
    fin.close()
    print "===data==="
    print(data)
    pdata = sequence.pad_sequences(data, maxlen=maxlen)
    print "====paded data===="
    print pdata
    return pdata


def evaluate_model(model, Xtest, Ytest, batch_size):
    pass




s_maxlen, s_counter = explore_data(DATA_DIR, ["train",
                                              "test"])
t_maxlen, t_counter = explore_data(DATA_DIR, ["train_tag",
                                              "test.tag"])

print "====counter===="
print t_counter
print(s_maxlen, len(s_counter), t_maxlen, len(t_counter))
# 7 21 7 9
# maxlen: 7
# size of source vocab: 21
# size of target vocab: 9

# lookup tables
s_word2id = {k: v + 1 for v, (k, _) in enumerate(s_counter.most_common())}
s_word2id["PAD"] = 0
print "=====word2id===="
print s_word2id["BOS"]
print s_word2id["EOS"]
s_id2word = {v: k for k, v in s_word2id.items()}
t_pos2id = {k: v + 1 for v, (k, _) in enumerate(t_counter.most_common())}
t_pos2id["PAD"] = 0
print "====pos2id===="
print t_pos2id["O"]
t_id2pos = {v: k for k, v in t_pos2id.items()}
embedding_matrx = create_embedding(s_word2id)
print "========number of class======"
print len(set(t_id2pos))
print len(set(t_pos2id))

# vectorize data
MAX_SEQLEN = 20

Xtrain = build_tensor(os.path.join(DATA_DIR, "train"),
                      4978, s_word2id, MAX_SEQLEN)
Xtest = build_tensor(os.path.join(DATA_DIR, "test"),
                     4978, s_word2id, MAX_SEQLEN)
Ytrain = build_tensor(os.path.join(DATA_DIR, "train_tag"),
                      4978, t_pos2id, MAX_SEQLEN, make_categorical=True)
Ytest = build_tensor(os.path.join(DATA_DIR, "test.tag"),
                     4978, t_pos2id, MAX_SEQLEN, make_categorical=True)
print "===tensor==="
print "shape"
print(Xtrain.shape, Xtest.shape, Ytrain.shape, Ytest.shape)
print "===tensor self===="
print Xtrain,Xtest,Ytrain,Ytest

# define network
EMBED_SIZE = 100
HIDDEN_SIZE = 32

BATCH_SIZE = 32
NUM_EPOCHS = 10

model = Sequential()
M = Masking(mask_value=0.)
M._input_shape = (MAX_SEQLEN,100)
model.add(Embedding(len(s_word2id), EMBED_SIZE,weights=[embedding_matrx],trainable=False,
                    input_length=MAX_SEQLEN))
# model.add(SpatialDropout1D(Dropout(0.2)))
# model.add(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
# model.add(GRU(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2))
# model.add(Bidirectional(LSTM(HIDDEN_SIZE, dropout=0.2, recurrent_dropout=0.2)))
# model.add(RepeatVector(MAX_SEQLEN))
# model.add(LSTM(HIDDEN_SIZE, return_sequences=True))
# model.add(GRU(HIDDEN_SIZE, return_sequences=True))
model.add(Bidirectional(LSTM(HIDDEN_SIZE, return_sequences=True,recurrent_dropout=0.2)))
model.add(TimeDistributed(Dense(len(t_pos2id))))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["acc"])

# print Xtrain.shape
# print Xtest.shape

metrics = Metrics()
history = model.fit(Xtrain, Ytrain, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS,
                    validation_data=[Xtest, Ytest])#,callbacks=[metrics])



# evaluate model
print "=======Evaluation=========(include padding)"
score, acc = model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
# BATCH_SIZEprint model.evaluate(Xtest, Ytest, batch_size=BATCH_SIZE)
print("Test score: %.3f, accuracy: %.3f" % (score, acc))



# custom evaluate
print "=======Evaluation=========(slot only)"
hit_rates = []
num_iters = Xtest.shape[0] // BATCH_SIZE

count_i = 0
count_j = 0
i_len = 0
j_len = 0
Y_labels = []
Y_pred = []

# print "====range(num_iters - 1)===="
# print num_iters - 1
for i in range(num_iters - 1):
    xtest = Xtest[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
    ytest = np.argmax(Ytest[i * BATCH_SIZE: (i + 1) * BATCH_SIZE], axis=2)
    ytest_ = np.argmax(model.predict(xtest), axis=2)
    #    print(ytest.shape, ytest_.shape)
    for j in range(BATCH_SIZE):
        # print "=====j's sample====="
        #        print("sentence:  " + " ".join([s_id2word[x] for x in xtest[j].tolist()]))
        #        print("predicted: " + " ".join([t_id2pos[y] for y in ytest_[j].tolist()]))
        #        print("label:     " + " ".join([t_id2pos[y] for y in ytest[j].tolist()]))
        word_indices = np.nonzero(xtest[j])

        pos_labels = ytest[j][word_indices]
        pos_pred = ytest_[j][word_indices]
        hit_rates.append(np.sum(pos_labels == pos_pred) / len(pos_pred))

        Y_labels += list(pos_labels)
        Y_pred += list(pos_pred)

        count_i += 1
        count_j += 1
        i_len += len(pos_labels)
        j_len += len(pos_pred)

    # print "=====pos_pred, pos_lables===="
    # print ytest, ytest_
    # break
accuracy = sum(hit_rates) / len(hit_rates)
print("accuracy: {:.3f}".format(accuracy))

# print count_i,count_j
# print i_len,j_len
# print "=====Y_labels====="
# print Y_labels
# print "=====Y_pred====="
# print Y_pred

# Y_labels = asarray(Y_labels).flatten()
# Y_pred = asarray(Y_pred).flatten()

eachClass_F1 = precision_recall_fscore_support(Y_labels,Y_pred,average='micro')#,labels=listofClass,average=None)
f1 = f1_score(Y_labels,Y_pred,average='micro')#,labels=listofClass,average=None)
print "=======Y lables, Y pred, precision, recall , F1 ========="
print eachClass_F1
# print "=======Y lables, Y pred, f1========="
# print f1


# print "=====Ytest,Ypredict===="
# Ypredict_conf = np.argmax(model.predict(Xtest), axis=2).flatten()
# Ytrue_conf = np.argmax(Ytest,axis=2).flatten()
# print Ytrue_conf
# print Ypredict_conf
#     # print ytest, ytest_
# confusion_matrix = confusion_matrix(Ytrue_conf, Ypredict_conf)
# print confusion_matrix
# print len(confusion_matrix)
#
# listofClass = [x for x in range(1,128)]
# eachClass_F1 = precision_recall_fscore_support(Ytrue_conf,Ypredict_conf,average='weighted')#,labels=listofClass,average=None)
# f1 = f1_score(Ytrue_conf,Ypredict_conf,average='weighted')#,labels=listofClass,average=None)
# print "=======precision, recall , F1========="
# print eachClass_F1
# print "=======f1========="
# print f1





# prediction
pred_ids = np.random.randint(0, 893, 5)
for pred_id in pred_ids:
    xtest = Xtest[pred_id].reshape(1, 20)
    ytest_ = np.argmax(model.predict(xtest), axis=2)
    ytest = np.argmax(Ytest[pred_id], axis=1)
    print("sentence:  " + " ".join([s_id2word[x] for x in xtest[0].tolist()]))
    print("predicted: " + " ".join([t_id2pos[y] for y in ytest_[0].tolist()]))
    print("label:     " + " ".join([t_id2pos[y] for y in ytest.tolist()]))
    word_indices = np.nonzero(xtest)[1]
    ypred_tags = ytest_[0][word_indices]
    ytrue_tags = ytest[word_indices]
    hit_rate = np.sum(ypred_tags == ytrue_tags) / len(ypred_tags)
    print("hit rate: {:.3f}".format(hit_rate))
    print()

# plot loss and accuracy
plt.subplot(211)
plt.title("Accuracy")
plt.plot(history.history["acc"], color="g", label="Train")
plt.plot(history.history["val_acc"], color="b", label="Validation")
plt.legend(loc="best")

plt.subplot(212)
plt.title("Loss")
plt.plot(history.history["loss"], color="g", label="Train")
plt.plot(history.history["val_loss"], color="b", label="Validation")
plt.legend(loc="best")

plt.tight_layout()
plt.show()