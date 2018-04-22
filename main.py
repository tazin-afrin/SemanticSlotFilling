import numpy as np

from read_data import read_data, get_intent_frames
from readxml import read_xml
from sklearn.metrics import confusion_matrix

from algo import get_featuresNames, get_feature_matrix, classify, bow, get_labels

if __name__ == '__main__':

    dataDir = '../Data/'

    trainFile = dataDir + 'atis.train.w-intent.iob'
    devFile = dataDir + 'atis-2.dev.w-intent.iob'
    testFile = dataDir + 'atis.test.w-intent.iob'

    trainxml = dataDir + 'train_noBOS.xml'
    testxml = dataDir + 'test_noBOS.xml'


    data_train = read_data(trainFile)
    dict_fname_train, dict_sent_train = read_xml(trainxml)
    print(len(data_train), len(dict_sent_train))

    data_test = read_data(testFile)
    dict_fname_test, dict_sent_test = read_xml(testxml)
    print(len(data_test), len(dict_sent_test))

    # bag of words features
    corpus_train = [txt[0] for txt in data_train]
    corpus_test = [txt[0] for txt in data_test]

    X_train_bow, X_test_bow = bow(corpus_train, corpus_test)


    # extract framenet features
    dict_intent = get_intent_frames(data_train, dict_sent_train)

    top_frames = 5
    features = get_featuresNames(dict_intent, top_frames)
    print('Number of features (frames) : ',len(features))

    X_train_framenet = get_feature_matrix(data_train, features, dict_sent_train)
    X_test_framenet = get_feature_matrix(data_test, features, dict_sent_test)


    # Concatenate bow and frame features
    X_train = np.concatenate((X_train_bow, X_train_framenet), axis=1)
    X_test = np.concatenate((X_test_bow, X_test_framenet), axis=1)

    # X_train = X_train_bow
    # X_test = X_test_bow

    # X_train = X_test_framenet
    # X_test = X_test_framenet

    y_train = get_labels(data_train)
    y_test = get_labels(data_test)

    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    classify(X_train, y_train, X_test, y_test, features)



