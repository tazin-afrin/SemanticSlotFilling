from collections import Counter
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix


from plot_confusion_matrix import plot_confusion_matrix


intent_list = ['abbreviation', 'aircraft', 'airfare', 'airline', 'airport', 'capacity', 'city',
              'distance', 'flight', 'flight_no', 'flight_time', 'ground_service','ground_fare',
              'meal', 'quantity', 'restriction', 'cheapest', 'other']

intent_num = [int(i) for i in range(1,len(intent_list)+1)]

def most_common(lst):
    return max(set(lst), key=lst.count)


def get_feature_matrix(data, features, sentence_dictionary):

    M = [[0 for i in range(len(features))] for j in range(len(sentence_dictionary))]
    df = pd.DataFrame(M)
    df.columns = list(features)

    i = 0
    for item in sentence_dictionary:
        c = Counter(sentence_dictionary[item])
        for fname in c:
            if fname in list(features):
                # print(fname, c[fname])
                df.iloc[i, df.columns.get_loc(fname)] = c[fname]

        i += 1

    X = np.array(df.as_matrix())

    return X


def get_labels(data):

    y = []

    for i in range(len(data)):
        intent = data[i][2].split('#')[0]
        intent_key = intent[5:]

        if intent_key in intent_list:
            y.append(intent_list.index(intent_key) + 1)
        else:
            y.append(18)

    return np.array(y)


def get_featuresNames(dict_intent, top_frames):

    features = set()

    for item in dict_intent:
        c = Counter(dict_intent[item])
        mc = c.most_common(top_frames)
        # print(item, '\t',len(set(dict_intent[item])) , '\t', mc)
        for m in mc:
            features.add(m[0])

    return features



def classify(X_train, y_train, X_test, y_test, features):
    print('classifying ...')
    clf = RandomForestClassifier(n_estimators=1000)
    clf.fit(X_train, y_train)

    # print('\nFeature Importance')
    # i = 0
    # ft = list(features)
    # for fi in clf.feature_importances_:
    #     print(ft[i], '\t', fi)
    #     i += 1

    y_pred = clf.predict(X_test)

    # print(list(y_test))
    # print(list(y_pred))

    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred, average='micro')
    a = accuracy_score(y_test, y_pred)
    print(a, p, r, f)


    #majority class label
    mc = most_common(list(y_test))
    y_pred_mc = [mc]*len(y_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, y_pred_mc, average='micro')
    a = accuracy_score(y_test, y_pred_mc)
    print(a, p, r, f)

    # Compute confusion matrix
    print(set(y_test))
    print(set(y_pred))

    y_test_intents = []
    y_pred_intents = []
    for i in range(len(y_test)):
        y_test_intents.append(intent_list[y_test[i]-1])
        y_pred_intents.append(intent_list[y_pred[i]-1])
    class_names = list(set(y_test_intents))

    cnf_matrix = confusion_matrix(y_test_intents, y_pred_intents, labels=class_names)
    np.set_printoptions(precision=2)

    # Plot non-normalized confusion matrix

    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=class_names,title='Confusion matrix, without normalization')

    plt.show()

def bow(corpus_train, corpus_test):

    print('vectorizing ...')
    vectorizer = CountVectorizer(ngram_range = (1, 2))
    X_train_counts = vectorizer.fit_transform(corpus_train)#.todense()
    X_test_counts = vectorizer.transform(corpus_test)#.todense()

    # print(vectorizer.vocabulary_)
    print(X_train_counts.shape, X_test_counts.shape)

    tfidf_transformer = TfidfTransformer(norm="l2")
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts).todense()
    X_test_tfidf = tfidf_transformer.transform(X_test_counts).todense()

    print(X_train_tfidf.shape, X_test_tfidf.shape)

    return X_train_tfidf, X_test_tfidf
