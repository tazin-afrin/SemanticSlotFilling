from xml.dom import minidom
from collections import Counter, defaultdict

import codecs
import pandas as pd
import numpy as np


def read_data(filename):
    '''
    :param filename: train or test file from atis
    :return: list of list [sentence, slot, intent]
    '''
    data = []

    with codecs.open(filename, "r", encoding='utf-8') as f_src:

        for line_src in f_src:

            line_src = line_src.replace('\n', '').replace('\r', '').strip()
            sentence, slot = line_src.split('\t')
            sentence = sentence.strip('BOS').strip('EOS').strip(' ')
            intent = slot.split(' ')[-1]
            slot = ' '.join(slot.split(' ')[0:-1])

            data.append([sentence,slot,intent])

    return data

# find for each intent store the frames
def get_intent_frames(data, dict_sent):

    dict_intent = defaultdict(list)
    intent_stat = defaultdict(int)

    for i in range(len(data)):

        intent = data[i][2].split('#') # because 1 utterance can have more than 1 intent
                                    # intent in form of atis_intent
        for item in intent:
            intent_key = item[5:] # remove 'atis_' , take the rest as intent key
            dict_intent[intent_key] += dict_sent[str(i)]
            intent_stat[intent_key] += 1

    return dict_intent

def get_featuresNames(dict_intent, top_frames):

    features = set()

    for item in dict_intent:
        c = Counter(dict_intent[item])
        mc = c.most_common(top_frames)
        # print(item, '\t',len(set(dict_intent[item])) , '\t', mc)
        for m in mc:
            features.add(m[0])

    return features

def read_xml(filename):

    xmldoc = minidom.parse(filename)
    dict_fname = defaultdict(list)
    dict_sent = defaultdict(list)
    dict_word = defaultdict(list)

    sentence = xmldoc.getElementsByTagName('sentence')
    # print(len(sentence))

    for node in sentence:

        id = node.attributes['ID'].value
        text = node.getElementsByTagName('text')
        text = text[0].firstChild.nodeValue
        annotations = node.getElementsByTagName('annotationSet')
        dict_sent[id] = []

        for a in annotations:
            fname = a.attributes['frameName'].value
            dict_sent[id].append(fname)
            layers = a.getElementsByTagName('layer')
            for layer in layers:
                if layer.attributes['name'].value == 'Target':
                    label = layer.getElementsByTagName('label')
                    for l in label:
                        s = int(l.attributes['start'].value)
                        e = int(l.attributes['end'].value)
                        dict_fname[fname].append(text[s:e+1])
                        dict_word[text[s:e+1]].append(fname)
                        # print(fname,',',text[s:e+1])

    return dict_fname, dict_sent, dict_word

def get_feature_dict(features, word_dictionary):

    features = list(features)

    M = [0 for i in range(len(features))]
    semdict = defaultdict(lambda: M)

    for item in word_dictionary:
        M = [0 for i in range(len(features))]
        c = Counter(word_dictionary[item])
        for fname in c:
            if fname in features:
                i = features.index(fname)
                M[i] = c[fname]

        semdict[item] = M


    return semdict


def create_semantic_dict(trainFile, trainxml):

    data_train = read_data(trainFile)
    dict_fname_train, dict_sent_train, dict_word_train = read_xml(trainxml)
    print(len(data_train), len(dict_sent_train))

    # extract framenet features
    dict_intent = get_intent_frames(data_train, dict_sent_train)

    top_frames = 5
    features = get_featuresNames(dict_intent, top_frames)
    print('Number of features (frames) : ', len(features))

    semantic_dict = get_feature_dict(features, dict_word_train)

    return semantic_dict


if __name__ == '__main__':

    dataDir = '../Data/'

    trainFile = dataDir + 'atis.train.w-intent.iob'
    trainxml = dataDir + 'train_noBOS.xml'



    semantic_dict = create_semantic_dict(trainFile, trainxml)

    print(semantic_dict['flight'])
    print(semantic_dict['test'])
    print(semantic_dict['xyz'])
