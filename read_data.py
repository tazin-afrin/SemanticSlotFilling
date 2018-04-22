# from wordSlotDataSet import dataSet
import codecs
from collections import Counter, defaultdict


# def readdataSet(filename):
#     devData = dataSet(filename, 'train', {}, {}, {}, {})
#     print(devData.dataSet['utterances'])


# input is a file from the atis dataset with "intention"
# output is a list of [sentence, slot, intent]
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

    for i in range(len(data)):

        intent = data[i][2].split('#') # because 1 utterance can have more than 1 intent
                                    # intent in form of atis_intent
        for item in intent:
            intent_key = item[5:] # remove 'atis_' , take the rest as intent key
            dict_intent[intent_key] += dict_sent[str(i)]

    return dict_intent



if __name__== '__main__':
    dataDir = '../Data/'
    trainFile = dataDir+'atis-2.train.w-intent.iob'
    devFile = dataDir+'atis-2.dev.w-intent.iob'
    testFile = dataDir+'Data/atis.test.iob'

    data = read_data(trainFile)
    print(data[0])