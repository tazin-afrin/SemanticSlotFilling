from xml.dom import minidom
from xml.dom.minidom import Node
from collections import Counter, defaultdict


# input: xml files from outputs of the SEMAFOR tool
# for each sentence it outputs the frames in xml format
# returns two dictionaries:
#     1. framename dictionary: stores the words associated with each framename key
#     2. sentence dictionary: store all framenames found in each sentence with sentence id (0,1..) as key

def read_xml(filename):

    xmldoc = minidom.parse(filename)
    dict_fname = defaultdict(list)
    dict_sent = defaultdict(list)

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
                        # print(fname,',',text[s:e+1])

    return dict_fname, dict_sent


if __name__=='__main__':
    dataDir = '../Data/'

    trainxml = dataDir + 'train_noBOS.xml'
    testxml = dataDir + 'test_noBOS.xml'

    dict_fname, dict_sent = read_xml(testxml)
    print(len(dict_fname), len(dict_sent))

    for item in dict_sent:
        print(item, set(dict_sent[item]))
        break

    for item in dict_fname:
        print(item, set(dict_fname[item]))
        break