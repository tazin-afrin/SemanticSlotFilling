from xml.dom import minidom
from xml.dom.minidom import Node
from collections import Counter, defaultdict

xmldoc = minidom.parse('../Data/test_noBOS.xml')
dict = defaultdict(list)

sentence = xmldoc.getElementsByTagName('sentence')
for node in sentence:
    text = node.getElementsByTagName('text')
    text = text[0].firstChild.nodeValue
    # print(text)
    annotations = node.getElementsByTagName('annotationSet')
    for a in annotations:
        fname = a.attributes['frameName'].value
        layers = a.getElementsByTagName('layer')
        for layer in layers:
            if layer.attributes['name'].value == 'Target':
                label = layer.getElementsByTagName('label')
                for l in label:
                    s = int(l.attributes['start'].value)
                    e = int(l.attributes['end'].value)
                    dict[fname].append(text[s:e+1])
                    # print(fname,',',text[s:e+1])

for item in dict:
    print(item)
    print(dict[item])
    print('')


itemlist = xmldoc.getElementsByTagName('annotationSet')
print(len(itemlist))

# framNames = []
# for s in itemlist:
#     framNames.append(s.attributes['frameName'].value)
#
# c = Counter(framNames)
#
# for item in c:
#     print(item, c[item])