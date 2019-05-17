from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
from xml.dom import minidom

print(np.__version__)
print(tf.__version__)


# To extract the chat with labels
df = pd.read_excel(r'BayzickBullyingData/Human Concensus/Packet1Concensus.xlsx')
files = np.array(df['Bully Tracer Consensus'])
labels = np.array(df['Unnamed: 1'])
files_to_labels = dict(zip(files[2:], labels[2:]))

data1 = list()
lbls = list()
for i in files_to_labels.keys():
    try:
        doc = minidom.parse("BayzickBullyingData/xml packet 1/" + i + ".xml")
        name = doc.getElementsByTagName("body")
        data = list()
        for j in name:
            try:
                data.append(j.firstChild.data)
            except AttributeError:
                print("***************************object has no attribute 'data'*****************************")
        res = [''.join(data)]
        data1.append(res)
        lbls.append(files_to_labels[i])
    except IOError:
        print("######################No such file {}".format("BayzickBullyingData/xml packet 1/" + i + ".xml") +
              "##############################")
flat_list = [item for x in data1 for item in x]
chat_with_labels = dict(zip(flat_list, lbls))

print(chat_with_labels)
print(len(chat_with_labels))
