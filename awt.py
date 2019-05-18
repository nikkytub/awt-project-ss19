from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd
from xml.dom import minidom
import xml
import glob

print(np.__version__)
print(tf.__version__)

path = 'BayzickBullyingData/Human Concensus/'


# To extract the file names with labels
def extract_files_labels(file):
    df = pd.read_excel(file)
    files = np.array(df['Bully Tracer Consensus'], dtype='string')
    labels = np.array(df['Unnamed: 1'], dtype='string')
    files_to_labels = dict(zip(files[2:], labels[2:]))
    return files_to_labels


# To extract the chats with labels
result = [extract_files_labels(files) for files in glob.glob(path + "*")]
data1 = list()
lbls = list()
for i in result:
    for j in i.keys():
        try:
            doc = minidom.parse("BayzickBullyingData/packet-all/" + j + ".xml")
            name = doc.getElementsByTagName("body")
            data = list()
            for k in name:
                try:
                    data.append(k.firstChild.data)
                except AttributeError:
                    pass
                    #print("***************************object has no attribute 'data'*****************************")
            res = [''.join(data)]
            data1.append(res)
            lbls.append(i[j])
        except IOError:
            pass
            #print("######################No such file {}".format("BayzickBullyingData/packet-all/" + j + ".xml") +
            #      "##############################")
        except xml.parsers.expat.ExpatError:
            pass
            #print("Not well formed: {}".format(j + ".xml"))
flat_list = [item for x in data1 for item in x]
chat_with_labels = dict(zip(flat_list, lbls))
for k, v in chat_with_labels.items():
    print(k, v)
print(len(chat_with_labels))
