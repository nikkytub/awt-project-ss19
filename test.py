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
labels = list()
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
            res = [''.join(data)]
            data1.append(res)
            labels.append(i[j])
        except IOError:
            pass
        except xml.parsers.expat.ExpatError:
            pass
conversation = [item for x in data1 for item in x]
chat_with_labels = dict(zip(conversation, labels))

# To convert labels from Yes, No to 1, 0
for n, i in enumerate(labels):
    if i == 'N':
        labels[n] = 0
    elif i == 'Y':
        labels[n] = 1

# To find an average length of all the conversations
average_conv_len = np.mean([len(x.split(" ")) for x in conversation])
#print(average_conv_len)

avg_length = 325

# To pad shorter and truncate longer conversation
vocabulary_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(avg_length)

# To transform each conversation to numbers
num_conversation = np.array(list(vocabulary_processor.fit_transform(conversation)))
labels_array = np.array(labels)

# To shuffle data
shuffle = np.random.permutation(np.arange(len(num_conversation)))

shuffled_conv = num_conversation[shuffle]
shuffled_labels = labels_array[shuffle]

training_size = 1576
total_size = 1582

train_conv = shuffled_conv[:training_size]
train_labels = shuffled_labels[:training_size]

test_conv = shuffled_conv[training_size:total_size]
test_labels = shuffled_labels[training_size:total_size]



sess=tf.Session()
signature_key = tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
input_key = 'x_input'
output_key = 'y_output'

export_path = './model_rnn'
meta_graph_def = tf.saved_model.loader.load(
           sess,
          ["myTag"],
          export_path)
signature = meta_graph_def.signature_def

x_tensor_name = signature[signature_key].inputs[input_key].name
y_tensor_name = signature[signature_key].outputs[output_key].name

x = sess.graph.get_tensor_by_name(x_tensor_name)
y = sess.graph.get_tensor_by_name(y_tensor_name)

y_out = sess.run(y, {x: test_conv})
print(y_out, test_conv)
print("Actual labels: ", labels[training_size:total_size])
for i in conversation[training_size:total_size]:
    print(i)
