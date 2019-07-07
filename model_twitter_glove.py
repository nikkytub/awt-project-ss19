from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import tensorflow as tf
import pandas as pd


data = pd.read_csv('data.csv', parse_dates=True)
data1 = pd.read_csv('tweets.csv', parse_dates=True)

tweet_id = data['Tweet ID'].to_list()
cb = data[' Bullying_Traces?'].to_list()
tweetid_cb = dict(zip(tweet_id, cb))

t_id = data1['Tweet ID'].to_list()
tw = data1['Tweets'].to_list()

tweetid_tweets = dict(zip(t_id, tw))
tweets_labels = dict()
for i in t_id:
    try:
        tweets_labels.update({tweetid_tweets[i]: tweetid_cb[i]})
    except KeyError:
        pass

conversation = list(tweets_labels.keys())
labels = list(tweets_labels.values())

# To convert labels from Yes, No to 1, 0
for n, i in enumerate(labels):
    if i == 'n':
        labels[n] = 0
    elif i == 'y':
        labels[n] = 1

# To find an average length of all the conversations
average_conv_len = np.mean([len(x.split(" ")) for x in conversation])

avg_length = 15

# To pad shorter and truncate longer conversation
vocabulary_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(avg_length)

# To transform each conversation to numbers
num_conversation = np.array(list(vocabulary_processor.fit_transform(conversation)))
labels_array = np.array(labels)

# To shuffle data
shuffle = np.random.permutation(np.arange(len(num_conversation)))

shuffled_conv = num_conversation[shuffle]
shuffled_labels = labels_array[shuffle]

training_size = 3900
total_size = 4304

train_conv = shuffled_conv[:training_size]
train_labels = shuffled_labels[:training_size]

test_conv = shuffled_conv[training_size:total_size]
test_labels = shuffled_labels[training_size:total_size]

tf.reset_default_graph()

a = tf.placeholder(tf.int32, [None, avg_length])
b = tf.placeholder(tf.int32, [None])

batch_size = 5
epochs = 30
embedding = 50
label_max = 2

emb = np.load('wordVectors.npy')

embed = tf.nn.embedding_lookup(emb, a)

# To instantiate LSTM cells
lstm = tf.contrib.rnn.BasicLSTMCell(embedding)

# To prevent overfitting
lstm = tf.contrib.rnn.DropoutWrapper(cell=lstm, output_keep_prob=0.80)

output, (final_state, state_info) = tf.nn.dynamic_rnn(lstm, embed, dtype=tf.float32)

logits = tf.layers.dense(final_state, label_max, activation=None)

c_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=b)

loss = tf.reduce_mean(c_entropy)

forecast = tf.equal(tf.argmax(logits, 1), tf.cast(b, tf.int64))

accuracy = tf.reduce_mean(tf.cast(forecast, tf.float32))

pred = tf.argmax(logits, 1)

# To use Adam optimizer to minimize the loss
adam_opt = tf.train.AdamOptimizer(0.01)
training_steps = adam_opt.minimize(loss)


init = tf.global_variables_initializer()

builder = tf.saved_model.builder.SavedModelBuilder("model_rnn_twitter_glove")

with tf.Session() as sess:
    init.run()
    for epoch in range(epochs):
        batches = int(len(train_conv) // batch_size) + 1
        for i in range(batches):
            minimum = i * batch_size
            maximum = np.min([len(train_conv), ((i+1) * batch_size)])

            x_train = train_conv[minimum:maximum]
            y_train = train_labels[minimum:maximum]

            training = {a: x_train, b: y_train}
            sess.run(training_steps, feed_dict=training)
            predic_train = sess.run([pred], {a: x_train})

            training_loss, training_accuracy = sess.run([loss, accuracy], feed_dict=training)

        testing = {a: test_conv, b: test_labels}
        test_loss, test_accuracy = sess.run([loss, accuracy], feed_dict=testing)
        print("Epoch--> {}, Test Accuracy--> {:.3}". format(epoch + 1, test_accuracy))
        predic_test = sess.run([pred], {a: test_conv})

    tensor_info_x = tf.saved_model.utils.build_tensor_info(a)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(pred)
    prediction_signature = (
        tf.saved_model.signature_def_utils.build_signature_def(
            inputs={'x_input': tensor_info_x},
            outputs={'y_output': tensor_info_y},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
    builder.add_meta_graph_and_variables(sess,
                                         ["myTag"],
                                         signature_def_map={
                                             tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                                                 prediction_signature})
builder.save()
