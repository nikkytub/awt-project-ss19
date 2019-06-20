import numpy as np
import tensorflow as tf

mes = ["wow you must have been in the homo bar. cuz they suppose that they are the best nationalty in the world   anyway mr gribble: is your car name Civic? " \
      "So you are judging a whole country based on some drunk idiots in a bar. Now that is stupid.  Actually you are really stupid for being a racist." \
      "It is stupid that your judge the whole country...   Hell I live in america and I moved to Texas from Mississippi and I got made fun of and asked to say stupid shit cause I sounded different so suck it up..." \
      "agree.plus it's not like you never make jokes about people that are from another country or look different!so don't complain!" \
      "you should hear how they talk here in Minesota. ya, sure ya btechya, don't ya know. its a lot like the sing sony sveds" \
      "i hate when people ask me to say things over and over again, but my accent is diff so i guess its new to them."]

mes1 = ["You bastard. I will kill you"]

avg_length = 325

# To pad shorter and truncate longer conversation
vocabulary_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(avg_length)

# To transform each conversation to numbers
num_conversation = np.array(list(vocabulary_processor.fit_transform(mes)))
num_conversation1 = np.array(list(vocabulary_processor.fit_transform(mes1)))

with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, ["serve"], "model")
    graph = tf.get_default_graph()
    m = sess.run('myOutput:0', feed_dict={'myInput:0': num_conversation})
    n = sess.run('myOutput:0', feed_dict={'myInput:0': num_conversation1})
    print(m)
    print(n)
