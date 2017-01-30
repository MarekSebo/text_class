import os
import tensorflow as tf
import numpy as np
from loading import DataClass
from build_model import build_graph, accuracy

url = os.getcwd()
vocabulary_size = 20000
batch_size = 16
learning_rate = 0.0001

data_train = DataClass(os.getcwd(), 'wiki_texts', batch_size, vocabulary_size, data_use="train")

model = build_graph('2lang_2')
model.data_shape(tf.int32, (None,), sequences=True)
model.embeddings(vocabulary_size=vocabulary_size, embedding_size=16)
model.lstm(128)
model.relu()
model.lstm(128, output='last')
model.relu()
model.fc(128)
model.relu()
model.fc(64)
model.relu()
model.fc(2)
model.finish(learning_rate)

step = 0
while step < 500:
    x, y, seqlen = data_train.next_batch()
    loss, predict, global_step = model.step(x, y, sequence_lengths=seqlen)
    step += 1
    if step % 250 == 0:
        print('step', global_step)
        print('loss', loss)
        print('real v predict', list(zip(np.argmax(y, axis=-1), np.argmax(predict, axis=-1))))
        print('accuracy', accuracy(predict, y))
model.save()

model.session.close()
