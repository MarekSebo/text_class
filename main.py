import os
import tensorflow as tf
import collections
import numpy as np
from loading import DataClass
from build_model import build_graph, accuracy

url = os.getcwd()
vocabulary_size = 20000
batch_size = 16

datka = DataClass(os.getcwd(), 'wiki_texts', batch_size, vocabulary_size, data_use="train")

skuska = build_graph('2lang_1')
skuska.data_shape(tf.int32, (None,), sequences=True)
skuska.embeddings(vocabulary_size=vocabulary_size, embedding_size=16)
skuska.lstm(128)
skuska.relu()
skuska.lstm(128, output='last')
skuska.relu()
skuska.fc(128)
skuska.relu()
skuska.fc(64)
skuska.relu()
skuska.fc(2)
skuska.finish()

step = 0
while step < 5000:
    x, y, seqlen = datka.next_batch()
    loss, predict, global_step = skuska.step(x, y, sequence_lengths=seqlen)
    step += 1
    if step % 250 == 0:
        print('step', global_step)
        print('loss', loss)
        print('real v predict', list(zip(np.argmax(y, axis=-1), np.argmax(predict, axis=-1))))
        print('accuracy', accuracy(predict, y))
skuska.save()

skuska.session.close()
