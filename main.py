import os
import tensorflow as tf
import collections
import numpy as np
from loading import DataClass
from build_model import build_graph, accuracy

url = os.getcwd()
vocabulary_size = 10000
batch_size = 16

datka = DataClass(os.getcwd(), 'wiki_texts', batch_size, vocabulary_size, data_use="train")

skuska = build_graph('text4')
skuska.data_shape(tf.int32, (None,), sequences=True)
skuska.embeddings(vocabulary_size=vocabulary_size, embedding_size=16)
skuska.lstm(64)
skuska.relu()
skuska.lstm(32, output='last')
skuska.relu()
skuska.fc(16)
skuska.relu()
skuska.fc(9)
skuska.finish()

step = 0
while step < 1000:
    x, y, seqlen = datka.next_batch()
    loss, predict, global_step = skuska.step(x, y, sequence_lengths=seqlen)
    step += 1
    if step % 1 == 0:
        print('step', global_step)
        print('loss', loss)
        print('real v predict', list(zip(np.argmax(y, axis=-1), np.argmax(predict, axis=-1))))
        print('accuracy', accuracy(predict, y))
skuska.save()

skuska.session.close()
