import os
import tensorflow as tf
import collections
import numpy as np
from loading import DataClass
from build_model import build_graph

url = '/home/marek/PycharmProjects/text_class/'
vocabulary_size = 10000
batch_size = 1

datka = DataClass(os.getcwd(), 'wiki_texts', batch_size, vocabulary_size, data_use="train")

skuska = build_graph('text1')
skuska.data_shape(tf.int32, (datka.maxlen,), sequences=True)
skuska.embeddings(vocabulary_size=vocabulary_size, embedding_size=16)
skuska.lstm(32)
skuska.relu()
skuska.lstm(16, output='last')
skuska.relu()
skuska.fc(10)
exec(skuska.finish())

