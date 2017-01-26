import os
import tensorflow as tf
import numpy as np
from loading import DataClass
from build_model import build_graph, accuracy, sample_distribution

url = os.getcwd()
vocabulary_size = 18000
batch_size = 2
learning_rate = 0.0001
lstm_size = 3

datka = DataClass(os.getcwd(), 'wiki_texts', batch_size, vocabulary_size, data_use="train")

skuska = build_graph('general3')
skuska.data_shape(tf.int32, (1,))
skuska.embeddings(vocabulary_size=vocabulary_size, embedding_size=16)
skuska.lstm(lstm_size)
skuska.relu()
skuska.fc(64)
skuska.relu()
skuska.fc(16)
skuska.relu()
skuska.fc(vocabulary_size)
skuska.finish(learning_rate)

step = 0
while step < 0:
    x, y = datka.next_batch()
    lstm_state = np.zeros((batch_size, lstm_size))
    lstm_state = zip(lstm_state, lstm_state)
    for x_, y_ in zip(x.T, y.T):
        loss, predict, global_step, lstm_state = skuska.step(x_.reshape(-1, 1), y_, lstm_state=lstm_state)
    step += 1
    if step % 1 == 0:
        print('real next word', datka.reverse_dictionary[y_[0]])
        print('predicted', datka.reverse_dictionary[sample_distribution(predict[0])])
        print('step', global_step)
        print('---------------')
        # print('loss', loss)
        # print('real v predict', list(zip(np.argmax(y, axis=-1), np.argmax(predict, axis=-1))))
skuska.save()

lstm_state = np.zeros((1, lstm_size))
lstm_state = zip(lstm_state, lstm_state)
x = np.array([1, 1])
x = x.reshape(-1, 1)
while step<50:
    x, lstm_state = skuska.predict(x, lstm_state=lstm_state)
    print(datka.reverse_dictionary[sample_distribution(x[0])])

skuska.session.close()

