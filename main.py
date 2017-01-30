import os
import tensorflow as tf
import numpy as np
from loading import DataClass
from build_model import BuildGraph, sample_distribution
import logging

if __name__ == '__main__':
    name = 'kapitan27'
    batch_size = 1
    learning_rate = 0.0001
    lstm_size = 128
    vocabulary_size = 18000

    logging.basicConfig(level=20)
    url = os.getcwd()

    data_train = DataClass(url, 'wiki_texts', batch_size, vocabulary_size, data_use="train")

    model = BuildGraph(name)
    # model.data_shape(tf.int32, (1,))
    # model.embeddings(vocabulary_size=vocabulary_size, embedding_size=16)
    # model.lstm(lstm_size)
    # model.relu()
    # model.fc(64)
    # model.relu()
    # model.fc(16)
    # model.relu()
    # model.fc(vocabulary_size)
    # model.finish(learning_rate)
    model.potom()

    step = 0
    while step < 2:
        x, y = data_train.next_batch()
        lstm_state = np.zeros((batch_size, lstm_size))
        lstm_state =[lstm_state, lstm_state]
        for x_, y_ in zip(x.T, y.T):
            loss, predict, global_step, lstm_state = model.step(x_.reshape(-1, 1), y_, lstm_state=lstm_state)
        step += 1
        if step % 1 == 0:
            logging.info('real next word: {}'.format(data_train.reverse_dictionary[y_[0]]))
            logging.info('predicted: {}'.format(data_train.reverse_dictionary[sample_distribution(predict[0])]))
            logging.info('global step {}'.format(global_step))
            logging.info('---------------')
    model.save()

    step = 0
    generated_text = []
    lstm_state = np.zeros((1, lstm_size))
    lstm_state = (lstm_state, lstm_state)

    x = np.array([1])
    x = x.reshape(-1, 1)
    while step < 50:
        x, lstm_state = model.predict(x, lstm_state=lstm_state)
        x = sample_distribution(x[0])
        # x = np.argmax(x[0][1:]) + 1
        generated_text.append(data_train.reverse_dictionary[x])
        x = np.array([x]).reshape(-1, 1)
        step += 1
    print(' '.join(generated_text))

    model.session.close()
