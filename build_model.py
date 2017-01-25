import tensorflow as tf
from tensorflow.python import framework
import numpy as np
import logging
import os

class build_graph(object):
    def __init__(self, name):
        self.current_shape = None
        self.w = 0
        self.weights = {}
        self.bias = {}
        self.lstm_cells = []
        self.lstms = 0
        self.session_name = name
        self.learning_rate = 0.0001

    def conv2d(self, filter, num_filters, strides, padding):
        self.weights['conv_weights_{}'.format(self.w)] = tf.get_variable(
                            'conv_weights_{}'.format(self.convs),
                            shape=(filter[0], filter[1], self.current_shape[-1], num_filters),
                            initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False, seed=None, dtype=tf.float32)
                            )

        self.bias['conv_bias_{}'.format(self.w)] = tf.get_variable(
            'conv_bias_{}'.format(self.w),
            shape=num_filters,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        )

        self.out = tf.nn.conv2d(self.out,
                                filter=self.weights['conv_weights_{}'.format(self.w)],
                                strides=strides,
                                padding=padding,
                                name='conv_layer_{}'.format(self.w)
                                )
        self.out = self.out + self.bias['conv_bias_{}'.format(self.w)]

        self.current_shape = self.out.get_shape()
        self.w += 1

    def fc(self, layer_size):
        self.current_shape = self.out.get_shape().as_list()
        self.out = tf.reshape(self.out, [-1, np.prod(self.current_shape[1:])])
        self.current_shape = self.out.get_shape()

        self.weights['fc_weights_{}'.format(self.w)] = tf.get_variable(
            'fc_weights_{}'.format(self.w),
            shape=(self.current_shape[-1], layer_size),
            initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=None, dtype=tf.float32)
        )

        self.bias['fc_bias_{}'.format(self.w)] = tf.get_variable(
            'conv_bias_{}'.format(self.w),
            shape=layer_size,
            initializer=tf.contrib.layers.xavier_initializer(uniform=True, seed=None, dtype=tf.float32)
        )

        self.out = tf.matmul(self.out, self.weights['fc_weights_{}'.format(self.w)])
        self.out = tf.add(self.out, self.bias['fc_bias_{}'.format(self.w)])

        self.current_shape = self.out.get_shape()
        self.w += 1

    def lstm(self, lstm_size, output = 'all'):
        with tf.variable_scope('lstm_{}'.format(self.lstms)):
            self.lstm_cells.append(tf.nn.rnn_cell.BasicLSTMCell(lstm_size))

            self.out, self.state_lstm = tf.nn.dynamic_rnn(
                cell=self.lstm_cells[-1], inputs=self.out, dtype=tf.float32, sequence_length=self.seq_len)

            if output == 'last':
                print('lstm out', self.out.get_shape().as_list())
                self.out = self.out[:,-1]
                print('lstm out', self.out.get_shape().as_list())

        self.current_shape = self.out.get_shape().as_list()
        self.lstms += 1

    def relu(self):
        self.out = tf.nn.relu(self.out)

    def loss(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.out, self.labels))

    def embeddings(self, vocabulary_size, embedding_size, trainable=True):
        self.embeddings = tf.get_variable('embedding',
                                          shape=[vocabulary_size, embedding_size],
                                          initializer=tf.random_uniform_initializer(minval=0, maxval=None, seed=None, dtype=tf.float32),
                                          trainable = trainable)
        self.out = tf.nn.embedding_lookup(self.embeddings, tf.cast(self.out, tf.int32))
        self.current_shape = self.out.get_shape().as_list()

    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)\
            .minimize(self.loss_op, global_step=self.global_step)

    def step(self, x, y, sequence_lengths = None):
        feed={self.input: x, self.labels: y}

        if sequence_lengths is not None:
            feed[self.seq_len] = sequence_lengths

        loss, _, predict, summary, step = self.session.run(
            [self.loss_op, self.optimizer_op, self.prediction, self.tb_loss_train, self.global_step], feed_dict=feed)

        # self.steps += 1
        self.file_writer.add_summary(summary, global_step=step)

        return loss, predict, step

    def save(self):
        self.saver.save(self.session, "{}/logs/{}/{}.ckpt".format(os.getcwd(), self.session_name, self.session_name),
                        global_step=self.global_step)

    def predict(self, x, sequence_lengths = None):
        feed = {self.input: x}

        if sequence_lengths:
            feed[self.seq_len] = sequence_lengths

        return self.session.run(self.prediction, feed_dict=feed)

    def data_shape(self, type, shape, sequences=False):
        self.current_shape = shape
        self.input = tf.placeholder(type, shape=(None, ) + shape)
        self.global_step = tf.Variable(0, name='global_step')
        self.out = self.input

        if sequences:
            self.seq_len = tf.placeholder(tf.int32, shape=[None])

    def finish(self):
        self.prediction = tf.nn.softmax(self.out)
        self.labels = tf.placeholder(self.out.dtype, self.out.get_shape())

        self.loss_op = self.loss()
        self.optimizer_op = self.optimizer()

        self.tb_loss_train = tf.summary.scalar('loss_function_batch', self.loss_op)
        self.tb_loss_valid = tf.summary.scalar('loss_function_valid', self.loss_op)
        self.saver = tf.train.Saver()
        self.file_writer = tf.summary.FileWriter('logs/{}'.format(self.session_name))

        self.session = tf.InteractiveSession()

        dir_ckpt = 'logs/{}'.format(self.session_name)
        if tf.train.latest_checkpoint(dir_ckpt) is not None:
            self.saver.restore(self.session, tf.train.latest_checkpoint(dir_ckpt))
        else:
            logging.exception('File not found, initializing new model')
            self.session.run(tf.global_variables_initializer())

def accuracy(predictions, labels):
    acc = np.sum([(labels[i, np.argmax(predictions[i, :])] == 1) for i in range(predictions.shape[0])]) \
          / predictions.shape[0]
    return acc