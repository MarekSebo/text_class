import tensorflow as tf
import numpy as np
import logging

class build_graph(object):
    def __init__(self):
        self.current_shape = None
        self.w = 0
        self.weights = {}
        self.bias = {}
        self.lstm_cells = []
        self.lstms = 0

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
        print(print('after reshape:', self.current_shape))

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

        self.current_shape = self.out.get_shape()
        self.w += 1

    def lstm(self, lstm_size):
        self.lstm_cells.append(tf.nn.rnn_cell.BasicLSTMCell(lstm_size))

        print(self.out.get_shape())
        self.seq_len = 10 * tf.ones(5)

        self.out, self.state_lstm = tf.nn.dynamic_rnn(
            cell=self.lstm_cells[-1], inputs=self.out, dtype=tf.float32, sequence_length=self.seq_len)

        self.current_shape = self.out.get_shape()
        self.lstms += 1

    def relu(self):
        self.out = tf.nn.relu(self.out)

    def loss(self):
        return tf.reduce_mean(tf.square(self.out - self.labels))

    def optimizer(self):
        return tf.train.AdamOptimizer(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-08)\
            .minimize(self.loss_op)

    def step(self, x, y):
        loss, _, predict = self.session.run(
            [self.loss_op, self.optimizer_op, self.prediction], feed_dict={self.input: x, self.labels: y})
        return loss, predict

    def predict(self, x):
        return self.session.run(self.prediction, feed_dict={self.input: x})

    def data_shape(self, type, shape):
        self.current_shape = shape
        self.input = tf.placeholder(type, shape=(None, ) + shape)
        self.out = self.input

    def finish(self):
        self.prediction = self.out
        self.labels = tf.placeholder(self.out.dtype, self.out.get_shape())

        self.loss_op = self.loss()
        self.optimizer_op = self.optimizer()

        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())




