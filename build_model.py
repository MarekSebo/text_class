import tensorflow as tf
from tensorflow.python import framework
import numpy as np
import logging
import os


class BuildGraph(object):
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
        # define conv layer (init weights + graph)
        # inputs:
        #   filter = kernel size
        #   padding = {'SAME', 'VALID'}

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
        # build one LSTM layer
        # input:
        #   output = {'all' , 'last'}
        with tf.variable_scope('lstm_{}'.format(self.lstms)):
            self.c_state = tf.placeholder(dtype=tf.float32, shape=(None, lstm_size))
            self.h_state = tf.placeholder(dtype=tf.float32, shape=(None, lstm_size))
            tf.add_to_collection('states', self.c_state)
            tf.add_to_collection('states', self.h_state)
            initial_state = tf.nn.rnn_cell.LSTMStateTuple(self.c_state, self.h_state)

            self.lstm_cells.append(tf.nn.rnn_cell.BasicLSTMCell(lstm_size))

            self.out, self.lstm_state = tf.nn.dynamic_rnn(
                cell=self.lstm_cells[-1], inputs=self.out, dtype=tf.float32, initial_state=initial_state)

            if output == 'last':
                self.out = self.out[:, -1, :]

            tf.add_to_collection('lstm_state', self.lstm_state[0])
            tf.add_to_collection('lstm_state', self.lstm_state[1])


        self.current_shape = self.out.get_shape().as_list()
        self.lstms += 1

    def relu(self):
        self.out = tf.nn.relu(self.out)

    def loss(self):
        # cross entropy
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.out, self.one_hots))

    def embeddings(self, vocabulary_size, embedding_size, trainable=True):
        # same inputs as in TF
        self.vocabulary_size = vocabulary_size
        self.embeddings = tf.get_variable('embedding',
                                          shape=[self.vocabulary_size, embedding_size],
                                          initializer=tf.random_uniform_initializer(minval=0, maxval=None, seed=None, dtype=tf.float32),
                                          trainable = trainable)
        self.out = tf.nn.embedding_lookup(self.embeddings, tf.cast(self.out, tf.int32))
        self.current_shape = self.out.get_shape().as_list()

    def optimizer(self, learning_rate):
        return tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08)\
            .minimize(self.loss_op, global_step=self.global_step)

    def step(self, x, y, sequence_lengths = None, lstm_state = None):
        # make one step
        feed={self.input: x, self.labels: y}

        if sequence_lengths is not None:
            feed[self.seq_len] = sequence_lengths

        if lstm_state is not None:
            feed[self.c_state], feed[self.h_state] = lstm_state

        loss, _, predict, summary, step, lstm_state = self.session.run(
            [self.loss_op, self.optimizer_op, self.prediction, self.tb_loss_train, self.global_step, self.lstm_state], feed_dict=feed)

        # self.steps += 1
        self.file_writer.add_summary(summary, global_step=step)
        return loss, predict, step, lstm_state


    def save(self):
        self.saver.save(self.session, '{}/logs/{}/{}'.format(os.getcwd(), self.session_name, self.session_name),
                        global_step=self.global_step)
        self.saver.export_meta_graph('{}/logs/{}/{}.meta'.format(os.getcwd(), self.session_name, self.session_name))

    def load(self, dir_ckpt):
        self.saver = tf.train.import_meta_graph('{}/logs/{}/{}.meta'.format(os.getcwd(), self.session_name, self.session_name))
        self.saver.restore(self.session, tf.train.latest_checkpoint(dir_ckpt))#nevie to nacitat lebo je v ckpt aj cislo kroku a tu nie

    def predict(self, x, sequence_lengths=None, lstm_state=None):
        feed = {self.input: x}

        if sequence_lengths is not None:
            feed[self.seq_len] = sequence_lengths

        if lstm_state is not None:
            feed[self.c_state], feed[self.h_state] = lstm_state

        return self.session.run([self.prediction, self.lstm_state], feed_dict=feed)

    def data_shape(self, type, shape, sequences=False):
        # run this before the first layer
        # inputs:
        #   shape = shape bez batch size dimenzie
        #   type = typ premennej , napr tf.float32
        #   sequences = bool, if I want to remember the lengths of the sequences in data (e.g in texts with variable length)
        self.current_shape = shape
        self.input = tf.placeholder(type, shape=(None, ) + shape)
        self.global_step = tf.Variable(0, name='global_step')
        self.out = self.input

        if sequences:
            self.seq_len = tf.placeholder(tf.int32, shape=[None])


    def finish(self, learning_rate):
        # defines loss etc....
        self.prediction = tf.nn.softmax(self.out)
        self.labels = tf.placeholder(tf.int64, shape=(None,), name='labels')
        self.one_hots = tf.one_hot(self.labels, self.vocabulary_size)

        self.loss_op = self.loss()
        self.optimizer_op = self.optimizer(learning_rate)

        self.tb_loss_train = tf.summary.scalar('loss_function_batch', self.loss_op)
        self.tb_loss_valid = tf.summary.scalar('loss_function_valid', self.loss_op)

        self.file_writer = tf.summary.FileWriter('logs/{}'.format(self.session_name))

        tf.add_to_collection('output', self.out)
        tf.add_to_collection('intput', self.input)
        tf.add_to_collection('labels', self.labels)
        tf.add_to_collection('prediction', self.prediction)
        tf.add_to_collection('others', self.loss_op)
        tf.add_to_collection('others', self.optimizer_op) #make it so that you can insert LR
        tf.add_to_collection('others', self.tb_loss_train)
        tf.add_to_collection('others', self.global_step)

        self.potom()

    def potom(self):
        self.session = tf.InteractiveSession()

        dir_ckpt = 'logs/{}'.format(self.session_name)
        if tf.train.latest_checkpoint(dir_ckpt) is not None:
            self.load(dir_ckpt)
        else:
            logging.exception('File not found, initializing new model')
            self.session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver()
            return

        self.out = tf.get_collection('output')[0]
        self.input = tf.get_collection('intput')[0]
        self.labels = tf.get_collection('labels')[0]
        self.prediction = tf.get_collection('prediction')[0]

        self.c_state, self.h_state= tf.get_collection('states')
        self.lstm_state = tf.nn.rnn_cell.LSTMStateTuple(tf.get_collection('lstm_state')[0],
                                                        tf.get_collection('lstm_state')[1])
        print(self.c_state, self.lstm_state)
        self.loss_op, self.optimizer_op, self.tb_loss_train, self.global_step = tf.get_collection('others')
        self.file_writer = tf.summary.FileWriter('logs/{}'.format(self.session_name))


def accuracy(predictions, labels):
    acc = np.sum([(labels[i, np.argmax(predictions[i, :])] == 1) for i in range(predictions.shape[0])]) \
          / predictions.shape[0]
    return acc

def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized
    probabilities.
    Exclude first element of distribution (UNK)
    """
    r = np.random.uniform(0, 1 - distribution[0])
    s = 0
    for i, prob in enumerate(distribution[1:]):
        s += prob
        if s >= r:
            return i + 1
    return len(distribution) - 1
