import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image as pilimg
from numpy import random
import tensorflow as tf
import collections

class DataClass(object):
    """
    POUZITIE
    -vytvor instanciu meno = DataClass(args)
    -vypytaj si novy batch: meno.next_batch()
    BACKEND
    -data sa nacitavaju v chunkoch zadanej velkosti
    -batche sa vzdy beru zaradom z aktualneho chunku
    -ked sa minie chunk (dojde sa na koniec), nacita sa novy chunk
    -ked sa minu chunky, premiesaju sa data a zacne sa znova
    """
    def __init__(self, url, path, batch_size, vocabulary_size, data_use="train"):
        self.data = None
        self.labels = None

        self.path = path
        self.vocabulary_size = vocabulary_size
        self.data_use = data_use

        _, self.dictionary, self.reverse_dictionary = DataClass.build_dictionaries(vocabulary_size, os.path.join(url, path), url)
        self.data, self.labels, _ = self.final_data(os.getcwd(), self.dictionary, url)

        self.total_data_size = len(self.labels)
        self.batch_size = batch_size
        self.batch_cursor = 0              # pozicia batchu vramci chunku

        self.maxlen = max([len(i) for i in self.data])

        self.languages = ['en', 'sk', 'de', 'fr', 'it', 'cz', 'pl', 'hr', 'nl']
        self.shuffle()

    def next_batch(self):
        data = self.data[self.batch_cursor:self.batch_cursor + self.batch_size]
        labels = self.labels[self.batch_cursor:self.batch_cursor + self.batch_size]
        labels = np.array([DataClass.onehot(label, self.languages) for label in labels])

        self.batch_cursor += self.batch_size
        if self.batch_cursor + self.batch_size > self.total_data_size:
            self.shuffle()
            self.batch_cursor = 0

        if len(labels) < self.batch_size:
            self.next_batch()

        seq_len = random.randint(1, 25, size=labels[:,0].shape)
        beginnings = [random.randint(0, len(d) - s) for d, s in zip(data, seq_len)]

        # seq_len = np.array([len(i) for i in data])
        maxout = np.max(seq_len)
        #toto treba riesit
        data = np.array(([np.pad(d[b : b + s], [(0, maxout - s)], mode='constant') for d,s,b in zip(data, seq_len, beginnings)]))

        return data, labels, seq_len

    def shuffle(self):
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)

        self.data[:], self.labels[:] = zip(*combined)

    @staticmethod
    def onehot(y, alphabet):
        result = np.zeros(len(alphabet))
        result[alphabet.index(y)] = 1
        return result

    @staticmethod
    def read_words(filename):
        with tf.gfile.GFile(filename, "r") as f:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()

    # read all texts in all dirs, record where the files and dirs end

    @staticmethod
    def read_all_texts(train_path, url):
        # output:
        #   words = all words from all texts as a list

        words = []

        # v train su iba priecinky, v kazdom su txt subory v jednej class
        label_names = os.listdir(train_path)
        label_names.sort()

        print("label_names =", label_names)
        word_count = 0

        for dir in label_names:
            os.chdir(os.path.join(train_path, dir))
            dir_files = os.listdir()
            dir_files.sort()
            print("files in directory {} : {}".format(dir, dir_files))
            for filename in dir_files:
                file_words = DataClass.read_words(filename)
                words.extend(file_words)
                word_count += len(file_words)
        os.chdir(os.path.join(url, train_path))  # change the dir back

        return words

    @staticmethod
    def build_dictionaries(vocabulary_size, path, url):
        # output:
        #   count = list of [word, count] for the words in vocabulary
        #   dictionary = <word>:<integer>
        #   reverse_dictionary = <integer>:<word>
        all_words = DataClass.read_all_texts(path, url)
        count = [['UNK', -1]]
        count.extend(collections.Counter(all_words).most_common(vocabulary_size - 1))
        # make a dict of [word: index]
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return count, dictionary, reverse_dictionary

    @staticmethod
    def words_to_integers(words, dictionary):
        # input:
        #   words = words in the text (as list)
        # output:
        #   words coded as integers, UNK = code 0
        #   unk_count = count of unknown words

        words_new = list()
        unk_count = 0

        for i, word in enumerate(words):

            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            words_new.append(index)
        return words_new, unk_count

    @staticmethod
    def final_data(train_path, dictionary, url):
        # v train su iba priecinky, v kazdom su txt subory v jednej class
        print("finaldata_begin dir: ", os.getcwd())
        label_names = os.listdir()
        label_names.sort()
        unk_count = 0  # count of unknown words
        data = []
        labels = []

        for dir in label_names:
            os.chdir(os.path.join(train_path, dir))  # enter the class dir
            dir_files = os.listdir()
            dir_files.sort()
            len_label = len(dir_files)  # count of files of the class
            labels += [str(dir)] * len_label

            for filename in dir_files:
                file_words = DataClass.read_words(filename)
                integers, unk_in_file = DataClass.words_to_integers(file_words, dictionary)
                unk_count += unk_in_file
                data.append(integers)

        os.chdir(os.path.join(url))  # change the dir back
        return data, labels, unk_count


def accuracy2(predictions, labels, printout = True):
    conf_matrix = np.zeros((2, 2))
    # true positive
    predictions = np.round(predictions).astype(bool)
    labels = np.array(labels).astype(bool)
    conf_matrix[0, 0] = np.sum(np.logical_and(predictions, labels))
    conf_matrix[1, 0] = np.sum(np.logical_and(predictions, np.logical_not(labels)))
    conf_matrix[0, 1] = np.sum(np.logical_and(np.logical_not(predictions), labels))
    conf_matrix[1, 1] = np.sum(np.logical_and(np.logical_not(predictions), np.logical_not(labels)))
    log = []
    hitrate = 'no real positives in sample'
    if sum(conf_matrix[0, :]):
        hitrate = conf_matrix[0, 0] / sum(conf_matrix[0, :]) * 100
    precision = 0
    f1_score = 'not available'
    if sum(conf_matrix[:, 0]):
        precision = conf_matrix[0, 0] / sum(conf_matrix[:, 0]) * 100
    log.append('Hit-rate (odhalene real positives) '
               + str(round(hitrate, ndigits = 2) ))
    log.append('Precision (uspesnost predicted positives) '
               + str(round(precision, ndigits=2) ))
    if precision and hitrate:
        f1_score = 2 / (1/precision + 1/hitrate)
        log.append('f1-score (harm. priemer precision a recall) '
               +str(round(f1_score, ndigits=2) ))
    log.append('Confusion matrix (real x predicted):\n'
               + str(conf_matrix))
    if printout:
        print('\n'.join(log))
    return log

def conv_output_size(padding, input_height, input_width, stride, kernel_height, kernel_width):
    output_height, output_width = (0, 0)
    if padding == "VALID":
        output_height = (input_height - kernel_height) / stride + 1
        output_width = (input_width - kernel_width) / stride + 1
        output_height = int(np.floor(output_height))
        output_width = int(np.floor(output_width))
    if padding == "SAME":
        output_height = input_height / stride
        output_width = input_width / stride
        output_height = int(np.ceil(output_height))
        output_width = int(np.ceil(output_width))
    return output_height, output_width

def draw_prediction(pred, real, h, w):

    def replace(tab, c, nc):
        (r, g, b) = c
        (nr, ng, nb) = nc
        tab[r] = nr
        tab[g + 256] = ng
        tab[b + 256 * 2] = nb
        return tab

    table = list(range(256)) * 3
    table = replace(table, [63] * 3, [255, 140, 0])
    table = replace(table, [191] * 3, [255, 0, 0])

    pred = pred.reshape([-1, h, w])
    real = real.reshape([-1, h, w])

    for i in range(pred.shape[0]):
        img = pilimg.fromarray(255 * pred[i])
        img = img.convert('RGB')
        img.save('predictions/im{}_B_pred.png'.format(i))

        img = pilimg.fromarray(255 * real[i])
        img = img.convert('RGB')
        img.save('predictions/im{}_A_real.png'.format(i))

        p = np.round(pred[i])
        suma = ((3*real[i] + p) * 255) // 4  # 0, 63, 191, 255
        img = pilimg.fromarray(suma)
        img = img.convert('RGB')
        #replace siva - cervena
        # seda - oranzova
        img = img.point(table)
        img.save('predictions/im{}_C_combo.png'.format(i))

