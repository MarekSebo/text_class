import numpy as np
import os
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

        self.put_into_batches(50)
        self.generating_text_data_format()

        print([self.reverse_dictionary[i] for i in range(200)])

        self.total_data_size = len(self.labels)
        self.batch_size = batch_size
        self.batch_cursor = 0              # pozicia batchu vramci chunku

        self.maxlen = max([len(i) for i in self.data])

        self.languages = ['sk']
        self.shuffle()

    def next_batch(self):
        data = self.data[self.batch_cursor:self.batch_cursor + self.batch_size]
        labels = self.labels[self.batch_cursor:self.batch_cursor + self.batch_size]

        self.batch_cursor += self.batch_size
        if self.batch_cursor + self.batch_size > self.total_data_size:
            self.shuffle()
            self.batch_cursor = 0

        if len(labels) < self.batch_size:
            self.next_batch()

        return data, labels

    def shuffle(self):
        combined = list(zip(self.data, self.labels))
        random.shuffle(combined)

        self.data[:], self.labels[:] = zip(*combined)

    def put_into_batches(self, n):
        data = []
        for article in self.data:
            data += article[:DataClass.max_len_divisible_n(len(article), n)]
        data = np.array(data).reshape(-1, n)
        self.data = data

    def generating_text_data_format(self):
        self.labels = self.data[:, 1:]
        self.data = self.data[:, :-1]

    @staticmethod
    def max_len_divisible_n(k, n):
        return (k // n) * n

    @staticmethod
    def onehot(y, alphabet):
        result = np.zeros(len(alphabet))
        result[alphabet.index(y)] = 1
        return result

    @staticmethod
    def read_words(filename):
        with tf.gfile.GFile(filename, "r") as f:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()

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
        #   word_count = total pocet slov v texte

        words_new = list()
        unk_count = 0

        for i, word in enumerate(words):

            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            words_new.append(index)
            total_count = len(words_new)
        return words_new, unk_count, total_count

    @staticmethod
    def final_data(train_path, dictionary, url):
        # v train su iba priecinky, v kazdom su txt subory v jednej class
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
            unk_count = 0
            total_count = 0
            for filename in dir_files:
                file_words = DataClass.read_words(filename)
                integers, unk_in_file, total_count_in_file = DataClass.words_to_integers(file_words, dictionary)
                unk_count += unk_in_file
                total_count += total_count_in_file
                data.append(integers)
            print("Total words in the language in the training set: ", total_count)
            print( "UNK in language {} : {} %".format(dir, unk_count / total_count * 100 ))
        os.chdir(os.path.join(url))  # change the dir back
        return data, labels, unk_count
