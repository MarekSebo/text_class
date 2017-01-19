import numpy as np
import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from PIL import Image as pilimg
from numpy import random
import tensorflow as tf
import collections


# najprv premenuj nazvy podpriecinkov na 1TestingSet a 1TrainingSet


# -------------------------------------------------------
# cesta k obrazkom
url = os.getcwd()
# url = '/home/katarina/PycharmProjects/Ostruvky/data/'


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
    def __init__(self, path, batch_size, chunk_size, vocabulary_size, data_use="train"):
        self.data = None
        self.labels = None

        self.path = path
        self.vocabulary_size = vocabulary_size
        self.data_use = data_use

        self.file_names = self.load_filenames()

        self.total_data_size = len(self.file_names)

        self.batch_size = batch_size
        self.batch_cursor = 0              # pozicia batchu vramci chunku

        self.chunk_size = chunk_size        # (chunk_size // batch_size) * batch_size
        self.chunk_cursor = 0           # pozicia chunku vramci datasetu

        self.next_chunk()

    # read txt file as list of words
    def read_words(self, filename):
        with tf.gfile.GFile(filename, "r") as f:
            return f.read().decode("utf-8").replace("\n", "<eos>").split()

    # read all texts in all dirs, record where the files and dirs end
    def read_all_texts(self, train_path):
        # TODO vytvor jeden dlhy text, pouzi load_filenames

        # output:
        #   all words from all texts as a list
        #   dictionaries {file_name: split} {label_name: split}
        words = []
        file_names_list = []
        splits_files_list = []
        splits_labels_list = []

        # v train su iba priecinky, v kazdom su txt subory v jednej class
        label_names = os.listdir(train_path)
        print("label_names =", label_names)
        word_count = 0

        for dir in label_names:
            splits_labels_list.extend(word_count)
            with os.path.join(train_path, dir) as path:
                dir_files = os.listdir(path)
                file_names_list.extend(dir_files)
                for filename in dir_files:
                    splits_files_list.extend(word_count)
                    file_words = read_words(filename)
                    words.extend(file_words)
                    word_count += len(file_words)

        '''
        # !!!!! MOCK
        words = ["prve", "slovo", "druhe", "slovo", "tretie", "slovo", "prve", "slovo", "znova"]
        splits_labels_list = [0, 4] # aj nula tam bude
        splits_files_list = [0, 2, 4, 7]
        label_names = ['svahilsky', 'madarsky']
        file_names_list = ['a', 'b', 'c', 'd']
        '''
        splits_labels = dict(zip(label_names, splits_labels_list))
        splits_files = dict(zip(file_names_list, splits_files_list))

        return words, splits_labels, splits_files


    def build_dictionaries(self, words):
        # words = the text as a list of words

        # produce a list of [word, count] for the most common words
        count = [['UNK', -1]]
        count.extend(collections.Counter(words).most_common(self.vocabulary_size - 1))
        # make a dict of [word: index]
        dictionary = dict()
        for word, _ in count:
            dictionary[word] = len(dictionary)
        reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
        return count, dictionary, reverse_dictionary


    def dataset_to_integers(words, dictionary, count):
        # input:
            # words = all words in all texts
        # output:
            # words coded as integers, UNK = code 0
            # count = counts, including the count of unknown words

        words_new = list()
        unk_count = 0

        for i, word in words, range(len(words)):
            if word in dictionary:
                index = dictionary[word]
            else:
                index = 0  # dictionary['UNK']
                unk_count += 1
            words_new.append(index)

        count_new = count
        count_new[0][1] = unk_count

        return words_new, count_new

    def integers_to_embeddings(self, integers):
        """MOCK"""
        embeddings = integers

        return embeddings


    def build_dataset(self, words_new, splits_labels, splits_files):
        # words_new = dataset as embeddings
        # splits_labels, splits_files = dictionaries {name: split index}

        # outputs:
            # data = list of texts (lists of indices).  dims = [text_nr, word_nr]
            # labels = list
        data = []
        labels = []
        for label_name in splits_labels.keys:
            labels.extend(label_name)
            for file_name in splits_files.keys:
                data.extend([words_new[] ])


    all_texts, splits_class, splits_files = train.read_all_texts()
    count, dictionary, reverse_dictionary = build_dictionaries(all_texts)
    #
    data = dataset_to_integers(all_texts, dictionary)



    def shuffle(self):
        if self.data_use == "train":
            random.shuffle(self.file_names)

    def load_chunk(self):

        chunk_imgs = []
        chunk_labels = []

        for img, lab in self.file_names[self.chunk_cursor:self.chunk_cursor + self.chunk_size]:
            im1 = pilimg.open(os.path.join(self.path, img))
            im1 = np.array(im1).astype(float) / 255

            im2 = pilimg.open(os.path.join(self.path, lab))
            im2_pil = im2
            im2 = im2.convert('L')
            im2 = np.array(im2).astype(float) / 255

            if self.data_use == 'train':
                im1, im2 = random_flip(im1, im2)

            if self.augm != 0:
                coin = random.randint(0, 1)

                if coin == 1:
                    seed_num = random.randint(1, 999)
                    shear_ran = 0.7
                    num_out_im = self.augm
                    im1, im2 = augment(im1, im2_pil, shear_ran, seed_num, num_out_im)
                    im2 = np.reshape(im2, [1536, 2048, 3])
                    # chunk_imgs.append(im_new)
                    # chunk_labels.append(lab_new)

            chunk_imgs.append(im1)
            chunk_labels.append(im2)

        self.chunk_cursor = (self.chunk_cursor + self.chunk_size)

        self.current_chunk_size = len(chunk_imgs)

        # docasne riesenie
        if self.chunk_cursor + self.chunk_size > self.total_data_size:
            # print('last chunk of the epoch')
            self.chunk_cursor = 0
            self.shuffle()

        return np.array(chunk_imgs), np.array(chunk_labels)

    def next_chunk(self):
        # print('Getting new chunk')
        self.data, self.labels = self.load_chunk()
         # print('Got it')

    def next_batch(self):
        data = self.data[self.batch_cursor:self.batch_cursor + self.batch_size]
        labels = self.labels[self.batch_cursor:self.batch_cursor + self.batch_size]

        self.batch_cursor += self.batch_size
        if self.batch_cursor + self.batch_size > self.current_chunk_size:
            self.batch_cursor = 0
            self.next_chunk()

        # ak z nejakeho dovodu nie je dost dat do  batchu (napr. malo suborov) tak to sposobi errory
        # tiez len docasne riesenie
        if len(labels) < self.batch_size:
            self.next_batch()

        return data, labels



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