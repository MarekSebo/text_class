import os
import tensorflow as tf
import collections
import numpy as np

url = '/home/marek/PycharmProjects/text_class/'
vocabulary_size = 10000


# read txt file as list of words
def read_words(filename):
    with tf.gfile.GFile(filename, "r") as f:
        return f.read().decode("utf-8").replace("\n", "<eos>").split()


# read all texts in all dirs, record where the files and dirs end
def read_all_texts(train_path):
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
        print("files in directory {} : {}".format(dir, dir_files) )
        for filename in dir_files:
            file_words = read_words(filename)
            words.extend(file_words)
            word_count += len(file_words)
    os.chdir(url) # change the dir back

    return words


def build_dictionaries(vocabulary_size, path ):
    # output:
    #   count = list of [word, count] for the words in vocabulary
    #   dictionary = <word>:<integer>
    #   reverse_dictionary = <integer>:<word>
    all_words = read_all_texts(path)
    count = [['UNK', -1]]
    count.extend(collections.Counter(all_words).most_common(vocabulary_size - 1))
    # make a dict of [word: index]
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return count, dictionary, reverse_dictionary


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


def final_data(train_path, dictionary):
    # v train su iba priecinky, v kazdom su txt subory v jednej class
    label_names = os.listdir(train_path)
    label_names.sort()
    unk_count = 0 # count of unknown words
    data = []
    labels = []

    for dir in label_names:
        os.chdir(os.path.join(train_path, dir)) # enter the class dir
        dir_files = os.listdir()
        dir_files.sort()
        len_label = len(dir_files)  # count of files of the class
        labels += [str(dir)] * len_label

        for filename in dir_files:
            file_words = read_words(filename)
            integers, unk_in_file = words_to_integers(file_words, dictionary)
            unk_count += unk_in_file
            data.append(integers)


    os.chdir(url)  # change the dir back
    return data, labels, unk_count






count, dictionary, reverse_dictionary = build_dictionaries(vocabulary_size, "/home/marek/PycharmProjects/text_class/train/")
data,labels, count[0] = final_data("/home/marek/PycharmProjects/text_class/train/", dictionary)
print([reverse_dictionary[data[50][i]] for i in range(len(data[50])) ])
print(labels[50])


print("victory")