import numpy as np
import re
import itertools
import codecs
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def load_data_and_labels(data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    with codecs.open(data_file, 'r', encoding='utf8') as f:
    # x_text = open(data_file, 'r', encoding='utf8').read().strip(' ').split('\n')
        x_text = f.read().strip(' ').split('\n')
    x_input = [i.split('\t', 1)[0] for i in x_text]
    x_input = [i.strip(' ') for i in x_input]
    y_input = [i.split('\t', 1)[-1] for i in x_text]
    y_input = [i.strip(' ') for i in y_input]

    tuple = np.shape(y_input)
    size = tuple[0]
    del y_input[size-1]
    del x_input[size-1]


    """
    k = 0
    j = 0
    for i, x in enumerate(x_input):
        if i % 5 == 0:
            x_dev[k] = x
            y_dev[k] = y_input[i]
            k++
        else:
            x_train[j] = x
            y_train[j] = y_input[i]
            j++
    """

    y = np.zeros([size, 2], dtype=int)

    for i, s in enumerate(y_input):
        if s == "Nao":
            y[i,0] = 1
            y[i,1] = 0
        if s == "Sim":
            y[i,0] = 0
            y[i,1] = 1

    return [x_input, y]