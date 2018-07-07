#! /usr/bin/env python

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
# from text_cnn import TextCNN
import numpy as np
import os
import time
import datetime
import data_helpers
torch.manual_seed(1)


# Parametros
# ==================================================

# Parametros para carregar os dados
dev_sample_percentage = 0.1 # Percentage of the training data to use for validation
data_file = "trainDevSetBahiaBalanced.txt" # Entrada


# Hiperparametros do modelo
word2vec = "None" # Word2vec file with pre-trained embeddings (default: None)

embedding_dim = 128 # "Dimensionality of character embedding (default: 128)
filter_sizes = [3, 4, 5] # Comma-separated filter sizes (default: '3,4,5')
num_filters = 128 # Number of filters per filter size (default: 128)
dropout_keep_prob = 0.5 # Dropout keep probability (default: 0.5)
l2_reg_lambda = 0.0 # L2 regularizaion lambda (default: 0.0)

# Parametros de treino
batch_size = 64 # Batch Size (default: 64)
num_epochs = 200 # Number of training epochs (default: 200)
evaluate_every = 100 # Evaluate model on dev set after this many steps (default: 100)
checkpoint_every = 100 # Save model after this many steps (default: 100)")
# Misc Parameters
allow_soft_placement = True # Allow device soft device placement
log_device_placement = False #Log placement of ops on devices

# Data Preparation
# ==================================================

# Load data
print("Loading data...")
x_text, y = data_helpers.load_data_and_labels(data_file)
print(x_text)
print(y)
# embeds = nn.Embedding(len(vocab_processor.vocabulary_), FLAGS.embedding_dim, "max_norm" = 0.25)

# trecho para carregar Word2Vec pronto
if word2vec != "None":
    # initial matrix with random uniform
    initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
    # load any vectors from the word2vec
    print("Load word2vec file {}\n".format(FLAGS.word2vec))
    with open(FLAGS.word2vec, "rb") as f:
        header = f.readline()
        vocab_size, layer1_size = map(int, header.split())
        binary_len = np.dtype('float32').itemsize * layer1_size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            idx = vocab_processor.vocabulary_.get(word)
            if idx != 0:
                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
            else:
                f.read(binary_len)

    embeds.load_state_dict({'weight': initW})