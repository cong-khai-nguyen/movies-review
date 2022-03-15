# Imports
from tensorflow import keras
from keras import preprocessing
import numpy as np

#Code starts here
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=10000)

# print(train_data[0])
# data with different length => can't come up with how many neurons for input
# print(len(test_data[0]), len(test_data[1]))
word_index = data.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for(key, value) in word_index.items()])

# Preprocess data by capping data length at 250 and for data with length less than 250, the data will be filled with 0
train_data = preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen =250)
test_data = preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen =250)
print(len(train_data[0]), len(train_data[1]))

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

