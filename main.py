# Imports
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras
from keras import preprocessing


#Code starts here
data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=88000)

# print(train_data[0])
# data with different length => can't come up with the shape needed for our neural network model
# print(len(train_data[0]), len(train_data[1]))
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
# print(len(train_data[0]), len(train_data[1]))

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

# for i in range(len(train_data)):
#     print(decode_review(train_data[i]))


# Set up model
# model = keras.Sequential()
# model.add(keras.layers.Embedding(88000, 16))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(16, activation = "relu"))
# model.add(keras.layers.Dense(1, activation = "sigmoid"))
#
# # print(model.summary())
#
# model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics=["accuracy"])
#
# x_valid = train_data[:10000]
# x_train = train_data[10000:]
#
# y_valid = train_labels[:10000]
# y_train = train_labels[10000:]
#
# fitModel = model.fit(x_train, y_train, epochs=40, batch_size=512, validation_data=(x_valid, y_valid), verbose=1)
# result = model.evaluate(test_data, test_labels)
#
# print(result)

# Save the model
# model.save('imdb.h5')


def review_encode(s):
    encoded = [1]

    for word in s:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)

    return encoded


model = keras.models.load_model("imdb.h5")
class_name = ["negative", "positive"]
with open("test.txt", encoding = "utf-8") as f:
    for line in f.readlines():
        nline = line.replace(",", "").replace("\"", "").replace(".", "").replace(")", "").replace("(", "").replace(":", "").replace("-", "").strip().split(" ")
        encode = review_encode(nline)
        encode = preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen =250)
        predict = model.predict(encode)
        print("Review\n", line)
        print(encode)
        print(class_name[round(predict[0])])




# class_name = ["negative", "positive"]
#
# test_review = test_data[2]
# prediction = model.predict([test_review])
# # print(prediction)
# print("Review:\n", decode_review(test_review))
# print("Prediction:\n", class_name[round(prediction[0][0])])
# print("Actual:\n", class_name[test_labels[2]])