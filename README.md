# Movies Reviews
Curious to test out? 

Link: https://colab.research.google.com/drive/1vFKBwrx6iX73MKS4weP1HLUCWawfj_xi?usp=sharing
# Description
The purpose of this project is to classify the sentiment of a movie review(negative or positive). Out of all the data I've used for projects, this one is the toughest for me to manipulate. Since each review is different length and hence produce a problem as to the model:different shape, I precocess data and cap the data length at 250. And for data with length less than 250, the data will be filled with 0, indicating to be skipped. In the end, using neural network models with four layers, I was able to correctly predict up to 89%.

The data set can also be found here: [IMBD](https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data)

For this project, I follow tensorflow documentation/tutorial here: https://www.tensorflow.org/tutorials/keras/text_classification and https://www.tensorflow.org/api_docs/python/tf/keras/datasets/imdb/load_data
# Install and Run the Project
This project requires installed Python libraries: tensorflow, keras.

Note: the imdb data is included when we download keras. You can access it by including this import and function below:
```
from tensorflow import keras
data = keras.datasets.imdb
