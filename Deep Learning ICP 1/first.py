import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import to_categorical
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("Breas Cancer.csv", header=0).values
x = dataset[:,1]
# encode class values as integers
import pandas
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import to_categorical
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
# load dataset
from sklearn.model_selection import train_test_split
import pandas as pd
dataset = pd.read_csv("Breas Cancer.csv", header=0).values
x = dataset[:,1]
# encode class values as integers
encoder = LabelEncoder()
encoder.fit(x)
encoded_Y = encoder.transform(x)
print(encoded_Y)
# convert integers to dummy variables (i.e. one hot encoded)
#dummy_y = np_utils.to_categorical(encoded_Y)
#print(dummy_y)
#encoded = to_categorical(data)
#print(encoded)
#inverted = argmax(dummy_y)
#print(inverted)
import numpy as np
X_train, X_test, Y_train, Y_test = train_test_split(dataset[:,2:30], encoded_Y,
                                                    test_size=0.25, random_state=87)
np.random.seed(155)
my_first_nn = Sequential() # create model
my_first_nn.add(Dense(20, input_dim=28, activation='relu')) # hidden layer
my_first_nn.add(Dense(8, activation='relu'))
my_first_nn.add(Dense(1, activation='sigmoid')) # output layer
my_first_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
my_first_nn_fitted = my_first_nn.fit(X_train, Y_train, epochs=100, verbose=0,
                                     initial_epoch=0)
print(my_first_nn.summary())
print(my_first_nn.evaluate(X_test, Y_test, verbose=0))