from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

from keras.models import Sequential
from keras.layers import Dense

class MLP(keras.Model):
    def __init__(self, state_size, action_size):
        super(MLP, self).__init__()
        self.dense1 = keras.layers.Dense(64, activation='relu', input_shape=(state_size,))
        self.dense2 = keras.layers.Dense(64, activation='relu')
        self.dense3 = keras.layers.Dense(64, activation='relu')
        self.out = keras.layers.Dense(action_size, activation='linear')

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.out(x)

    def intro(self):
        self.summary()