from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import tensorflow as tf
def build_linear_model():
    tf.random.set_seed(20)
    model1 = Sequential(
        [Dense(25, activation='relu'),
         Dense(15, activation='relu'),
         Dense(1, activation='linear') ],name='model1'
    )

    model2 = Sequential(
        [Dense(20, activation='relu'),
         Dense(12, activation='relu'),
         Dense(12, activation='relu'),
         Dense(20, activation='relu'),
         Dense(1, activation='linear'),
         ], name='model2'
    )

    model3 = Sequential(
        [Dense(32, activation='relu'),
         Dense(16, activation='relu'),
         Dense(8, activation='relu'),
         Dense(4, activation='relu'),
         Dense(12, activation='relu'),
         Dense(1, activation='linear'),
         ], name='model3'
    )

    return [model1, model2, model3]
