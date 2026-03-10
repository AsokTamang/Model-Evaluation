from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2
def build_linear_nn_model():
    tf.random.set_seed(20)
    model1 = Sequential(
        [Dense(30, activation='relu'),
         Dense(15, activation='relu'),
         Dense(1, activation='linear') ],name='model1'
    )

    model2 = Sequential(
        [Dense(30, activation='relu'),
         Dense(15, activation='relu'),
         Dense(10, activation='relu'),
         Dense(8, activation='relu'),
         Dense(1, activation='linear'),
         ], name='model2'
    )

    model3 = Sequential(
        [Dense(32, activation='relu'),
         Dense(16, activation='relu'),
         Dense(8, activation='relu'),
         Dense(4, activation='relu'),
         Dense(1, activation='linear'),
         ], name='model3'
    )

    return [model1, model2, model3]

def build_logistic_nn_model():
    tf.random.set_seed(20)

    model1 = Sequential([
        Dense(16, activation='relu'),
        Dense(8,  activation='relu'),
        Dense(1,  activation='sigmoid')
    ], name='model_1')

    model2 = Sequential([
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(8,  activation='relu'),
        Dense(1,  activation='sigmoid')
    ], name='model_2')

    model3 = Sequential([
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),  #using the regularization
        Dense(16, activation='relu', kernel_regularizer=l2(0.01)),
        Dense(8,  activation='relu', kernel_regularizer=l2(0.01)),
        Dense(1,  activation='sigmoid')
    ], name='model_3')

    return [model1, model2, model3]





def plot_loss_function(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], color='blue', label='Train Loss')

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve Over Training Epochs')
    plt.legend()
    plt.grid(True)
    plt.show()