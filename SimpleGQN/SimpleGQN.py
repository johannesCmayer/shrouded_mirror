import numpy as np
import tensorflow as tf
from tensorflow import keras
from scipy import misc
import glob
import matplotlib.pyplot as plt


def get_data(num_data_points=None, data_dir='../trainingData/GQN_SimpleRoom'):
    data_paths = glob.glob(data_dir)
    data = []
    for i, dp in enumerate(data_paths):
        data.append(misc.imread(dp))
        if num_data_points and i+1 >= num_data_points:
            break
    return data



def autoencoder(number_of_pixels):
    model = keras.Sequential([
        #keras.layers.1D
        keras.layers.Dense(100, 'relu', True),
        keras.layers.Dense(20, 'relu', True),
        keras.layers.Dense(100, 'relu', True),
        keras.layers.Dense(number_of_pixels, 'relu', True)
    ])
    model.compile(keras.optimizers.Adam, 'mse', ['mse'])
    return model


def gqn(picture_input, position_querry_input, number_of_pixels):
    x = keras.layers.Dense(100, 'relu', True)(picture_input)
    x = keras.layers.Dense(20, 'relu', True)(x + position_querry_input)
    x = keras.layers.Dense(100, 'relu', True)(x)
    predictions = keras.layers.Dense(number_of_pixels, 'relu', True)(x)


def run():
    # image_shape = (100, 100)
    # number_of_pixels = sum(image_shape)
    #
    # picture_input = keras.Input(shape=image_shape)
    # position_querry_input = keras.Input(shape=(6,))


    data = get_data(1)
    plt.imshow(data[0])
    plt.show()


if __name__ == '__main__':
    run()