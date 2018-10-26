import os

# USE_GPU = True
# if not USE_GPU:
#     os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
#     os.environ["CUDA_VISIBLE_DEVICES"] = ""

from tensorflow import keras
import tensorflow as tf
import numpy as np
from scipy import misc
import glob
import matplotlib.pyplot as plt
from .util import ImgDrawer
import ntpath


def rename_data(data_dir='../trainingData/GQN_SimpleRoom/*'):
    data_paths = glob.iglob(data_dir)
    for i, dp in enumerate(data_paths):
        os.rename(dp, dp.replace('GQN_SimpleRoom_', ''))


def get_coordinates_from_filename(string):
    name_data_list = ntpath.basename(string)\
        .replace('(', '').replace(')', '')\
        .split('_')
    coordinates = []
    for num in name_data_list[0].split(','):
        coordinates.append(float(num))
    rotations = []
    for num in name_data_list[1].split(','):
        rotations.append(float(num))
    return coordinates, rotations

def get_data(num_data_points=None, data_dir='../trainingData/GQN_SimpleRoom/*'):
    data_paths = glob.iglob(data_dir)
    images = []
    coordinates = []
    rotations = []
    for i, dp in enumerate(data_paths):
        coordinate, rotation = get_coordinates_from_filename(dp)
        coordinates.append(coordinate)
        rotations.append(rotation)
        images.append(misc.imread(dp))
        if num_data_points and i+1 >= num_data_points:
            break
    return np.array(images) / 255, \
           np.array(coordinates) / np.argmax(coordinates), \
           np.array(rotations) / np.argmax(rotations)


def product(iterable):
    r = 1
    for num in iterable:
        r *= num
    return r


def autoencoder(picture_input_shape):
    number_of_pixels = product(picture_input_shape)
    model = keras.Sequential([
        keras.layers.Conv3D(28, (3,3), input_shape=picture_input_shape),
        keras.layers.Dense(100, 'relu'),
        keras.layers.Dense(20, 'relu'),
        keras.layers.Dense(100, 'relu'),
        keras.layers.Conv3D(28,(3, 3)),
        keras.layers.Dense(number_of_pixels, 'relu')
    ])
    model.compile(keras.optimizers.Adam, 'mse', ['mse'])
    return model


def get_gqn_model(picture_input_shape, coordinates_input_shape):
    # TODO Add ability to add up multiple observations to the latent representation
    number_of_pixels = product(picture_input_shape)

    picture_input = keras.Input(picture_input_shape, name='picture_input')
    coordinates_input = keras.Input(coordinates_input_shape, name='coordinates_input')

    #x = keras.layers.Dense(1000, 'relu', True)(picture_input)
    x = keras.layers.Dense(500, 'relu', True)(picture_input)

    x = keras.layers.concatenate([x, coordinates_input])
    #x = keras.layers.Dense(1000, 'relu', True)(x)
    predictions = keras.layers.Dense(number_of_pixels, 'relu', True)(x)

    model = keras.Model(inputs=[picture_input, coordinates_input], outputs=predictions)
    model.compile('rmsprop', 'mse')
    return model


def network_inputs_from_coordinates(position_datas, rotation_datas):
    coordinates = []
    for p, r in zip(position_datas, rotation_datas):
        coordinates_vec = []
        coordinates_vec.extend(p)
        coordinates_vec.extend(r)
        coordinates_vec = np.reshape(coordinates_vec, -1)
        coordinates.append(coordinates_vec)
    return np.array(coordinates)


def run(load_model=False, model_save_file='./latest_model.hdf5'):
    image_data, position_data, rotation_data = get_data(6)
    image_data = np.sum(image_data, -1) / 4

    coordinate_inputs = network_inputs_from_coordinates(position_data, rotation_data)
    flat_image_inputs = np.reshape(image_data, (-1, 10000))

    model = None
    if load_model:
        model = keras.models.load_model(model_save_file)
    else:
        gqn_model = get_gqn_model(np.shape(flat_image_inputs[0]), np.shape(coordinate_inputs[0]))
        # TODO Make it so that coordinate input and label are not the same as input image
        gqn_model.fit([flat_image_inputs, coordinate_inputs], flat_image_inputs, batch_size=None, epochs=100)
        model = gqn_model
        print('saving model')
        gqn_model.save(model_save_file)

    output = model.predict([flat_image_inputs, coordinate_inputs])


    # num_comparisons = 6
    # for i in range(1, num_comparisons + 1):
    #     plt.subplot(num_comparisons/2,4,i*2-1)
    #     plt.imshow(np.reshape(output[i-1], np.shape(image_data[0])), cmap='gray')
    #     plt.yticks([])
    #     plt.xticks([])
    #
    #     plt.subplot(num_comparisons/2,4,i*2)
    #     plt.imshow(image_data[i-1], cmap='gray')
    #     plt.yticks([])
    #     plt.xticks([])
    # plt.show()

    # TODO implement movement in gqn and rendering
    current_position
    current_rotation
    while True:


if __name__ == '__main__':
    run(False)
