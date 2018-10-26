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
# TODO somehow fix the false import error
from util import ImgDrawer
import ntpath
import pygame
import datetime
import time
import random
import re
from typing import Dict


# TODO refactor everything

# TODO Implement this and understand re
def delete_data(data_dir, regex: str):
    data_paths = glob.iglob(data_dir)
    for i, dp in enumerate(data_paths):
        #if re.
        #os.remove(dp)
        pass


def rename_data(data_dir, dict_str_to_replace: Dict):
    data_paths = glob.iglob(data_dir + '/*')
    for i, dp in enumerate(data_paths):
        for string_to_find, string_to_replace_with in zip(dict_str_to_replace.keys(), dict_str_to_replace.values()):
            os.rename(dp, dp.replace(string_to_find, string_to_replace_with))


def get_coordinates_from_filename(string):
    name_data_list = ntpath.basename(string)\
        .replace('(', '').replace(')', '')\
        .split('_')
    coordinates = []
    for num in name_data_list[0].split(', '):
        coordinates.append(float(num.replace(',', '.')))
    rotations = []
    for num in name_data_list[1].split(', '):
        rotations.append(float(num.replace(',', '.')))
    return coordinates, rotations

def get_data(data_dir, num_data_points=None):
    data_paths = glob.iglob(data_dir + '/*')
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
    return images, coordinates, rotations


def normalize_data(images, coordinates, rotations):
    return np.array(images) / np.max(images), \
           np.array(coordinates) / np.max(coordinates), \
           np.array(rotations) / np.max(rotations)


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
    x = keras.layers.Dense(600, 'relu', True)(picture_input)
    x = keras.layers.Dense(600, 'relu', True)(x)
    x = keras.layers.Dense(600, 'relu', True)(x)
    x = keras.layers.Dense(500, 'relu', True)(x)

    x = keras.layers.concatenate([x, coordinates_input])
    x = keras.layers.Dense(600, 'relu', True)(x)
    x = keras.layers.Dense(600, 'relu', True)(x)
    x = keras.layers.Dense(600, 'relu', True)(x)
    predictions = keras.layers.Dense(number_of_pixels, 'relu', True)(x)

    model = keras.Model(inputs=[picture_input, coordinates_input], outputs=predictions)
    model.compile('rmsprop', 'mse')
    return model


def network_inputs_from_coordinates(position_datas, rotation_datas):
    coordinates = []
    for p, r in zip(position_datas, rotation_datas):
        coordinates_vec = []
        coordinates_vec.extend(list(p))
        coordinates_vec.extend(list(r))
        coordinates_vec = np.reshape(coordinates_vec, -1)
        coordinates.append(coordinates_vec)
    return np.array(coordinates)


def wrap_around_quaternion_rotation(normal_quaternion_iterable):
    if normal_quaternion_iterable[3] > 0.99:
        normal_quaternion_iterable = np.array([0, 0.98, 0, 0.011])
    if normal_quaternion_iterable[3] < 0.01:
        normal_quaternion_iterable = np.array([0, 0.011, 0, 0.98])
    return normal_quaternion_iterable

def normalize_quarternion(quaternion_iterable):
    quat_mag = 0
    for e in quaternion_iterable:
        quat_mag += np.power(e, 2)
    quat_mag = np.sqrt(quat_mag)
    normalized_quat = []
    for e in quaternion_iterable:
        normalized_quat.append(e / quat_mag)
    # TODO extract the wraparound behaviour to somewhere else
    return wrap_around_quaternion_rotation(np.array(normalized_quat))


def get_unique_model_save_name(img_shape, name=''):
    return f'{datetime.datetime.now()}_{name}_{img_shape}.hdf5'.replace(':', '-')


# TODO clean up and split up run method
def run(data_dir, model_save_file_path, image_dim, load_model_name=None):
    num_samples = 5500
    image_data_un, position_data_un, rotation_data_un = get_data(data_dir, num_samples)
    image_data, position_data, rotation_data = normalize_data(image_data_un, position_data_un, rotation_data_un)
    image_data = np.sum(image_data, -1) / 4

    coordinate_inputs = network_inputs_from_coordinates(position_data, rotation_data)
    flat_image_inputs = np.reshape(image_data, (-1, product(image_dim)))

    model = None
    if load_model_name:
        if load_model_name.split('.')[-1] != 'hdf5':
            load_model_name += '.hdf5'
        model = keras.models.load_model(load_model_name)
    else:
        gqn_model = get_gqn_model(np.shape(flat_image_inputs[0]), np.shape(coordinate_inputs[0]))
        epochs = 200000
        comp_times = 0
        for i in range(int(epochs/10)):
            start_time = time.time()
            scrampled_flat_image_inputs = np.random.permutation(flat_image_inputs)
            gqn_model.fit([scrampled_flat_image_inputs, coordinate_inputs], flat_image_inputs, batch_size=None,
                          epochs=10, callbacks=[keras.callbacks.EarlyStopping(monitor='loss', patience=4),
                          keras.callbacks.LambdaCallback(on_epoch_end=lambda _, _b: print(f'TrueEpoch {i*10}/{epochs}'))])
            comp_times = (comp_times + time.time() - start_time) / 2
            eta = comp_times * (epochs - i * 10)
            # TODO fix eta measure (complete wrong time currently)
            print(f'ETA: {eta}')
        model = gqn_model
        print('saving model')
        if model_save_file_path.split('.')[-1] == 'hdf5':
            model_save_file_path = ''.join(model_save_file_path.split('.')[0:-2])
        gqn_model.save(model_save_file_path + '_' + str(image_dim) + '.hdf5')


    def draw_autoencoding_evaluation():
        num_comparisons = 6
        for i in range(1, num_comparisons + 1):
            plt.subplot(num_comparisons/2,4,i*2-1)
            plt.imshow(np.reshape(output[i-1], np.shape(image_data[0])), cmap='gray')
            plt.yticks([])
            plt.xticks([])

            plt.subplot(num_comparisons/2,4,i*2)
            plt.imshow(image_data[i-1], cmap='gray')
            plt.yticks([])
            plt.xticks([])
        plt.show()

    def get_closesd_image_to_coordinates(pos, rot):
        similarity_at_idx = []
        for im_pos, im_rot in zip(position_data, rotation_data):
            similarity_at_idx.append((np.sum(np.abs(pos - im_pos)) + np.sum(np.abs(rot - im_rot)) * 10))
        return image_data[np.argmax(similarity_at_idx)]


    def black_n_white_to_rgb255(img):
        img = np.stack([img] * 3, -1)
        return img / np.max(img) * 255


    def get_random_observation_input():
         return flat_image_inputs[random.randint(0, num_samples - 1)]

    current_position = np.array([0, 1.5, 0]) / np.argmax(position_data_un)
    current_rotation = normalize_quarternion(np.array([0, 0.5, 0, 0.5]))
    img_drawer = ImgDrawer((1200*2, 1200))
    prev_time = 0

    observation_input = get_random_observation_input()
    while True:
        delta_time = time.time() - prev_time
        prev_time = time.time()


        output = model.predict([[observation_input], network_inputs_from_coordinates([current_position], [current_rotation])])
        output = np.reshape(output[0], np.shape((image_data[0])))

        stacked_output = black_n_white_to_rgb255(output)

        img_drawer.draw_image(stacked_output, display_duration=0, size=(1200, 1200))
        closesed_image = get_closesd_image_to_coordinates(current_position, current_rotation)
        img_drawer.draw_image(black_n_white_to_rgb255(closesed_image), display_duration=0, size=(600, 600), position=(1200, 0))
        img_drawer.draw_image(black_n_white_to_rgb255(np.reshape(observation_input, np.shape(image_data[0]))), size=(600,600), position=(1200,600))

        img_drawer.draw_text(f'{str(current_position * np.argmax(position_data_un))} max position {np.max(position_data_un, 0)}', (10, 10))
        img_drawer.draw_text(str(current_rotation), (10, 50))

        img_drawer.execute()


        normal_speed = 0.2
        ludacris_speed = 20
        move_speed = ludacris_speed * delta_time
        rotate_speed = 15 * delta_time

        # TODO Make it that you can properly rotate the camera; Implement unity quaternion cube

        pygame.event.pump()
        keys = pygame.key.get_pressed()
        if keys[pygame.K_a]:
            current_rotation = normalize_quarternion(current_rotation + rotate_speed * np.array([0., 0.05, 0, 0]))
        if keys[pygame.K_s]:
            current_rotation = normalize_quarternion(current_rotation + rotate_speed * np.array([0., 0, 0, 0.05]))

        if keys[pygame.K_UP]:
            current_position += move_speed * np.array([0.05, 0., 0.])
        if keys[pygame.K_DOWN]:
            current_position += move_speed * np.array([-0.05, 0., 0.])
        if keys[pygame.K_LEFT]:
            current_position += move_speed * np.array([0., 0., 0.05])
        if keys[pygame.K_RIGHT]:
            current_position += move_speed * np.array([0., 0., -0.05])

        if keys[pygame.K_KP1]:
            observation_input = get_random_observation_input()

if __name__ == '__main__':
    model_names = {
        0: None,
        1: 'first_large.hdf5',
        2: '2018-10-26.15-41-54.307222_super-long-run'
    }

    data_dir_base = r'D:\JohannesCMayer\GQN_Experimentation\trainingData/'
    data_dirs = {
        1: 'GQN_SimpleRoom_32x32'
    }
    def get_data_dir(key):
        return data_dir_base + data_dirs.get(key)

    def get_img_dim_form_data_dir(dir):
        dims = dir.split('_')[-1].split('x')
        return int(dims[0]), int(dims[1])

    data_dir = get_data_dir(1)
    img_dims = get_img_dim_form_data_dir(data_dir)
    run(data_dir=data_dir, load_model_name=model_names.get(0), image_dim=img_dims,
        model_save_file_path=get_unique_model_save_name('weekend-run'))