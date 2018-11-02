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
import collections
import math
import keyboard

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
    data_paths = glob.glob(data_dir + '/*')
    print(f'loading {num_data_points if num_data_points else len(data_paths)} data points.')
    images = []
    coordinates = []
    rotations = []
    corrupt_files = []
    for i, dp in enumerate(data_paths):
        coordinate, rotation = get_coordinates_from_filename(dp)
        try:
            images.append(misc.imread(dp))
        except OSError:
            corrupt_files.append(dp)
            continue
        coordinates.append(coordinate)
        rotations.append(rotation)
        if num_data_points and i+1 >= num_data_points:
            break
    if len(corrupt_files) > 0:
        if input(f'{len(corrupt_files)} out of {len(data_paths)} files could not be opened. '
                 f'The files might be corrup. Delete them? y/n\n') == 'y':
            for file in corrupt_files:
                os.remove(file)
            print(f'{len(corrupt_files)} files deleted.')
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
    x = keras.layers.Dense(1024, 'relu', True)(picture_input)
    for _ in range(9):
        x = keras.layers.Dense(1024, 'relu', True)(x)
    x = keras.layers.Dense(512, 'relu', True)(x)

    x = keras.layers.concatenate([x, coordinates_input])
    for _ in range(10):
        x = keras.layers.Dense(1024, 'relu', True)(x)
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


def get_unique_model_save_name(img_shape, name=''):
    return f'{datetime.datetime.now()}_{img_shape[0]}x{img_shape[1]}_{name}.hdf5'.replace(':', '-')


def remove_extension(path, extension='hdf5'):
    if path.split('.')[-1] == extension:
        return ''.join(path.split('.')[0:-2])
    return path


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


class CharacterController:
    def __init__(self, position_data_un):
        self.center_pos = np.array([0, 1.5, 0]) / np.argmax(position_data_un)
        self.current_position = self.center_pos
        self.current_y_rotation = 0
        self.prev_time = 0
        self.move_speed = 0.8
        self.rotate_speed = 0.8

    @staticmethod
    def y_rot_to_quaternion(rot):
        if rot > np.pi:
            rot -= np.pi * (int(rot / np.pi))
        if rot < 0:
            rot += np.pi * (1 + int(rot / np.pi))
        return np.array([0, np.sin(rot), 0, np.cos(rot)])

    @property
    def current_rotation_quaternion(self):
        return self.y_rot_to_quaternion(self.current_y_rotation)

    def movement_update(self):
        delta_time = time.time() - self.prev_time
        self.prev_time = time.time()

        current_rot_x2 = self.y_rot_to_quaternion(self.current_y_rotation * 2)
        forward_vec = np.array([np.sin(current_rot_x2[1]), 0., np.sin(current_rot_x2[3])])
        current_rot_x2_90_deg_offset = self.y_rot_to_quaternion(self.current_y_rotation * 2 + np.pi / 2)
        right_vec = -1 * np.array([np.sin(current_rot_x2_90_deg_offset[1]), 0., np.sin(current_rot_x2_90_deg_offset[3])])

        pygame.event.pump()
        keys = pygame.key.get_pressed()

        if keys[pygame.K_a]:
            self.current_y_rotation += self.rotate_speed * delta_time
        if keys[pygame.K_s]:
            self.current_y_rotation -= self.rotate_speed * delta_time

        if keys[pygame.K_UP]:
            self.current_position += self.move_speed * delta_time * forward_vec
        if keys[pygame.K_DOWN]:
            self.current_position += self.move_speed * delta_time * -forward_vec
        if keys[pygame.K_LEFT]:
            self.current_position += self.move_speed * delta_time * -right_vec
        if keys[pygame.K_RIGHT]:
            self.current_position += self.move_speed * delta_time * right_vec

        if keys[pygame.K_KP2]:
            self.current_position = self.center_pos


def train_model(flat_image_inputs, coordinate_inputs, model_save_file_path, model_to_train=None, batch_size=None):
    model_save_file_path = remove_extension(model_save_file_path)
    checkpoint_save_path = f'{model_save_file_path}.checkpoint'
    model_final_save = f'{model_save_file_path}.hdf5'

    if not model_to_train:
        model_to_train = get_gqn_model(np.shape(flat_image_inputs[0]), np.shape(coordinate_inputs[0]))
    epochs = 500
    sub_epochs = 2
    training_aborted = False
    for i in range(int(epochs / sub_epochs)):
        start_time = time.time()
        scrampled_flat_image_inputs = np.random.permutation(flat_image_inputs)
        model_to_train.fit([scrampled_flat_image_inputs, coordinate_inputs], flat_image_inputs, batch_size=batch_size,
                      epochs=sub_epochs, callbacks=[
                keras.callbacks.EarlyStopping(monitor='loss', patience=4),
                keras.callbacks.LambdaCallback(on_epoch_end=lambda _, _b: print(f'\n\nTrueEpoch {i*sub_epochs}/{epochs}')),
                keras.callbacks.ModelCheckpoint(checkpoint_save_path, period=sub_epochs)
            ])
        if keyboard.is_pressed('q'):
            print('learning aborted by user')
            training_aborted = True
            break

    print('saving model')
    model_to_train.save(model_final_save)
    if not training_aborted:
        print('removing checkpoint save')
        os.remove(checkpoint_save_path)
    return model_to_train


# TODO clean up and split up run method
def run(data_dir, model_save_file_path, image_dim, load_model_path=None, train=True,
        num_samples_to_load=None, batch_size=None, run_environment=True):
    '''
    Run the main Programm
    :param data_dir: the directory containing the training data.
    :param model_save_file_path: the path where to save a model.
    :param image_dim: the dimensions of the images in the data path.
    :param load_model_path: the name of the model to load. None = train new model.
    :param num_samples_to_load: number of samples to load from the data dir. None = all.
    :return: None
    '''
    image_data_un, position_data_un, rotation_data_un = get_data(data_dir, num_samples_to_load)
    image_data, position_data, rotation_data = normalize_data(image_data_un, position_data_un, rotation_data_un)
    image_data = np.sum(image_data, -1) / 4

    coordinate_inputs = network_inputs_from_coordinates(position_data, rotation_data)
    flat_image_inputs = np.reshape(image_data, (-1, product(image_dim)))

    def get_closesd_image_to_coordinates(pos, rot):
        similarity_at_idx = []
        for im_pos, im_rot in zip(position_data, rotation_data):
            similarity_at_idx.append((np.sum(np.abs(pos - im_pos)) + np.sum(np.abs(rot - im_rot)) * 10))
        return image_data[np.argmax(similarity_at_idx)]

    def black_n_white_to_rgb255(img):
        img = np.stack([img] * 3, -1)
        return img / np.max(img) * 255

    def get_random_observation_input():
        num_s = num_samples_to_load if num_samples_to_load else len(data_dirs)
        return flat_image_inputs[random.randint(0, num_s - 1)]

    model = None
    if load_model_path:
        model = keras.models.load_model(load_model_path)
    if train or not load_model_path:
        model = train_model(flat_image_inputs, coordinate_inputs, model_save_file_path, model, batch_size=batch_size)

    if not run_environment:
        return

    window_size = collections.namedtuple('Rect', field_names='x y')(x=1200*2, y=600*2)

    character_controller = CharacterController(position_data_un)
    img_drawer = ImgDrawer(window_size)
    prev_time = 0

    observation_input = get_random_observation_input()
    while True:
        output = model.predict([[observation_input], network_inputs_from_coordinates([character_controller.current_position], [character_controller.current_rotation_quaternion])])
        output = np.reshape(output[0], np.shape((image_data[0])))

        stacked_output = black_n_white_to_rgb255(output)

        img_drawer.draw_image(stacked_output, display_duration=0, size=(window_size.x // 2, window_size.y))
        closesed_image = get_closesd_image_to_coordinates(character_controller.current_position, character_controller.current_rotation_quaternion)
        img_drawer.draw_image(black_n_white_to_rgb255(closesed_image), display_duration=0,
                              size=(window_size.x // 4, window_size.y // 2), position=(window_size.x // 2, 0))
        img_drawer.draw_image(black_n_white_to_rgb255(np.reshape(observation_input, np.shape(image_data[0]))),
                              size=(window_size.x // 4, window_size.y // 2), position=(window_size.x // 2, window_size.y // 2))

        img_drawer.draw_text(f'{str(character_controller.current_position * np.argmax(position_data_un))} max position {np.max(position_data_un, 0)}', (10, 10))
        img_drawer.draw_text(str(character_controller.current_rotation_quaternion), (10, 50))

        img_drawer.execute()
        character_controller.movement_update()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_KP1]:
            observation_input = get_random_observation_input()


if __name__ == '__main__':
    model_names_home = {
        1: 'first_large.hdf5',
        2: '2018-10-26.15-41-54.307222_super-long-run',
        3: '2018-11-01 21-15-16_(32, 32).hdf5',
        4: '2018-11-01 22-18-28_(32, 32).hdf5',
        5: '2018-11-02 01-19-11_(32, 32).checkpoint',
    }
    model_names_uni = {
        -1: '2018-11-02 13-47-15.hdf5'
    }
    model_names = {'train': None, 0: None, **model_names_home, **model_names_uni}

    data_base_dirs = ['D:\\Projects\\Unity_Projects\\GQN_Experimentation\\trainingData',
                      r'D:\JohannesCMayer\GQN_Experimentation\trainingData']
    data_dirs = {
        1: 'GQN_SimpleRoom_32x32',
        2: 'GQN_SimpleRoom_withobj_32x32',
    }
    resolutions = {
        'uhd': (2400, 1200),
        'hd': (1200, 600),
    }

    def get_data_dir(key):
        for base_dir in data_base_dirs:
            dir = (base_dir + '/' + data_dirs.get(key))
            if os.path.isdir(dir):
                return dir
        raise OSError('None of specified data base dirs exist.')

    def get_img_dim_form_data_dir(dir):
        dims = dir.split('_')[-1].split('x')
        return int(dims[0]), int(dims[1])

    data_dir = get_data_dir(2)
    img_dims = get_img_dim_form_data_dir(data_dir)
    run(data_dir=data_dir, load_model_path=model_names.get(0), image_dim=img_dims,
        model_save_file_path=get_unique_model_save_name(img_dims, name='normal-run'), num_samples_to_load=None,
        batch_size=None, run_environment=True, train=True)
