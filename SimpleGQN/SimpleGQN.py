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


def delete_corrupt_files(corrupt_files, ask=True):
    if len(corrupt_files) > 0:
        if input(f'{len(corrupt_files)} out of {len(data_paths)} files could not be opened. '
                 f'The files might be corrup. Delete them? y/n\n') == 'y':
            for file in corrupt_files:
                os.remove(file)
            print(f'{len(corrupt_files)} files deleted.')


EnvData = collections.namedtuple('EnvData', 'images coordinates rotations')
def get_data_for_environments(data_dirs, num_envs_to_load=None, num_data_from_env=None, randomized_draw=True):
    data_dirs = [data_dirs + '\\' + x for x in os.listdir(data_dirs)]
    if randomized_draw:
        random.shuffle(data_dirs)
    env_data = []
    for data_paths in data_dirs[0:num_envs_to_load]:
        data_paths = glob.glob(data_paths + '/*')
        if randomized_draw:
            random.shuffle(data_paths)
        images, coordinates, rotations, corrupt_files = [], [], [], []
        for i, dp in enumerate(data_paths):
            coordinate, rotation = get_coordinates_from_filename(dp)
            try:
                images.append(misc.imread(dp))
            except OSError:
                corrupt_files.append(dp)
                continue
            coordinates.append(coordinate)
            rotations.append(rotation)
            if num_data_from_env and i+1 >= num_data_from_env:
                break
        env_data.append(EnvData(images, coordinates, rotations))
        print(f'loaded {len(images)} entrys from  environment '
              f'{len(env_data)}/{num_envs_to_load if num_envs_to_load else len(data_dirs)}', end='\n')
    delete_corrupt_files(corrupt_files)
    return env_data


def normalize_data(data_tuple):
    return (np.asarray(x) / np.max(x) for x in data_tuple)


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
    print('creating model')
    # TODO Add ability to add up multiple observations to the latent representation
    number_of_pixels = product(picture_input_shape)

    picture_input = keras.Input(picture_input_shape, name='picture_input')
    coordinates_input = keras.Input(coordinates_input_shape, name='coordinates_input')

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

    print('model created')
    return model


# TODO test: might not work anymore
def network_inputs_from_coordinates_vect(position_datas, rotation_datas):
    coordinates = []
    for p, r in zip(position_datas, rotation_datas):
        network_inputs_from_coordinates_single(p, r)
        coordinates.append(coordinates_vec)
    return np.asarray(coordinates)

def network_inputs_from_coordinates_single(position_data, rotation_data):
    coordinates_vec = []
    coordinates_vec.extend(list(position_data))
    coordinates_vec.extend(list(rotation_data))
    return np.asarray(coordinates_vec)


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
    def __init__(self, center_pos=(0, 1.5, 0)):
        self.center_pos = np.asarray(center_pos)
        self.current_position = np.array(self.center_pos)
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
        return np.asarray([0, np.sin(rot), 0, np.cos(rot)])

    @property
    def current_rotation_quaternion(self):
        return self.y_rot_to_quaternion(self.current_y_rotation)

    def movement_update(self):
        delta_time = time.time() - self.prev_time
        self.prev_time = time.time()

        current_rot_x2 = self.y_rot_to_quaternion(self.current_y_rotation * 2)
        forward_vec = np.asarray([np.sin(current_rot_x2[1]), 0., np.sin(current_rot_x2[3])])
        current_rot_x2_90_deg_offset = self.y_rot_to_quaternion(self.current_y_rotation * 2 + np.pi / 2)
        right_vec = -1 * np.asarray([np.sin(current_rot_x2_90_deg_offset[1]), 0., np.sin(current_rot_x2_90_deg_offset[3])])

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
            print('reset to center')
            self.current_position = np.array(self.center_pos)


def train_model(network_inputs, model_save_file_path, model_to_train, environment_batch_size=None, batch_size=None):
    model_save_file_path = remove_extension(model_save_file_path)
    checkpoint_save_path = f'{model_save_file_path}.checkpoint'
    model_final_save = f'{model_save_file_path}.hdf5'

    epochs = 20
    sub_epochs = 2
    training_aborted = False
    for i in range(int(epochs / sub_epochs)):
        random.shuffle(network_inputs)
        for i, (flat_image_inputs, coordinate_inputs) in enumerate(network_inputs):
            if environment_batch_size and i > environment_batch_size:
                break
            flat_image_inputs = np.asarray(flat_image_inputs)
            coordinate_inputs = np.asarray(coordinate_inputs)
            scrambled_flat_image_inputs_2 = np.random.permutation(flat_image_inputs)

            model_to_train.fit([scrambled_flat_image_inputs_2, coordinate_inputs], flat_image_inputs, batch_size=batch_size,
                                epochs=sub_epochs,
                                callbacks=[
                                    keras.callbacks.EarlyStopping(monitor='loss', patience=4),
                                    keras.callbacks.LambdaCallback(on_epoch_end=lambda _, _b: print(f'\n\nTrueEpoch {i*sub_epochs}/{epochs} - Environment epoch {i}')),
                                    keras.callbacks.ModelCheckpoint(checkpoint_save_path, period=sub_epochs)
                                ]
            )
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
def run(data_dirs, model_save_file_path, image_dim, load_model_path=None, train=True,
        num_envs_to_load=None, num_data_from_env=None, batch_size=None, run_environment=True, black_n_white=True, window_size=(1200, 600),
        window_size_coef=1):
    '''
    Run the main Programm
    :param data_dirs: the directory containing the training data.
    :param model_save_file_path: the path where to save a model.
    :param image_dim: the dimensions of the images in the data path.
    :param load_model_path: the name of the model to load. None = train new model.
    :param num_samples_to_load: number of samples to load from the data dir. None = all.
    :return: None
    '''
    def get_closesd_image_to_coordinates(pos, rot):
        return np.zeros((32,32)) + 255
        # similarity_at_idx = []
        # for im_pos, im_rot in zip(position_data, rotation_data):
        #     similarity_at_idx.append((np.sum(np.abs(pos - im_pos)) + np.sum(np.abs(rot - im_rot)) * 10))
        # return env[np.argmax(similarity_at_idx)]

    def black_n_white_to_rgb255(img):
        img = np.stack([img] * 3, -1)
        return img / np.max(img) * 255

    def rgba_to_black_n_white(img):
        return np.sum(img, -1) / 4

    def get_random_observation_inputs(network_inputs, number_of_observations=1):
        return random.choices(random.choice(network_inputs)[0], k=number_of_observations)

    def get_max_env_data_values(environment_data):
        max_img, max_pos, max_rot = 0, 0, 0
        for e in environment_data:
            im, pos, rot = e
            max_img = max(np.max(im), max_img)
            max_pos = max(np.max(pos), max_pos)
            max_rot = max(np.max(rot), max_rot)
        return max_img, max_pos, max_rot

    def normalize_environment_data(environment_data):
        max_img, max_pos, max_rot = get_max_env_data_values(environment_data)
        return [(img / max_img, pos / max_pos, rot / max_rot) for img, pos, rot in environment_data]

    def unnormal_data_to_network_input(unnormalized_environment_data, black_n_white=False):
        envs_data = []
        for env in normalize_environment_data(unnormalized_environment_data):
            images, poss, rots = env
            if black_n_white:
                images = [rgba_to_black_n_white(img) for img in images]
            images = [np.reshape(img, product(image_dim)) for img in images]
            coordinates = [network_inputs_from_coordinates_single(pos, rot) for pos, rot in zip(poss, rots)]
            envs_data.append((np.asarray(images), np.asarray(coordinates)))
        return envs_data

    window_size = collections.namedtuple('Rect', field_names='x y')(
        x=int(window_size[0] * window_size_coef),
        y=int(window_size[1] * window_size_coef))

    unnormalized_environment_data = get_data_for_environments(data_dirs, num_envs_to_load, num_data_from_env)
    _, max_pos_val, max_rot_val = get_max_env_data_values(unnormalized_environment_data)
    img_data_shape_rgb = np.shape(unnormalized_environment_data[0][0][0])
    img_data_shape_bw = img_data_shape_rgb[:-1]
    network_inputs = unnormal_data_to_network_input(unnormalized_environment_data, black_n_white=True)

    model = None
    if load_model_path:
        model = keras.models.load_model(load_model_path)
    else:
        model = get_gqn_model(np.shape(network_inputs[0][0][0]), np.shape(network_inputs[0][1][0]))
    if train:
        model = train_model(network_inputs, model_save_file_path, model, batch_size=batch_size)

    if not run_environment:
        return


    character_controller = CharacterController(center_pos=(0,1.5,0) / max_pos_val)
    img_drawer = ImgDrawer(window_size)

    # TODO make it work with network input strukture
    observation_input = get_random_observation_inputs(network_inputs, 1)
    coordinate_input = [network_inputs_from_coordinates_single(character_controller.current_position, character_controller.current_rotation_quaternion)]
    while True:
        output = model.predict([observation_input, coordinate_input])
        output = np.reshape(output[0], img_data_shape_bw)

        stacked_output = black_n_white_to_rgb255(output)

        img_drawer.draw_image(stacked_output, display_duration=0, size=(window_size.x // 2, window_size.y))
        closesed_image = get_closesd_image_to_coordinates(character_controller.current_position, character_controller.current_rotation_quaternion)
        img_drawer.draw_image(black_n_white_to_rgb255(closesed_image), display_duration=0,
                              size=(window_size.x // 4, window_size.y // 2), position=(window_size.x // 2, 0))
        img_drawer.draw_image(black_n_white_to_rgb255(np.reshape(observation_input, img_data_shape_bw)),
                              size=(window_size.x // 4, window_size.y // 2), position=(window_size.x // 2, window_size.y // 2))

        img_drawer.draw_text(f'{str(character_controller.current_position * max_pos_val)} max position {max_pos_val}', (10, 10))
        img_drawer.draw_text(str(character_controller.current_rotation_quaternion), (10, 50))

        img_drawer.execute()
        character_controller.movement_update()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_KP1]:
            observation_input = get_random_observation_inputs(network_inputs)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return


def get_data_dir(key, resolution_key):
    for base_dir in data_base_dirs:
        dir = (f'{base_dir}\\{data_dirs.get(key)}\\{image_resolutions[resolution_key]}')
        if os.path.isdir(dir):
            print(f'Found data in: {dir}')
            return dir
        else:
            print(f'Data not found in: {dir}')
    raise OSError('None of specified data base dirs exist.')

def get_img_dim_form_data_dir(dir):
    dims = dir.split('\\')[-1].split('x')
    return int(dims[0]), int(dims[1])


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
    1: 'GQN_SimpleRoom',
    2: 'GQN_SimpleRoom_withobj',
    3: 'GQN_SimpleRoom_RandomizedObjects_2',
}
image_resolutions = {
    32: '32x32',
    64: '64x64',
    128: '128x128',
    256: '256x256',
}
window_resolutions = {
    'uhd': (2400, 1200),
    'hd': (1200, 600),
}


if __name__ == '__main__':
    data_dir = get_data_dir(3, 32)
    img_dims = get_img_dim_form_data_dir(data_dir)
    run(data_dirs=data_dir,
        load_model_path=model_names.get(0),
        image_dim=img_dims,
        model_save_file_path=get_unique_model_save_name(img_dims, name='normal-run'),
        num_envs_to_load=2,
        num_data_from_env=2,
        batch_size=None,
        run_environment=True,
        train=True,
        window_size=window_resolutions['hd'])
