import os
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dense, Concatenate, Flatten
import tensorflow as tf
import numpy as np
from scipy import misc
import glob
import matplotlib.pyplot as plt
from util import ImgDrawer, Spinner
import ntpath
import pygame
import datetime
import time
import random
from typing import Dict
import collections
import keyboard
import winsound
import names
import json
import music
import pprint
import math
import logging
import functools

print(f'started execution at {datetime.datetime.now()}')

def convert_to_valid_os_name(string, substitute_char='-'):
    return replace_multiple(string, '\\ / : * ? " < > |'.split(' '), substitute_char)


def pause_and_notify(msg='programm suspendet', activation_return_pairs={'y': True, 'n': False}, timeout=None):
    start_time = time.time()
    print(msg + ' you can specify these return values with chars ' + pprint.pformat(activation_return_pairs))
    for i in music.infinity():
        for key, val in activation_return_pairs.items():
            if keyboard.is_pressed(key):
                return val
        if timeout and time.time() - start_time > timeout:
            return activation_return_pairs.get('timeout', False)
        try:
            music.play_next_note_of_song(i)
        except Exception as e:
            print(f'winsound cant play: {e}')


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
    num_envs_to_load = num_envs_to_load if num_envs_to_load else len(data_dirs)
    if randomized_draw:
        random.shuffle(data_dirs)
    env_data = []
    num_loaded = 0
    for data_paths in data_dirs[0:num_envs_to_load]:
        data_paths = glob.glob(data_paths + '/*')
        if randomized_draw:
            random.shuffle(data_paths)
        images, coordinates, rotations, corrupt_files = [], [], [], []
        for i, dp in enumerate(data_paths):
            coordinate, rotation = get_coordinates_from_filename(dp)
            try:
                images.append(misc.imread(dp))
                num_loaded += 1
            except OSError:
                corrupt_files.append(dp)
                continue
            coordinates.append(coordinate)
            rotations.append(rotation)
            if num_data_from_env and i+1 >= num_data_from_env:
                break
        env_data.append(EnvData(images, coordinates, rotations))
        sc = spinner.get_spin_char()
        print(f'\r{sc} loaded {len(images)} entrys from environment '
              f'{len(env_data)}/{num_envs_to_load} - '
              f'{int(len(env_data) / num_envs_to_load * 100)}% - '
              f'{num_loaded} images loaded', end='')
    print()
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


# TODO make it so it can take non flat input (test if it already can)
def get_gqn_model(picture_input_shape, coordinates_input_shape, num_layers_encoder=3, num_layers_decoder=None,
                  num_neurons_per_layer=1024, num_state_neurons=1024):
    print('creating model')
    if not num_layers_decoder:
        num_layers_decoder = num_layers_encoder

    # TODO Add ability to add up multiple observations to the latent representation
    number_of_pixels = product(picture_input_shape)

    picture_input = keras.Input(picture_input_shape, name='picture_input')
    coordinates_input = keras.Input(coordinates_input_shape, name='coordinates_input')

    x = keras.layers.Dense(num_neurons_per_layer, 'relu', True)(picture_input)
    for _ in range(num_layers_encoder):
        x = keras.layers.Dense(num_neurons_per_layer, 'relu', True)(x)
    x = keras.layers.Dense(num_state_neurons, 'relu', True)(x)

    x = keras.layers.concatenate([x, coordinates_input])
    for _ in range(num_layers_decoder):
        x = keras.layers.Dense(num_neurons_per_layer, 'relu', True)(x)
    predictions = keras.layers.Dense(number_of_pixels, 'relu', True)(x)

    joint_model = keras.Model(inputs=[picture_input, coordinates_input], outputs=predictions)
    joint_model.compile('rmsprop', 'mse')

    joint_model.summary()
    return joint_model

# TODO find out how to remove gray shadow walls
def get_convolitional_gqn_model(picture_input_shape, coordinates_input_shape, num_layers_encoder=1,
                                num_layers_decoder=None, num_neurons_per_layer=1024, num_state_neurons_coef=1024,
                                downsampled_res=32):
    print('creating model')
    if not num_layers_decoder:
        num_layers_decoder = num_layers_encoder

    # TODO Add ability to add up multiple observations to the latent representation
    number_of_pixels = product(picture_input_shape)

    input_img = keras.Input(picture_input_shape, name='picture_input')
    coordinates_input = keras.Input(coordinates_input_shape, name='coordinates_input')

    downsampled_res = downsampled_res
    if any(shape % downsampled_res != 0 for shape in picture_input_shape[:2]) \
            or picture_input_shape[0] != picture_input_shape[1]:
        raise ValueError(f'Picture input shape needs to be multiple of 16 in x and y but is {picture_input_shape[:2]}')
    num_of_sampeling_steps = int(math.log2(picture_input_shape[0]) - math.log2(downsampled_res))

    B = True

    x = input_img
    for _ in range(num_of_sampeling_steps):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    # TODO replace this with a flat GQN
    if B:
        x = keras.layers.Flatten()(x)
        for _ in range(2):
            x = keras.layers.Dense(1024, 'relu', True)(x)
        encoded = x

        x = keras.layers.concatenate([encoded, coordinates_input])
        for _ in range(2):
            x = keras.layers.Dense(1024, 'relu', True)(x)
        x = Dense(downsampled_res * downsampled_res * 3, 'relu')(x)

    x = keras.layers.Reshape((downsampled_res, downsampled_res, 3))(x)

    for _ in range(num_of_sampeling_steps):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
    if x.shape != picture_input_shape:
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)

    #output_shape = [1,] + [int(x) for x in x.shape[2:]]
    #x = keras.layers.Lambda(lambda x: x[:, :, :, :3], output_shape=output_shape)(x)
    #decoded = keras.layers.Reshape(picture_input_shape)(x)
    decoded = x

    joint_model = keras.Model(inputs=[input_img, coordinates_input], outputs=decoded)
    joint_model.compile('adam', 'mse')

    joint_model.summary()
    return joint_model


# TODO parameterise all model generaton funcitons, potentialy put into class
# TODO Implement this
def get_multi_input_gqn_model(picture_input_shape, coordinates_input_shape, num_layers_encoder=3,
                              num_layers_decoder=None, num_neurons_per_layer=1024, num_state_neurons=1024):
    print('creating model')
    if not num_layers_decoder:
        num_layers_decoder = num_layers_encoder

    # TODO Add ability to add up multiple observations to the latent representation
    number_of_pixels = product(picture_input_shape)

    picture_input = keras.Input(picture_input_shape, name='picture_input')
    x = keras.layers.Dense(num_neurons_per_layer, 'relu', True)(picture_input)
    for _ in range(num_layers_encoder):
        x = keras.layers.Dense(num_neurons_per_layer, 'relu', True)(x)
    x = keras.layers.Dense(num_state_neurons, 'relu', True)(x)
    encoder_model = keras.Model(inputs=picture_input, outputs=x)

    #picture_input = keras.Input(picture_input_shape, name='picture_input')
    coordinates_input = keras.Input(coordinates_input_shape, name='coordinates_input')

    encoder_model_input = encoder_model(picture_input)
    x = keras.layers.concatenate([encoder_model_input, coordinates_input])
    for _ in range(num_layers_decoder):
        x = keras.layers.Dense(num_neurons_per_layer, 'relu', True)(x)
    predictions = keras.layers.Dense(number_of_pixels, 'relu', True)(x)
    decoder_model = keras.Model(inputs=[encoder_model_input, coordinates_input], outputs=predictions)

    decoder_model_output = decoder_model(inputs=[picture_input, coordinates_input])
    joint_model = keras.Model(inputs=[picture_input, coordinates_input], outputs=decoder_model_output)
    joint_model.compile('rmsprop', 'mse')

    joint_model.summary()
    return joint_model


# TODO test: might not work anymore
def network_inputs_from_coordinates_vect(position_datas, rotation_datas):
    coordinates = []
    for p, r in zip(position_datas, rotation_datas):
        coordinates_vec = network_inputs_from_coordinates_single(p, r)
        coordinates.append(coordinates_vec)
    return np.asarray(coordinates)

def network_inputs_from_coordinates_single(position_data, rotation_data):
    coordinates_vec = []
    coordinates_vec.extend(list(position_data))
    coordinates_vec.extend(list(rotation_data))
    return np.asarray(coordinates_vec)


def replace_multiple(str, old, new):
    for tr in old:
        str = str.replace(tr, new)
    return str


def get_unique_model_save_name(name, version, id):
    def fm(val):
        string = replace_multiple(str(val), ['=', ', ', ' ', '.'], '-')
        return convert_to_valid_os_name(string, substitute_char='-')
    date = datetime.datetime.now().date()
    time = datetime.datetime.now().time()
    d = {'date': date, 'time': time, 'name': name, 'version': version, 'id': id}
    return functools.reduce(lambda acc, val: f'{acc}_{val[0]}={fm(val[1])}', d.items(), '')[1:]


def get_new_unique_model_save_name():
    name = names.get_full_name()
    id = random.randint(1000, 10000)
    return get_unique_model_save_name(name, 1, id)


def get_model_name(previous_name=None):
    if previous_name:
        old_name_dir = os.path.dirname(previous_name)
        previous_name = os.path.basename(previous_name)
        param_dict = {key: val for key, val in [e.split('=') for e in previous_name.split('_')]}
        id = param_dict['id']
        name = param_dict['name']
        version = 0
        # TODO fix incremental versioning
        for p in glob.glob(f'{old_name_dir}\\*{name}*{id}*'):
            version = max(int(param_dict['version']), version)
        version += 1
        return get_unique_model_save_name(name, version, id)
    else:
        return get_new_unique_model_save_name()


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
        self.mouse_rotate_speed = 0.001

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


        mouse_delta = pygame.mouse.get_rel()
        keys = pygame.key.get_pressed()

        if pygame.mouse.get_focused() and not pygame.event.get_grab():
            pygame.event.set_grab(True)
            pygame.mouse.set_visible(False)
            print('grabbygrabby')
        if keys[pygame.K_ESCAPE]:
            pygame.event.set_grab(False)
            pygame.mouse.set_visible(True)

        self.current_y_rotation += -mouse_delta[0] * self.mouse_rotate_speed * delta_time

        if keys[pygame.K_a]:
            self.current_y_rotation += self.rotate_speed * delta_time
        if keys[pygame.K_s]:
            self.current_y_rotation += self.rotate_speed * delta_time
        if keys[pygame.K_UP]:
            self.current_position += self.move_speed * delta_time * forward_vec
        if keys[pygame.K_DOWN]:
            self.current_position += self.move_speed * delta_time * -forward_vec
        if keys[pygame.K_LEFT]:
            self.current_position += self.move_speed * delta_time * -right_vec
        if keys[pygame.K_RIGHT]:
            self.current_position += self.move_speed * delta_time * right_vec

        if keys[pygame.K_KP2]:
            print('reset position to center')
            self.current_position = np.array(self.center_pos)
        if keys[pygame.K_KP3]:
            print('reset rotation')
            self.current_y_rotation = 0


def black_n_white_1_to_rgb_255(img):
    img = np.stack([img] * 3, -1)
    return img / np.max(img) * 255


def rgba_to_rgb(img):
    return np.delete(img, [3], -1)


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


def unnormal_data_to_network_input(unnormalized_environment_data, black_n_white=False, flatten_images=True):
    envs_data = []
    for env in normalize_environment_data(unnormalized_environment_data):
        images, poss, rots = env
        if black_n_white:
            images = [rgba_to_black_n_white(img) for img in images]
        else:
            images = [rgba_to_rgb(img) for img in images]
        if flatten_images:
            images = [np.reshape(img, product(np.shape(img))) for img in images]
        coordinates = [network_inputs_from_coordinates_single(pos, rot) for pos, rot in zip(poss, rots)]
        envs_data.append((np.asarray(images), np.asarray(coordinates)))
    return envs_data


def return_mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def train_model_pregen(network_inputs, model_name, model_to_train, epochs=100, batch_size=None, save_model=True,
                       data_composition_multiplier=10, data_recomposition_frequency=1, log_frequency=10,
                       save_frequency = 30):
    model_dir = os.path.dirname(__file__) + '\\models'
    model_name = os.path.basename(model_name)

    checkpoint_save_path = return_mkdir(f'{model_dir}\\checkpoints') + f'\\{model_name}.checkpoint'
    final_save_path = return_mkdir(f'{model_dir}\\final') + f'\\{model_name}.hdf5'
    meta_data_save_path = return_mkdir(f'{model_dir}\\meta_data\\') + f'{model_name}.checkpoint'

    print(f'model name: {model_name}')

    for i in music.infinity():
        if i % data_recomposition_frequency == 0:
            print('composing training data')
            scrambled_image_inputs, image_inputs, coordinate_inputs = [], [], []
            total_number_of_compositions = 0
            number_of_compositions = 0
            for _ in range(data_composition_multiplier):
                for img_input_list, coordinate_input_list in network_inputs:
                    total_number_of_compositions += len(coordinate_input_list)
            for _ in range(data_composition_multiplier):
                for img_input_list, coordinate_input_list in network_inputs:
                    scrambled_image_inputs.extend(np.random.permutation(img_input_list))
                    image_inputs.extend(img_input_list)
                    coordinate_inputs.extend(coordinate_input_list)

                    number_of_compositions += len(coordinate_input_list)
                    print(f'\r{number_of_compositions}/{total_number_of_compositions} - '
                          f'{int(100 * number_of_compositions/total_number_of_compositions)}% data points composed', end='')
            print('\nconverting to numpy arrays')
            scrambled_image_inputs, image_inputs, coordinate_inputs = \
                np.asarray(scrambled_image_inputs), np.asarray(image_inputs), np.asarray(coordinate_inputs)
            print()
        print('starting training')
        model_to_train.fit([scrambled_image_inputs, coordinate_inputs], image_inputs, batch_size=batch_size,
                           epochs=epochs, callbacks=[
                                keras.callbacks.ModelCheckpoint(checkpoint_save_path, period=save_frequency, verbose=1),
                                #keras.callbacks.TensorBoard(log_dir=f'./models/tb_logs/{model_name}', write_graph=False,
            ])
        if pause_and_notify('do you want to stop training y=yes other=no', {'y': True, 'n': False}, timeout=1):
            break

    if save_model:
        print(f'saving model as {final_save_path}')
        model_to_train.save(final_save_path)
    return model_to_train


# TODO clean up and split up run method
def run(unnormalized_environment_data, model_save_file_path, model_to_generate, model_to_train=None, model_load_file_path=None, train=True,
        epochs=100, batch_size=None, data_composition_multiplier=10, log_frequency=10, save_frequency = 30, run_environment=True, black_n_white=True,
        window_size=(1200, 600), window_size_coef=1, additional_meta_data={}, save_model=True):
    '''
    Run the main Programm
    :param data_dirs: the directory containing the training data.
    :param model_save_file_path: the path where to save a model.
    :param image_dim: the dimensions of the images in the data path.
    :param model_load_file_path: the name of the model to load. None = train new model.
    :param num_samples_to_load: number of samples to load from the data dir. None = all.
    :return: Trained model
    '''
    model_generators = {
        'conv': get_convolitional_gqn_model,
        'flat': get_gqn_model,
        'multi_flat': get_multi_input_gqn_model,
    }

    input_parameters = locals()
    window_size = collections.namedtuple('Rect', field_names='x y')(
        x=int(window_size[0] * window_size_coef),
        y=int(window_size[1] * window_size_coef))

    _, max_pos_val, max_rot_val = get_max_env_data_values(unnormalized_environment_data)
    orig_img_data_shape = np.shape(unnormalized_environment_data[0][0][0])
    network_inputs = unnormal_data_to_network_input(unnormalized_environment_data, black_n_white=black_n_white,
                                                    flatten_images=False if model_to_generate == 'conv' else True)
    img_data_shape = orig_img_data_shape[0], orig_img_data_shape[1]
    if not black_n_white:
        img_data_shape = img_data_shape + (3,)

    model = model_to_train
    if not model:
        if model_load_file_path:
            model = keras.models.load_model(model_load_file_path)
        else:
            model = model_generators[model_to_generate](np.shape(network_inputs[0][0][0]),
                                                        np.shape(network_inputs[0][1][0]))
    if train:
        model = train_model_pregen(network_inputs, model_save_file_path, model, epochs=epochs,
                                   batch_size=batch_size, save_model=save_model,
                                   data_composition_multiplier=data_composition_multiplier,
                                   log_frequency=log_frequency, save_frequency = save_frequency)
    if not run_environment and not pause_and_notify('training completed, run environment? y/n'):
        return

    character_controller = CharacterController(center_pos=(0,1.5,0) / max_pos_val)
    img_drawer = ImgDrawer(window_size)

    observation_input = get_random_observation_inputs(network_inputs, 1)
    while True:
        coordinate_input = [network_inputs_from_coordinates_single(character_controller.current_position,
                                                                   character_controller.current_rotation_quaternion)]
        output_img = model.predict([observation_input, coordinate_input])
        output_img = np.reshape(output_img[0], img_data_shape)

        if black_n_white:
            output_img = black_n_white_1_to_rgb_255(output_img)
            observation_input_drawable = black_n_white_1_to_rgb_255(np.reshape(observation_input, img_data_shape))
        else:
            output_img = output_img / np.max(output_img) * 255
            observation_input_drawable = np.reshape(observation_input, img_data_shape) * 255

        img_drawer.draw_image(output_img, size=(window_size.x // 2, window_size.y))
        img_drawer.draw_image(observation_input_drawable,
                              size=(window_size.x // 4, window_size.y // 2),
                              position=(window_size.x // 2, window_size.y // 2))

        img_drawer.draw_text(f'{str(character_controller.current_position * max_pos_val)} '
                             f'max position {max_pos_val}', (10, 10))
        img_drawer.draw_text(str(character_controller.current_rotation_quaternion), (10, 50))

        img_drawer.execute()
        character_controller.movement_update()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_KP1]:
            observation_input = get_random_observation_inputs(network_inputs)
        if keys[pygame.K_SPACE]:
            pygame.quit()
            return model

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return model
    return model


models_dir = os.path.dirname(__file__) + '\\models\\'
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)
model_names_home = {
    1: 'first_large.hdf5',
    2: '2018-10-26.15-41-54.307222_super-long-run',
    3: '2018-11-01 21-15-16_(32, 32).hdf5',
    4: '2018-11-01 22-18-28_(32, 32).hdf5',
    5: '2018-11-02 01-19-11_(32, 32).checkpoint',
    }
model_names_uni = {
    -1: 'final\\date=2018-11-09_time=20-06-34-982949_name=Joe-Cruz_version=1_id=8419_idim=(32-32).hdf5',
    -2: 'final\\date=2018-11-10_time=02-04-25-643243_name=Walter-Meltzer_version=1_id=5155.hdf5',
    -3: 'final\\date=2018-11-10_time=04-01-37-084561_name=Allen-Sullivan_version=1_id=8980.hdf5'
}
TRAIN_NEW = 'train'
CONRAD = -1
HAROLD = -3
model_names = {**model_names_home, **model_names_uni}
model_names = {id: models_dir + model_name for id, model_name in zip(model_names.keys(), model_names.values())}
model_names = {'train': None, 0: None, **model_names}

data_base_dirs = [
    os.path.dirname(__file__) + '\\..\\trainingData',
]
data_dirs = {
    1: 'GQN_SimpleRoom',
    2: 'GQN_SimpleRoom_withobj',
    3: 'GQN_SimpleRoom_RandomizedObjects_2',
    4: 'GQN_SimpleRoom_nocolorchange-oneobject',
    5: 'GQN_SimpleRoom_nocolorchange-oneobject-randompositioned',
    6: 'GQN_SimpleRoom_no_variation',
    7: 'GQN_SimpleRoom_rand-sky-color',
    8: 'GQN_SimpleRoom_sphere_rand_‎pos',
    9: 'GQN_SimpleRoom_sphere_rand_‎pos+rand_sky',
}
image_resolutions = {
    8: '8x8',
    16: '16x16',
    32: '32x32',
    64: '64x64',
    128: '128x128',
    256: '256x256',
}
window_resolutions = {
    'uhd': (2400, 1200),
    'hd': (1200, 600),
}


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

spinner = Spinner()


def save_dict(save_path, dict_to_save, keys_to_skip=[]):
    with open(save_path + '.mm', 'w') as file:
        model_meta = {}
        for key, value in dict_to_save.items():
            if key not in keys_to_skip:
                model_meta[key] = value
        json.dump(model_meta, file)


FAST_DEBUG_MODE = False
# TODO create training schedule manager, to manage sequential training of networks
if __name__ == '__main__':
    data_dirs_path = get_data_dir(6, 128)
    model_name_to_load = model_names.get(-3)
    img_dims = get_img_dim_form_data_dir(data_dirs_path)

    data_dirs_arg = {'num_envs_to_load': None, 'num_data_from_env': None}
    if FAST_DEBUG_MODE:
        data_dirs_arg = {'num_envs_to_load': 10, 'num_data_from_env': 10}

    unnormalized_environment_data = \
        get_data_for_environments(data_dirs_path, **data_dirs_arg)

    model_save_path = models_dir + get_model_name(model_name_to_load)
    run_params = {
        'model_to_generate': 'conv',
        'unnormalized_environment_data': unnormalized_environment_data,
        'model_load_file_path': model_name_to_load,
        'model_save_file_path': model_save_path,
        'epochs': 1,
        'batch_size': None,
        'data_composition_multiplier': 1,
        'log_frequency': 10,
        'save_frequency': 30,
        'run_environment': True,
        'train': True,
        'black_n_white': False,
        'window_size': window_resolutions['hd'],
        'save_model': not FAST_DEBUG_MODE
    }
    print('\nparams')
    pprint.pprint(run_params, depth=1, compact=True)
    print()

    trained_model = run(**run_params)
    #save_dict(model_save_path, run_params, ['unnormalized_environment_data', 'model_to_train'])

    run_params['model_to_train'] = trained_model
    run_params['model_save_file_path'] = models_dir + \
                                            get_model_name(run_params['model_save_file_path'])
    run_params['train'] = True

    while True:
        run(**run_params)
        #save_dict(model_save_path, run_params, ['unnormalized_environment_data'])
        run_params['model_save_file_path'] = models_dir + \
                                                get_model_name(run_params['model_save_file_path'])