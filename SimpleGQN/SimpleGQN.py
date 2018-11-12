import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dense, Concatenate, Flatten, Reshape
import numpy as np
from scipy import misc
import glob
import matplotlib.pyplot as plt
from util import ImgDrawer, Spinner, AsyncKeyChecker, CharacterController
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
import pprint
import math
import logging
import functools
import multiprocessing
import music
import gqn
import yaml


print(f'started execution at {datetime.datetime.now()}')

def convert_to_valid_os_name(string, substitute_char='-'):
    return replace_multiple(string, '\\ / : * ? " < > |'.split(' '), substitute_char)


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


def get_gqn_encoder(picture_input_shape, num_layers=6, num_neurons_layer=1024, num_state_neurons=None, masking=True):
    num_state_neurons = num_state_neurons if num_state_neurons else num_neurons_layer
    picture_input = keras.Input(picture_input_shape, name='picture_input')

    x = Flatten()(picture_input)
    if masking:
        x = keras.layers.Masking(mask_value=0.0)(x)
    for _ in range(num_layers):
        x = Dense(num_neurons_layer, 'relu', True)(x)
    output = Dense(num_state_neurons, 'relu', True)(x)
    return keras.Model(picture_input, output)


def get_gqn_decoder(state_input_shape, coordinate_input_shape, output_dim, num_layers=6, num_neurons_layer=1024):
    state_input = keras.Input(state_input_shape, name='picture_input')
    coordinate_input = keras.Input(coordinate_input_shape, name='coordinate_input')
    x = Concatenate()([state_input, coordinate_input])
    for _ in range(num_layers - 1):
        x = Dense(num_neurons_layer, 'relu', True)(x)
    x = Dense(product(output_dim), 'relu', True)(x)
    predictions = Reshape(output_dim)(x)
    return keras.Model(inputs=[state_input, coordinate_input], outputs=predictions)


# TODO make it so it can take non flat input (test if it already can)
def get_gqn_model(picture_input_shape, coordinates_input_shape, num_layers_encoder=3, num_layers_decoder=None,
                  num_neurons_per_layer=1024, num_state_neurons=1024, compile=True):
    print('creating model')
    if not num_layers_decoder:
        num_layers_decoder = num_layers_encoder

    # TODO Add ability to add up multiple observations to the latent representation
    number_of_data_points = product(picture_input_shape)

    picture_input = keras.Input(picture_input_shape, name='picture_input')
    coordinates_input = keras.Input(coordinates_input_shape, name='coordinates_input')

    x = picture_input
    if len(picture_input_shape) > 1:
        x = Flatten()(picture_input)

    for _ in range(num_layers_encoder):
        x = Dense(num_neurons_per_layer, 'relu', True)(x)
    x = Dense(num_state_neurons, 'relu', True)(x)

    x = Concatenate()([x, coordinates_input])
    for _ in range(num_layers_decoder):
        x = Dense(num_neurons_per_layer, 'relu', True)(x)
    x = Dense(number_of_data_points, 'relu', True)(x)
    predictions = Reshape(picture_input_shape)(x)

    joint_model = keras.Model(inputs=[picture_input, coordinates_input], outputs=predictions)
    if compile:
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

    input_img = keras.Input(picture_input_shape, name='picture_input')
    coordinates_input = keras.Input(coordinates_input_shape, name='coordinates_input')

    if any(shape % downsampled_res != 0 for shape in picture_input_shape[:2]) \
            or picture_input_shape[0] != picture_input_shape[1]:
        raise ValueError(f'Picture input shape needs to be multiple of 16 in x and y but is {picture_input_shape[:2]}')
    num_of_sampeling_steps = int(math.log2(picture_input_shape[0]) - math.log2(downsampled_res))

    x = input_img
    for _ in range(num_of_sampeling_steps):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)

    x = get_gqn_model(x.shape[1:], coordinates_input_shape, compile=False)([x, coordinates_input])

    for _ in range(num_of_sampeling_steps):
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
    if x.shape != picture_input_shape:
        x = Conv2D(3, (3, 3), activation='relu', padding='same')(x)
    decoded = x

    joint_model = keras.Model(inputs=[input_img, coordinates_input], outputs=decoded)
    joint_model.compile('adam', 'mse')

    joint_model.summary()
    return joint_model


def get_multi_input_gqn_model(pictures_list_input_shape, num_input_observations, coordinates_input_shape, num_layers_encoder=6,
                              num_layers_decoder=None, num_neurons_per_layer=1024, num_state_neurons=1024):
    print('creating model')
    if not num_layers_decoder:
        num_layers_decoder = num_layers_encoder

    number_of_pixels = product(num_input_observations * pictures_list_input_shape)

    picture_input = [keras.Input(pictures_list_input_shape, name=f'picture_input{i}') for i in range(num_input_observations)]
    coordinate_input = keras.Input(coordinates_input_shape, name='coordinate_input')

    encoder = get_gqn_encoder(pictures_list_input_shape)

    encoded = [encoder(o) for o in picture_input]
    if len(encoded) > 1:
        encoded = keras.layers.Add()(encoded)
    else:
        encoded = encoded[0]

    decoded = get_gqn_decoder(encoded.shape[1:], coordinates_input_shape, output_dim=pictures_list_input_shape)([encoded, coordinate_input])

    joint_model = keras.Model(inputs=[*picture_input, coordinate_input], outputs=decoded)
    joint_model.compile('rmsprop', 'mse')
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


def parse_path_to_params(path, params_seperator='_', key_val_seperator='='):
    previous_name = replace_multiple(os.path.basename(path), ['.hdf5', '.checkpoint', '-hdf5', '-checkpoint'], '')
    return {key: val for key, val in [e.split(key_val_seperator) for e in previous_name.split(params_seperator)]}


def time_int_from_param_dict(params):
    return int((params['day'] + params['time']).replace('-', ''))

model_file_locations = os.path.dirname(__file__) + '\\models'
model_file_locations = [model_file_locations + p for p in ['\\final', '\\checkpoints']]

def generate_model_name(previous_name=None):
    if previous_name:
        param_dict = parse_path_to_params(previous_name)
        id = param_dict['id']
        name = param_dict['name']
        version = 0
        for d in model_file_locations:
            for p in glob.glob(f'{d}\\*{name}*{id}*'):
                exisiting_param_dict = parse_path_to_params(p)
                version = max(int(exisiting_param_dict['version']), version)
        version += 1
        return get_unique_model_save_name(name, version, id)
    else:
        return get_new_unique_model_save_name()


def get_model_load_path(name, id):
    newest = ''
    highest_version = 0
    for d in model_file_locations:
        for p in glob.glob(f'{d}\\*{name}*{id}*'):
            exisiting_param_dict = parse_path_to_params(p)
            version_is_higher = int(exisiting_param_dict['version']) >= highest_version
            if version_is_higher or version_is_higher and newest != '' \
                    and time_int_from_param_dict(exisiting_param_dict) >= time_int_from_param_dict(newest):
                newest = p
    if newest != '':
        return newest
    else:
        raise Exception(f'model file for {name} {id} not found')


def remove_extension(path, extension='hdf5'):
    if path.split('.')[-1] == extension:
        return ''.join(path.split('.')[0:-2])
    return path


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


def generate_data(network_inputs, num_input_observations, data_composition_multiplier):
    print('composing training data')
    scrambled_image_inputs_list, image_inputs, coordinate_inputs = [[] for _ in
                                                                    range(num_input_observations)], [], []
    total_number_of_compositions = 0
    number_of_compositions = 0
    for _ in range(data_composition_multiplier):
        for img_input_list, coordinate_input_list in network_inputs:
            total_number_of_compositions += len(coordinate_input_list)
    for _ in range(data_composition_multiplier):
        for img_input_list, coordinate_input_list in network_inputs:
            for sub_list in scrambled_image_inputs_list:
                #sub_list.extend(np.asarray(np.random.permutation(img_input_list)))
                sub_list.extend(np.asarray([x * random.randint(0, 1) for x in np.random.permutation(img_input_list)]))
            image_inputs.extend(img_input_list)
            coordinate_inputs.extend(coordinate_input_list)

            number_of_compositions += len(coordinate_input_list)
            print(f'\r{number_of_compositions}/{total_number_of_compositions} - '
                  f'{int(100 * number_of_compositions/total_number_of_compositions)}% data points composed', end='')
    print('\nconverting to numpy arrays')
    scrambled_image_inputs_list = np.asarray([np.asarray(x) for x in scrambled_image_inputs_list])
    image_inputs = np.asarray(image_inputs)
    coordinate_inputs = np.asarray(coordinate_inputs)
    print('completed conversion')
    return scrambled_image_inputs_list, image_inputs, coordinate_inputs


def train_model_pregen(network_inputs, num_input_observations, model_name, model_to_train, epochs=100, batch_size=None, save_model=True,
                       data_composition_multiplier=10, data_recomposition_frequency=1, log_frequency=10,
                       save_frequency = 30):
    model_dir = os.path.dirname(__file__) + '\\models'
    model_name = os.path.basename(model_name)

    checkpoint_save_path = return_mkdir(f'{model_dir}\\checkpoints') + f'\\{model_name}.checkpoint'
    final_save_path = return_mkdir(f'{model_dir}\\final') + f'\\{model_name}.hdf5'
    meta_data_save_path = return_mkdir(f'{model_dir}\\meta_data\\') + f'{model_name}.modelmeta'

    print(f'model name: {model_name}')
    with AsyncKeyChecker('q') as kc:
        for i in music.infinity():
            if i % data_recomposition_frequency == 0:
                scrambled_image_inputs_list, image_inputs, coordinate_inputs = \
                    generate_data(network_inputs, num_input_observations, data_composition_multiplier)
            print('starting training')
            model_to_train.fit([*scrambled_image_inputs_list, coordinate_inputs], image_inputs, batch_size=batch_size, verbose=1,
                               epochs=epochs, callbacks=[
                                    keras.callbacks.ModelCheckpoint(checkpoint_save_path, period=save_frequency, verbose=1),
                                    #keras.callbacks.TensorBoard(log_dir=f'./models/tb_logs/{model_name}', write_graph=False,
                ])
            if kc.key_was_pressed():
                break
    if save_model:
        print(f'saving model as {final_save_path}')
        model_to_train.save(final_save_path)
        with open(meta_data_save_path, 'w') as f:
            json.dump(model_to_train.to_json(), f)
    winsound.Beep(280, 300)
    return model_to_train


def mask_observation_inputs(obs_inputs, num_to_mask):
    return [x * (1 if i < num_to_mask else 0) for i, x in enumerate(obs_inputs)]


# TODO clean up and split up run method
def run(unnormalized_environment_data, num_input_observations, model_save_file_path, model_to_generate,
        model_to_train=None, model_load_file_path=None, train=True, epochs=100, batch_size=None,
        data_multiplier=10, log_frequency=10, save_frequency = 30, run_environment=True, black_n_white=True,
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
        'multi': get_multi_input_gqn_model,
    }

    input_parameters = locals()
    window_size = collections.namedtuple('Rect', field_names='x y')(
        x=int(window_size[0] * window_size_coef),
        y=int(window_size[1] * window_size_coef))

    _, max_pos_val, max_rot_val = get_max_env_data_values(unnormalized_environment_data)
    orig_img_data_shape = np.shape(unnormalized_environment_data[0][0][0])
    network_inputs = unnormal_data_to_network_input(unnormalized_environment_data, black_n_white=black_n_white,
                                                    flatten_images=False if
                                                    any([model_to_generate == x for x in ['conv', 'multi']]) else True)
    img_data_shape = orig_img_data_shape[0], orig_img_data_shape[1]
    if not black_n_white:
        img_data_shape = img_data_shape + (3,)

    model = model_to_train
    if not model:
        if model_load_file_path:
            model = keras.models.load_model(model_load_file_path)
        else:
            model = model_generators[model_to_generate](np.shape(network_inputs[0][0][0]), num_input_observations,
                                                        np.shape(network_inputs[0][1][0]))
    model.summary()
    if train:
        model = train_model_pregen(network_inputs, num_input_observations, model_save_file_path, model, epochs=epochs,
                                   batch_size=batch_size, save_model=save_model,
                                   data_composition_multiplier=data_multiplier,
                                   log_frequency=log_frequency, save_frequency = save_frequency)
    if not run_environment:
        return

    character_controller = CharacterController(center_pos=(0,1.5,0) / max_pos_val)
    img_drawer = ImgDrawer(window_size)

    def get_random_observation_input_list():
        return [np.asarray([x]) for x in get_random_observation_inputs(network_inputs, num_input_observations)]

    num_unmask_inputs = 1
    observation_inputs = get_random_observation_input_list()
    masked_observation_inputs = mask_observation_inputs(observation_inputs, num_unmask_inputs)
    while True:
        coordinate_input = np.asarray([network_inputs_from_coordinates_single(character_controller.current_position,
                                                                   character_controller.current_rotation_quaternion)])
        output_img = model.predict([*masked_observation_inputs, coordinate_input])
        output_img = np.reshape(output_img[0], img_data_shape)

        if black_n_white:
            output_img = black_n_white_1_to_rgb_255(output_img)
            #observation_input_drawable = black_n_white_1_to_rgb_255(np.reshape(observation_inputs, img_data_shape))
        else:
            output_img = 255 * output_img / np.max(output_img)
            observation_inputs_drawable = [np.reshape(obs_input, img_data_shape) * 255 for obs_input in masked_observation_inputs]

        img_drawer.draw_image(output_img, size=(window_size.x // 2, window_size.y))
        for i, o in enumerate(observation_inputs_drawable):
            size = (window_size.x // 8, window_size.y // 4)
            origin_pos = (window_size.x // 2, 0)
            image_columns = 4
            offset = (size[0] * (i % image_columns), size[1]*(i // image_columns))
            pos = (origin_pos[0] + offset[0], origin_pos[1] + offset[1])
            img_drawer.draw_image(o, size=size, position=pos)

        text_orig_pos = (window_size.x // 2 + 10, window_size.y // 2 + 10)
        img_drawer.draw_text_auto_pos(f'{str(character_controller.current_position * max_pos_val)} '
                             f'max position {max_pos_val}', text_orig_pos)
        img_drawer.draw_text_auto_pos(str(character_controller.current_rotation_quaternion), text_orig_pos)
        img_drawer.draw_text_auto_pos(os.path.basename(model_save_file_path), text_orig_pos)

        img_drawer.execute()
        character_controller.movement_update()

        for event in pygame.event.get():
            if event.type == pygame.QUIT or event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                pygame.quit()
                return model
            if event.type == pygame.KEYDOWN and event.key == pygame.K_KP_MINUS:
                num_unmask_inputs = max(num_unmask_inputs - 1, 0)
                masked_observation_inputs = mask_observation_inputs(observation_inputs, num_unmask_inputs)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_KP_PLUS:
                num_unmask_inputs = min(num_unmask_inputs + 1, num_input_observations)
                masked_observation_inputs = mask_observation_inputs(observation_inputs, num_unmask_inputs)
            if event.type == pygame.KEYDOWN and event.key == pygame.K_KP_ENTER:
                observation_inputs = get_random_observation_input_list()
                masked_observation_inputs = mask_observation_inputs(observation_inputs, num_unmask_inputs)





def get_data_dir(data_dir_name, resolution_string):
    dir = ''
    for base_dir in data_base_dirs:
        dir = (f'{base_dir}\\{data_dir_name}\\{resolution_string}')
        if os.path.isdir(dir):
            print(f'Found data in: {dir}')
            return dir
        else:
            exsting_dir = os.path.dirname(dir)
            while not os.path.isdir(exsting_dir) and exsting_dir != '':
                exsting_dir = os.path.dirname(exsting_dir)
            print(f'Data not found in:      {dir} \n'
                  f'nearest existing dir is {exsting_dir}')
    raise OSError('None of specified data base dirs exist.')


def get_img_dim_form_data_dir(dir):
    dims = dir.split('\\')[-1].split('x')
    return int(dims[0]), int(dims[1])


def save_dict(save_path, dict_to_save, keys_to_skip=[]):
    with open(save_path + '.mm', 'w') as file:
        model_meta = {}
        for key, value in dict_to_save.items():
            if key not in keys_to_skip:
                model_meta[key] = value
        json.dump(model_meta, file)


def specs():
    specs.x = 'hello'
    with open(os.path.dirname(__file__) + '\\run_config.yaml') as f:
        return yaml.load(f)

models_dir = os.path.dirname(__file__) + '\\models\\'
if not os.path.isdir(models_dir):
    os.mkdir(models_dir)

data_base_dirs = [
    'C:\\trainingData',
    os.path.abspath(os.path.dirname(__file__) + '\\..\\trainingData'),
]

spinner = Spinner()

# TODO create training schedule manager, to manage sequential training of networks
if __name__ == '__main__':
    data_spec, param_spec, run_spec = specs()['data_spec'], specs()['param_spec'], specs()['run_spec']

    data_dirs_path = get_data_dir(data_spec['data_dir'], data_spec['image_resolution'])
    model_load_path = data_spec['model_load_path']
    if model_load_path:
        name_parameters = parse_path_to_params(model_load_path)
        model_load_path = get_model_load_path(name_parameters['name'], name_parameters['id'])
    data_dirs_arg = {'num_envs_to_load': None, 'num_data_from_env': None}
    if run_spec['fast_debug_mode']:
        data_dirs_arg = {'num_envs_to_load': 10, 'num_data_from_env': 10}

    unnormalized_environment_data = \
        get_data_for_environments(data_dirs_path, **data_dirs_arg)

    model_save_path = models_dir + generate_model_name(model_load_path)

    run_params = {
        'unnormalized_environment_data': unnormalized_environment_data,
        'model_load_file_path': model_load_path,
        'model_save_file_path': model_save_path,
        'window_size': data_spec['window_resolution'],
        'save_model': not run_spec['fast_debug_mode'],
        **param_spec
    }
    print('\nparams')
    pprint.pprint(run_params, depth=1, compact=True)
    print()

    trained_model = run(**run_params)
    #save_dict(model_save_path, run_params, ['unnormalized_environment_data', 'model_to_train'])

    run_params['model_to_train'] = trained_model
    run_params['model_save_file_path'] = models_dir + \
                                         generate_model_name(run_params['model_save_file_path'])
    run_params['train'] = True

    while True:
        run(**run_params)
        #save_dict(model_save_path, run_params, ['unnormalized_environment_data'])
        run_params['model_save_file_path'] = models_dir + \
                                             generate_model_name(run_params['model_save_file_path'])