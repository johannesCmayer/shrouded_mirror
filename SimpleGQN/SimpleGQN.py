import os, io
import tensorflow as tf
import numpy as np
import glob
import ntpath
import pygame
import datetime
import random
import collections
import winsound
import names
import json
import pprint
import math
import functools
import gc
import yaml
import socket
from PIL import Image
from tensorflow import keras
from tensorflow.python.tools import freeze_graph
from scipy import misc
from typing import Dict

import music
from util import (
    ImgDrawer,
    Spinner,
    AsyncKeyChecker,
    CharacterController,
    product
)
from models import (
    get_multi_input_gqn_model,
    simple_conv_model,
    get_latent_variable_gqn_model,
)


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


def get_unique_model_save_name(name, version, id, env_name):
    def fm(val):
        string = replace_multiple(str(val), ['=', ', ', ' ', '.'], '-')
        return convert_to_valid_os_name(string, substitute_char='-')
    date = datetime.datetime.now().date()
    time = datetime.datetime.now().time()
    env_name = env_name.replace('_', '-')
    d = {'date': date, 'time': time, 'env': env_name, 'name': name, 'version': version, 'id': id}
    return functools.reduce(lambda acc, val: f'{acc}_{val[0]}={fm(val[1])}', d.items(), '')[1:]


def get_new_unique_model_save_name(env_name):
    name = names.get_full_name()
    id = random.randint(1000, 10000)
    return get_unique_model_save_name(name, 1, id, env_name)


def parse_path_to_params(path, params_seperator='_', key_val_seperator='='):
    previous_name = replace_multiple(os.path.basename(path), ['.hdf5', '.checkpoint', '-hdf5', '-checkpoint'], '')
    return {key: val for key, val in [e.split(key_val_seperator) for e in previous_name.split(params_seperator)]}


def time_int_from_param_dict(params):
    return int((params['day'] + params['time']).replace('-', ''))

model_file_locations = os.path.dirname(__file__) + '\\models'
model_file_locations = [model_file_locations + p for p in ['\\final', '\\checkpoints']]

def generate_model_name(env_name, previous_name=None):
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
        return get_unique_model_save_name(name, version, id, env_name)
    else:
        return get_new_unique_model_save_name(env_name)


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

class EnvDataNormalizer:
    def __init__(self, unnormalized_environment_data):
        self.img_max, self.pos_max, self.rot_max = self._get_max_env_data_values(unnormalized_environment_data)

    def _get_max_env_data_values(self, environment_data):
        max_img, max_pos, max_rot = 0, 0, 0
        for e in environment_data:
            im, pos, rot = e
            max_img = max(np.max(im), max_img)
            max_pos = max(np.max(pos), max_pos)
            max_rot = max(np.max(rot), max_rot)
        return max_img, max_pos, max_rot

    def normalize_envirenment_data_sigle(self, pos, rot):
        return pos / self.pos_max, rot / self.rot_max

    def normalize_environment_data(self, environment_data):
        return [(img / self.img_max, pos / self.pos_max, rot / self.rot_max) for img, pos, rot in environment_data]

    def unnormal_data_to_network_input(self, unnormalized_environment_data, black_n_white=False, flatten_images=True):
        envs_data = []
        for env in self.normalize_environment_data(unnormalized_environment_data):
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


def generate_data(network_inputs, num_input_observations, data_composition_multiplier, num_of_samples=None):
    print('composing training data')
    images_1, coordinates_1 = [[] for _ in range(num_input_observations)], \
                              [[] for _ in range(num_input_observations)]
    image_2, coordinates_2 = [], []

    total_number_of_compositions = 0
    number_of_compositions = 0
    for img_input_list, coordinate_input_list in network_inputs:
        total_number_of_compositions += len(coordinate_input_list)
    total_number_of_compositions *= data_composition_multiplier
    abort = False
    for _ in range(math.ceil(data_composition_multiplier)):
        for img_input_list, coordinate_input_list in network_inputs:
            for img_list, coord_list in zip(images_1, coordinates_1):
                take = random.randint(0, 1)
                idx = np.random.permutation(np.arange(coordinate_input_list.shape[0]))
                img_list.extend(np.asarray([x * take for x in np.take(img_input_list, idx, axis=0)]))
                coord_list.extend(np.asarray([x * take for x in np.take(coordinate_input_list, idx, axis=0)]))
            image_2.extend(img_input_list)
            coordinates_2.extend(coordinate_input_list)

            number_of_compositions += len(coordinate_input_list)
            if number_of_compositions >= total_number_of_compositions:
                abort = True
                break
            print(f'\r{number_of_compositions}/{total_number_of_compositions} - '
                  f'{int(100 * number_of_compositions/total_number_of_compositions)}% data points composed', end='')
        if abort:
            break
    print('\nconverting to numpy arrays')
    images_1 = [np.asarray(x) for x in images_1]
    coordinates_1 = [np.asarray(x) for x in coordinates_1]
    image_2 = np.asarray(image_2)
    coordinates_2 = np.asarray(coordinates_2)
    print('completed conversion')
    return images_1, coordinates_1, image_2, coordinates_2


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph


def train_model_pregen(network_inputs, num_input_observations, model_name, model_to_train, epochs=100, batch_size=None, save_model=True,
                       data_composition_multiplier=10, data_recomposition_frequency=1, log_frequency=10,
                       save_frequency = 30):
    model_dir = os.path.dirname(__file__) + '\\models'
    model_name = os.path.basename(model_name)

    checkpoint_save_path = return_mkdir(f'{model_dir}\\checkpoints') + f'\\{model_name}.checkpoint'
    final_save_path = return_mkdir(f'{model_dir}\\final') + f'\\{model_name}.hdf5'
    meta_data_save_path = return_mkdir(f'{model_dir}\\meta_data\\') + f'{model_name}.modelmeta'
    pb_save_path_base_dir = return_mkdir(f'{model_dir}\\pb_models\\')
    pb_model_name = f'{model_name}' + '.bytes'

    print(f'model name: {model_name}')
    with AsyncKeyChecker('q') as kc:
        for i in music.infinity():
            if i % data_recomposition_frequency == 0:
                images_1, image_coordinates_1, images_2, image_coordinates_2 = \
                    generate_data(network_inputs, num_input_observations, data_composition_multiplier)
            print('starting training')
            model_to_train.fit([*images_1, *image_coordinates_1, image_coordinates_2], images_2, batch_size=batch_size, verbose=1,
                               epochs=epochs, callbacks=[
                                    keras.callbacks.ModelCheckpoint(checkpoint_save_path, period=save_frequency, verbose=1),
                                    #keras.callbacks.TensorBoard(log_dir=f'./models/tb_logs/{model_name}', write_graph=False,
                ])
            for e in images_1, image_coordinates_1, images_2, image_coordinates_2:
                del(e)
            gc.collect()
            if kc.key_was_pressed():
                break
    if save_model:
        print(f'saving model as {final_save_path}')
        model_to_train.save(final_save_path)
        with open(meta_data_save_path, 'w') as f:
            json.dump(model_to_train.to_json(), f)

        f_graph = freeze_session(keras.backend.get_session(), output_names=[out.op.name for out in model_to_train.outputs])
        tf.train.write_graph(f_graph, pb_save_path_base_dir, pb_model_name, as_text=False)
    winsound.Beep(280, 300)
    return model_to_train


def mask_observation_inputs(obs_inputs, num_to_mask):
    l = []
    for i, (img, coord) in enumerate(zip(*obs_inputs)):
        take = 1 if i < num_to_mask else 0
        l.append(([img * take], [coord * take]))
    return list(zip(*l))


# TODO clean up and split up run method
def run(unnormalized_environment_data, num_input_observations, model_save_file_path, model_to_generate,
        model_to_train=None, model_load_file_path=None, train=True, epochs=100, batch_size=None,
        data_multiplier=10, log_frequency=10, save_frequency=30, run_environment=True, black_n_white=True,
        window_size=(1200, 600), window_size_coef=1, additional_meta_data={}, save_model=True, fast_debug_mode=False,
        run_pygame=False, udp_image_send_port=None, num_layers_encoder=6, num_layers_decoder=6,
        num_neurons_per_layer=1024, num_state_neurons=1024):
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
        'multi': get_multi_input_gqn_model,
        'conf': simple_conv_model,
        'lv_gqn': get_latent_variable_gqn_model,
    }

    input_parameters = locals()
    window_size = collections.namedtuple('Rect', field_names='x y')(
        x=int(window_size[0] * window_size_coef),
        y=int(window_size[1] * window_size_coef))

    normalizer = EnvDataNormalizer(unnormalized_environment_data)
    orig_img_data_shape = np.shape(unnormalized_environment_data[0][0][0])
    network_inputs = normalizer.unnormal_data_to_network_input(unnormalized_environment_data, black_n_white=black_n_white,
                                                    flatten_images=False) #if
                                                    #any([model_to_generate == x for x in ['conv', 'multi']]) else True)
    img_data_shape = orig_img_data_shape[0], orig_img_data_shape[1]
    if not black_n_white:
        img_data_shape = img_data_shape + (3,)

    model = model_to_train
    if not model:
        if model_load_file_path:
            model = keras.models.load_model(model_load_file_path)
        else:
            model = model_generators[model_to_generate](
                np.shape(network_inputs[0][0][0]), np.shape(network_inputs[0][1][0]), num_input_observations,
                num_layers_encoder=6, num_layers_decoder=6, num_neurons_per_layer=2048, num_state_neurons=1024)
    model.summary()
    if train:
        model = train_model_pregen(network_inputs, num_input_observations, model_save_file_path, model, epochs=epochs,
                                   batch_size=batch_size, save_model=save_model,
                                   data_composition_multiplier=data_multiplier, log_frequency=log_frequency,
                                   save_frequency = save_frequency, num_state_neurons=num_state_neurons)
    if not run_environment:
        return

    def get_random_observation_input_list():
        env_idx = random.choice(np.arange(len(network_inputs)))
        env = network_inputs[env_idx]
        obs_idxs = np.random.permutation(np.arange(len(env[0])))
        return np.take(env[0], obs_idxs[:num_input_observations], axis=0), \
               np.take(env[1], obs_idxs[:num_input_observations], axis=0)

    num_unmask_inputs = 1
    observation_inputs = get_random_observation_input_list()
    masked_observation_inputs = mask_observation_inputs(observation_inputs, num_unmask_inputs)

    def get_unity_position(env_data_normalizer):
        UDP_IP = '127.0.0.1'
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(10)
        sock.bind((UDP_IP, 9797))
        pos = (0,0,0)
        rot = (1,0,0,0)
        while True:
            try:
                data, _ = sock.recvfrom(1024)
                data = data.decode('UTF-8')
                pos, rot = data.split('_')
                pos = [float(x) for x in pos.split(', ')]
                rot = [float(x) for x in rot.split(', ')]
                break
            except socket.timeout:
                print('socket timed out, no coordinates received, restarting receive')
        pos, rot = env_data_normalizer.normalize_envirenment_data_sigle(pos, rot)
        return pos, rot

    with AsyncKeyChecker("'") as ac:
        if (run_pygame):
            img_drawer = ImgDrawer(window_size)
            #character_controller = CharacterController(center_pos=(0, 1.5, 0) / max_pos_val)
        for i in music.infinity():
            if ac.key_was_pressed:
                print('async keychecker triggered')

            if (run_pygame):
                print('Character controller needs to be updated new rot format')
                #pos, rot = character_controller.current_position, character_controller.current_rotation_quaternion
            else:
                pos, rot = get_unity_position(normalizer)

            coordinate_input = np.asarray([network_inputs_from_coordinates_single(pos, rot)])
            output_img = model.predict([*masked_observation_inputs[0], *masked_observation_inputs[1], coordinate_input])
            output_img = np.reshape(output_img[0], img_data_shape)

            if black_n_white:
                output_img = black_n_white_1_to_rgb_255(output_img)
                #observation_input_drawable = black_n_white_1_to_rgb_255(np.reshape(observation_inputs, img_data_shape))
            else:
                output_img = 255 * output_img / np.max(output_img)
                image_inputs_drawable = [np.reshape(obs_input, img_data_shape) * 255 for obs_input in masked_observation_inputs[0]]

            def get_jpeg_bytes():
                im = Image.fromarray(np.uint8(output_img))
                buffer = io.BytesIO()
                im.save(buffer, 'jpeg')
                return buffer.getvalue()

            def send_udp_local(msg, port):
                UDP_IP = '127.0.0.1'
                MESSAGE = msg
                sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                sock.sendto(MESSAGE, (UDP_IP, port))

            send_udp_local(get_jpeg_bytes(), udp_image_send_port)

            if run_pygame:
                img_drawer.draw_image(output_img, size=(window_size.x // 2, window_size.y))
                for i, o in enumerate(image_inputs_drawable):
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
    data_spec, param_spec, run_spec, network_spec = specs()['data_spec'], specs()['param_spec'], specs()['run_spec'], specs()['network_spec']

    data_dirs_path = get_data_dir(data_spec['data_dir'], data_spec['image_resolution'])
    model_load_path = data_spec['model_load_path']
    if model_load_path:
        name_parameters = parse_path_to_params(model_load_path)
        model_load_path = get_model_load_path(name_parameters['name'], name_parameters['id'])
    data_dirs_arg = {'num_envs_to_load': None, 'num_data_from_env': None}
    if run_spec['fast_debug_mode']:
        data_dirs_arg = {'num_envs_to_load': 100, 'num_data_from_env': 100}

    unnormalized_environment_data = \
        get_data_for_environments(data_dirs_path, **data_dirs_arg)

    env_name = data_spec['data_dir']
    model_save_path = models_dir + generate_model_name(env_name, model_load_path)

    run_params = {
        'unnormalized_environment_data': unnormalized_environment_data,
        'model_load_file_path': model_load_path,
        'model_save_file_path': model_save_path,
        'window_size': data_spec['window_resolution'],
        'save_model': not run_spec['fast_debug_mode'],
        **param_spec,
        **run_spec,
        **network_spec,
    }
    print('\nparams')
    pprint.pprint(run_params, depth=1, compact=True)
    print()

    trained_model = run(**run_params)

    run_params['model_to_train'] = trained_model
    run_params['model_save_file_path'] = models_dir + \
                                         generate_model_name(env_name, run_params['model_save_file_path'])
    run_params['train'] = True

    while True:
        run(**run_params)
        run_params['model_save_file_path'] = models_dir + \
                                             generate_model_name(env_name, run_params['model_save_file_path'])