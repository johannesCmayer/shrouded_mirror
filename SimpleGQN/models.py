import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, UpSampling2D, MaxPooling2D, Dense, Concatenate, Flatten, Reshape
from util import product
import numpy as np


def get_gqn_encoder(picture_input_shape, coordinate_input_shape, num_layers, num_neurons_layer=1024, num_state_neurons=None, masking=True):
    num_state_neurons = num_state_neurons if num_state_neurons else num_neurons_layer
    picture_input = keras.Input(picture_input_shape, name='picture_input')
    coordinates_picture_input = keras.Input(coordinate_input_shape, name='coordinates_picture_input')

    x = Flatten()(picture_input)
    c = Flatten()(coordinates_picture_input)
    x = Concatenate()([x, c])
    if masking:
        x = keras.layers.Masking(mask_value=0.0)(x)
    for _ in range(num_layers):
        x = Dense(num_neurons_layer, 'relu', True)(x)
    output = Dense(num_state_neurons, 'relu', True)(x)
    return keras.Model([picture_input, coordinates_picture_input], output, name='gqn_encoder')


def get_gqn_conv_encoder(picture_input_shape, coordinate_input_shape, num_layers, num_neurons_layer=1024, num_state_neurons=None, masking=True):
    num_state_neurons = num_state_neurons if num_state_neurons else num_neurons_layer
    picture_input = keras.Input(picture_input_shape, name='picture_input')
    coordinates_picture_input = keras.Input(coordinate_input_shape, name='coordinates_picture_input')

    x = Conv2D(256, (2, 2), (1,1), padding='same')(picture_input)
    out_x = Conv2D(128, (3,3), (1,1), padding='same')(x)
    x = keras.layers.Add()([x, out_x])
    x = Conv2D(256, (2,2), (2,2), padding='valid')(x)

    expanded_coord = keras.backend.expand_dims(coordinates_picture_input)
    expanded_coord = keras.backend.expand_dims(expanded_coord)
    expanded_coord = keras.backend.repeat_elements(expanded_coord, x.output_shape[0], 0)
    expanded_coord = keras.backend.repeat_elements(expanded_coord, x.output_shape[1], 1)

    x = Concatenate()([x, expanded_coord])
    out_x = Conv2D(128, (3,3), (1,1), padding='same')(x)
    x = keras.layers.Add()([x, out_x])


def get_gqn_decoder(state_input_shape, coordinate_input_shape, output_dim, num_layers, num_neurons_layer=1024):
    state_input = keras.Input(state_input_shape, name='picture_input')
    coordinate_input = keras.Input(coordinate_input_shape, name='coordinate_input')
    x = Concatenate()([state_input, coordinate_input])
    for _ in range(num_layers - 1):
        x = Dense(num_neurons_layer, 'relu', True)(x)
    x = Dense(product(output_dim), 'relu', True)(x)
    predictions = Reshape(output_dim)(x)
    return keras.Model(inputs=[state_input, coordinate_input], outputs=predictions, name='gqn_decoder')


def get_multi_input_gqn_model(pictures_input_shape, coordinates_input_shape, num_input_observations, num_layers_encoder=8,
                              num_layers_decoder=8, num_neurons_per_layer=1024, num_state_neurons=1024):
    print('creating model')
    if not num_layers_decoder:
        num_layers_decoder = num_layers_encoder

    number_of_pixels = product(pictures_input_shape) * num_input_observations

    picture_input = [keras.Input(pictures_input_shape, name=f'picture_input{i}') for i in range(num_input_observations)]
    coordinates_picture_input = [keras.Input(coordinates_input_shape, name=f'coordinates_picture_input{i}') for i in range(num_input_observations)]
    querry_coordinates = keras.Input(coordinates_input_shape, name='querry_coordinates')

    encoder = get_gqn_encoder(pictures_input_shape, coordinates_input_shape, num_layers_encoder,
                              num_neurons_per_layer, num_state_neurons)

    encoded = [encoder([o, c]) for o, c in zip(picture_input, coordinates_picture_input)]
    if len(encoded) > 1:
        encoded = keras.layers.Add()(encoded)
    else:
        encoded = encoded[0]
    #max = keras.backend.max(encoded)
    #encoded /= max

    decoder = get_gqn_decoder(encoded.shape[1:], coordinates_input_shape, output_dim=pictures_input_shape,
                              num_layers=num_layers_decoder, num_neurons_layer=num_neurons_per_layer)
    decoded = decoder([encoded, querry_coordinates])

    joint_model = keras.Model(inputs=[*picture_input, *coordinates_picture_input, querry_coordinates], outputs=decoded)
    joint_model.compile('rmsprop', 'mse')
    return joint_model


def get_embedding_model(pictures_input_shape, coordinates_input_shape, num_input_observations, num_layers_encoder=8,
                              num_layers_decoder=8, num_neurons_per_layer=1024, num_state_neurons=1024):
    print('creating model')
    picture_input = [keras.Input(pictures_input_shape, name=f'picture_input{i}') for i in range(num_input_observations)]
    coordinates_picture_input = [keras.Input(coordinates_input_shape, name=f'coordinates_picture_input{i}') for i in
                                 range(num_input_observations)]

    encoded = keras.backend.variable(np.ones(coordinates_input_shape))
    querry_coordinates = keras.Input(coordinates_input_shape, name='querry_coordinates')
    decoder = get_gqn_decoder(encoded.shape[1:], coordinates_input_shape, output_dim=pictures_input_shape,
                              num_layers=num_layers_decoder, num_neurons_layer=num_neurons_per_layer)
    decoded = decoder([encoded, querry_coordinates])

    joint_model = keras.Model(inputs=[*picture_input, *coordinates_picture_input, querry_coordinates], outputs=decoded)
    joint_model.compile('rmsprop', 'mse')
    return joint_model