from sacred import Experiment

default = Experiment()

@default.config
def default_config():
    window_resolution = (1800, 900)
    image_resolution = '32x32'
    data_dir = 'Maze_3_gridCapturePointCapture'

    model_path = 'date=2019-01-16_time=21-42-23-452204_env=Maze-3-gridCapturePointCapture_name=Sidney-Arbogast_version=1_id=3801'

    data_multiplier = 1
    log_frequency = 10
    save_frequency = 30
    train = False
    black_n_white = False

    fast_debug_mode = False  # load less data and dont save model
    run_environment = True
    run_pygame = False
    udp_image_send_port = 8686

    model_to_generate = 'multi'
    num_layers_encoder = 8
    num_layers_decoder = 8
    num_neurons_per_layer = 2048
    num_state_neurons = 2048
    num_input_observations = 4
    epochs = 4
    batch_size = 32
