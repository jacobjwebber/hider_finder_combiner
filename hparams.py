import argparse
import torch

hp = {
    # File locations etc
    'dataset_dir' : '/disk/scratch/s1116548/datasets',

    # File locations for vocoder
    'vocoder_wavs' : '/home/jjw/datasets/vocoder_wavs',
    'vocoder_mels' : '/home/jjw/datasets/vocoder_mels',

    # DSP
    'sr' : 22050,
    'n_fft' : 1024,
    'num_freq' : 1025,
    'win_length' : 1024,
    'num_mels' : 80,
    'hop_length' : 256, # in samples
    'fmin' : 0,
    'fmax' : 8000,
    'fmax_for_loss' : None,
    # Normalize db scale mel specs between these values
    'min_level_db' : -100,
    'max_level_db' : 20,

    ## Take the mean of the cv and keep constant throughout utt
    'use_mean' : False, #TODO
    'cv_twice' : False,

    # Datasets

    'epochs' : 5,
    ## Alas, only support batch size of one for now. Add stacker later
    'train_batch_size' : 1,
    'valid_batch_size' : 1,
    'validation_split' : 0.004,
    # Tiny dataset for debugging runs
    'tiny_dataset' : False,
    'validation_interval' : 200, # No of steps between validation calcs
    'annealing_rate' : 0.95,
    'clip' : 0.5,

    ## Legacy dataset params TODO replace with proper torchaudio dataset
    'wav_dir' : '../qual_params/data/wav/',
    'mel_dir' : '../hfc_2/mel',
    'save_path' : '',
    ## In the future will use something clever like this TODO NOT USED YET

    # Compute
    ## Use cuda if it exists, can disable with argument
    'use_cuda' : torch.cuda.is_available(),

    # Dimensions
    'input_width' : 80, # Input mel/spect dimension
    'hidden_size' : 80, # Size of output from hider/input to combiner
    'cv_bins' : 110, # Number of bins to quantize cv into

    'cv' : 'f0', # The control variable!

    # Networks
    'beta' : 600., # See paper
    'denoising' : 0.00, # Randomly dropout in hidden embedding? No

    ## Hider
    'h_rnn_size' : 800, # rnn hidden layer size
    'h_layers' : 3, # Number of layers in hider RNN
    'h_drop' : 0.01, # Dropout in hider

    ## Finder
    'f_rnn_size' : 100,
    'f_layers' : 2,
    'f_drop' : 0.01,

    ## Combiner
    'c_rnn_size' : 1200,
    'c_layers' : 3,
    'c_drop' : 0.01,

    # Learning rate for two categories of network
    'g_lr' : 0.0001,
    'f_lr' : 0.004,

    'f0_range' : (60., 250.),
    'CoG_range' : (100., 2500.),
    'ST_range' : (-70., 40.),
    'I_range' : (0., 17)
}


def get_hparams():
    parser = argparse.ArgumentParser(description='Process updated hparams that differ from definition in hparams.py')
    for key, value in hp.items():
        parser.add_argument('--' + key, type=type(value), default=value)

    args = parser.parse_args()
    ranges = {
        'f0' : args.f0_range,
        'CoG' : args.CoG_range,
        'ST' : args.ST_range,
        'I' : args.I_range
    }
    args.f0_range = ranges['f0']
    args.cv_range = ranges[args.cv]

    return args

