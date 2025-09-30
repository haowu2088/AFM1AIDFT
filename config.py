"""
This file is used to load YAML file of the project and transform it to Dict object of Python.
"""

from argparse import ArgumentParser    # parsing command-line arguments, used in scripts intended to have a command-line interface.
import os
import os.path as osp
import pprint

import yaml   # for YAML file parsing and emitting, YAML files are commonly used for configuration files.
import numpy as np
import torch

# Wu: specify which YAML file to use for a run
def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--file', dest='filename', required=True)
    return parser

#  Loads a YAML file into a Python dictionary
def load_config(yaml_filename):
    if os.path.exists(yaml_filename):
        with open(yaml_filename, 'r', encoding='utf-8') as stream:
            content = yaml.load(stream, Loader=yaml.FullLoader)
        return content
    else:
        print('config file don\'t exist!')
        exit(1)

# ensure reproducibility
def set_random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Ensures that all necessary directories specified in the configuration exist, and if not, creates them. 
# This is useful for organizing output files, logs, checkpoints, etc.
def config_mkdir(config, dir_keys):
    for key in dir_keys:
        if key not in config:
            continue
        path = config[key].rsplit('/', 1)[0]
        if not osp.exists(path):
            os.makedirs(path)

# Modifies paths in the configuration using the experiment_name variable, 
# allowing for dynamic path adjustments based on the experiment being run.
# def update_path(config, keys):
#     for key in keys:
#         if key in config:
#             config[key] = config[key].format(config['experiment_name'])

# Wu check sub-key (the lines above seems only valid when no sub-key)
def update_path(config, keys):
    for key in keys:
        if key in config:
            if isinstance(config[key], dict):  # Check if the value is a dictionary
                for sub_key, path in config[key].items():
                    config[key][sub_key] = path.format(config['experiment_name'])
            else:
                config[key] = config[key].format(config['experiment_name'])


def process_config(config):
    # set random seed
    if 'random_seed' in config:
        set_random_seed(config['random_seed'])

    # set data type
    if 'precision' in config and config['precision'] == 'float64':
        config['dtype'] = torch.float64
        torch.set_default_dtype(torch.float64)
    else:
        config['dtype'] = torch.float32
        torch.set_default_dtype(torch.float32)

    # update path in config
    path_keys = [
    'log_file', 'ckpt_path', 'pre_ckpt_path', 'figs_train', 
    'figs_pretrain', 'output_file', 'figs_analysis', 'restore_loss', 
    'encoded_features_file']

    update_path(config, path_keys)
    # mkdir
    dir_keys = ['log_file', 'pre_ckpt_path', 'figs_pretrain', 'output_file', 'restore_loss']
    config_mkdir(config, dir_keys)

    return config


if __name__ == '__main__':
    # This is a test to ensure load YAML file correctly

    from kogger import Logger

    args   = get_parser().parse_args()
    config = load_config(yaml_filename=args.filename)
    config = process_config(config)
    logger = Logger('CONFIG')
    logger.info('Load config successfully!')
    logger.info(pprint.pformat(config))
