import importlib
import os
from numpy import array
from yaml import safe_load


def load_cached(metadata:dict, metric:str):
    file_prefix, policy = metadata['file_prefix'], metadata['policy']
    cached_name = f'{file_prefix}_{policy}_{metric}'
    cached = importlib.import_module(f"trace.clustering.cached.{cached_name}")
    if policy == 'ground_truth':
        sensitivity_name = f'{file_prefix}_{policy}_sensitivity'
        med_dist = importlib.import_module(f"trace.clustering.cached.{sensitivity_name}").med_dist
    else: med_dist = None

    return array(cached.labels), cached.medoid_indices, med_dist


def load_colors(path:str|None=None, suffix:str='plt'):
    if path is None: path = f'trace/configs/colors.yaml'
    return safe_load(open(path))[suffix]


def clear_graph_dir(metadata:dict, policy:str|None=None):
    file_prefix = metadata['file_prefix']
    if policy is None: policy = metadata['policy']
    graph_directory = f"plots/{file_prefix}/{policy}/"

    os.makedirs(graph_directory, exist_ok=True)
    for file in os.listdir(graph_directory):
        path = os.path.join(graph_directory, file)
        if os.path.isfile(path): os.remove(path)

    return graph_directory
