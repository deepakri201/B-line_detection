"""
Copy trained segmentation checkpoints into one ensemble folder.

Same result as evaluation/1_create_ensemble_folder.py. The model directory,
the fold folders, and the ensemble name are passed on the command line.
Inference averages every model_*.pth in the new folder.

    python scripts_dk/9_create_ensemble_folder.py \
        --models 0101_resnet18_unet_0 0102_resnet18_unet_1 0103_resnet18_unet_2 0104_resnet18_unet_3 0105_resnet18_unet_4 \
        --ensemble-name ensemble_resnet18_unet \
        --project-root /path/to/lahey_training
"""

import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from shutil import copyfile

import models.unet  # noqa: F401
import models.segmentation  # noqa: F401


# same default project tree as scripts_dk/5_create_splits.py
project_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/lahey_training'


def load_checkpoint(path: str) -> torch.nn.Module:
    """
    Args:
        path:  path to a torch-saved module.

    Returns:
        model:  the loaded module, used to confirm the checkpoint unpickles.
    """
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def main() -> None:
    parser = argparse.ArgumentParser(description='Copy fold checkpoints into one ensemble folder.')
    parser.add_argument('--project-root', default=project_root, help='Project tree. Models are read from intermediate/models unless --models-dir is set.')
    parser.add_argument('--models-dir', default=None, help='Directory that contains the trained model folders.')
    parser.add_argument('--models', nargs='+', required=True, help='Fold folders to include, in the order they should be numbered.')
    parser.add_argument('--ensemble-name', required=True, help='Name of the new folder written inside the models directory.')
    args = parser.parse_args()

    models_dir = args.models_dir or os.path.join(os.path.abspath(args.project_root), 'intermediate', 'models')
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f'Model directory not found: {models_dir}')

    available = set(os.listdir(models_dir))
    for name in args.models:
        if name not in available:
            raise FileNotFoundError(f'Model {name} is not available in {models_dir}')
        folder = os.path.join(models_dir, name)
        for filename in ('model.pth', 'experiment_settings.pkl'):
            if not os.path.isfile(os.path.join(folder, filename)):
                raise FileNotFoundError(f'{folder} has no {filename}')

    print(f'The following {len(args.models)} models were selected: {args.models}')

    label = None
    for name in args.models:
        settings = pd.read_pickle(os.path.join(models_dir, name, 'experiment_settings.pkl'))
        if label is None:
            label = settings['label_folder']
        elif label != settings['label_folder']:
            raise ValueError('Not all selected models were trained using the same labels.')

    output_path = os.path.join(models_dir, args.ensemble_name)
    if os.path.exists(output_path):
        raise FileExistsError(f'Output directory already exists: {output_path}')
    os.mkdir(output_path)

    origin_dict = {'origin': [], 'model': [], 'settings': []}
    for idx, name in enumerate(args.models):
        source = os.path.join(models_dir, name)
        model_name = f'model_{idx}.pth'
        settings_name = f'experiment_settings_{idx}.pkl'
        copyfile(os.path.join(source, 'model.pth'), os.path.join(output_path, model_name))
        copyfile(os.path.join(source, 'experiment_settings.pkl'), os.path.join(output_path, settings_name))
        load_checkpoint(os.path.join(output_path, model_name))
        origin_dict['origin'].append(name)
        origin_dict['model'].append(model_name)
        origin_dict['settings'].append(settings_name)

    pd.DataFrame.from_dict(origin_dict).to_csv(os.path.join(output_path, 'model_origins.csv'), index=False)
    print(f'Wrote {output_path}')


if __name__ == '__main__':
    main()
