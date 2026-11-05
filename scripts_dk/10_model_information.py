"""
Write a torchinfo summary next to each selected checkpoint.

Same result as evaluation/2_model_information.py. The model directory, the
folders to summarize, and the input shape are passed on the command line.
Pass --models all to summarize every folder in the models directory.

    python scripts_dk/10_model_information.py \
        --models ensemble_resnet18_unet \
        --project-root /path/to/lahey_training
"""

import os
import sys
import io
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import pandas as pd
from tqdm import tqdm
from torchinfo import summary
from contextlib import redirect_stdout

import models.unet  # noqa: F401
import models.segmentation  # noqa: F401


# same default project tree as scripts_dk/5_create_splits.py
project_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/lahey_training'


def load_checkpoint(path: str, device: str) -> torch.nn.Module:
    """
    Args:
        path:  path to a torch-saved module.
        device:  'cuda' or 'cpu'.

    Returns:
        model:  evaluation-mode module on the requested device.
    """
    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    model.eval()
    model.to(device)
    return model


def checkpoint_pairs(folder: str) -> list:
    """
    Args:
        folder:  one model or ensemble directory.

    Returns:
        pairs:  (checkpoint filename, settings filename) matched by index.
    """
    names = sorted(name for name in os.listdir(folder) if name.startswith('model') and name.endswith('.pth'))
    pairs = []
    for name in names:
        stem = os.path.splitext(name)[0]
        suffix = stem[len('model'):]
        settings_name = f'experiment_settings{suffix}.pkl'
        if not os.path.isfile(os.path.join(folder, settings_name)):
            raise FileNotFoundError(f'{folder} has {name} but no {settings_name}')
        pairs.append((name, settings_name))
    return pairs


def selected_models(models_dir: str, names: list) -> list:
    """
    Args:
        models_dir:  directory that contains model folders.
        names:  folder names, or a single entry 'all'.

    Returns:
        folders:  model folder names to summarize.
    """
    if names == ['all']:
        return sorted(name for name in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, name)))
    missing = [name for name in names if not os.path.isdir(os.path.join(models_dir, name))]
    if missing:
        raise FileNotFoundError(f'Model folders not found in {models_dir}: {", ".join(missing)}')
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description='Write a torchinfo summary for each selected checkpoint.')
    parser.add_argument('--project-root', default=project_root, help='Project tree. Models are read from intermediate/models unless --models-dir is set.')
    parser.add_argument('--models-dir', default=None, help='Directory that contains the trained model folders.')
    parser.add_argument('--models', nargs='+', required=True, help='Model folders to summarize. Pass all to include every folder.')
    parser.add_argument('--output-shape', nargs=2, type=int, default=(256, 384), metavar=('HEIGHT', 'WIDTH'), help='Spatial size used for the summary. Defaults to 256 384.')
    args = parser.parse_args()

    models_dir = args.models_dir or os.path.join(os.path.abspath(args.project_root), 'intermediate', 'models')
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f'Model directory not found: {models_dir}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')
    height, width = args.output_shape

    for model_name in tqdm(selected_models(models_dir, args.models)):
        folder = os.path.join(models_dir, model_name)
        for name, settings_name in checkpoint_pairs(folder):
            settings = pd.read_pickle(os.path.join(folder, settings_name))
            model = load_checkpoint(os.path.join(folder, name), device)
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                summary(model, (1, settings['input_channels'], height, width), verbose=1, device=device)
            summary_path = os.path.join(folder, f'{os.path.splitext(name)[0]}_summary.txt')
            with io.open(summary_path, 'w', encoding='utf-8') as writer:
                writer.write(buffer.getvalue())
            print(f'Wrote {summary_path}')
            del model


if __name__ == '__main__':
    main()
