"""
Run every pixel-level segmentation checkpoint on the preprocessed Lahey frames.

For each experiment folder, predictions from model*.pth are softmaxed and
averaged (or ensemble.pth is used as already-averaged probabilities), matching
evaluation/segmentation/3_instance_detection.py. The foreground channel is
thresholded and written as an 8-bit mask.

Output layout (under predictions_root):
    {model_name}/{stem}/frame_{index:03d}.png
"""

import os
import sys
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from tqdm import tqdm
from natsort import natsorted
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio
imwrite = imageio.imwrite
from torch.utils.data import Dataset, DataLoader

from utils.training_utils import get_prediction


# preprocessed frames from 1_preprocess_lahey.py
images_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/processed/images'
predictions_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/predictions'

# pixel-level checkpoints (one subdirectory per experiment)
model_dir = '/Users/dk422/Documents/Data/LUS/BEDLUS-113-data/networks/pixel_level'
model_names = 'all'

# foreground probability threshold, same default as the segmentation evaluation scripts
prob_threshold = 0.50
batch_size = 16

# run one segmentation experiment on one clip from one annotator and check the masks
# command line: python scripts_dk/2_run_inference.py --test --rater andrew --model 0101_resnet18_unet_0
test_one_clip = False
test_model = None
test_rater = None
test_stem = None
# test_one_clip = True
# test_model = "0101_resnet18_unet_0"
# test_rater = "andrew"
# test_stem = "5144160282_18161988"

# --------------------------------------------------------------------------------------------------


class FrameDataset(Dataset):
    """Load a grayscale PNG and scale it to 0-1, repeating channels when the network is pretrained."""

    def __init__(self, paths: list, pretrained: bool) -> None:
        self.paths = paths
        self.pretrained = pretrained

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        array = sitk.GetArrayFromImage(sitk.ReadImage(self.paths[index]))
        image = torch.from_numpy(np.ascontiguousarray(array)).float() / 255.0
        if image.ndim == 2:
            image = image[None, ...]
        elif image.ndim == 3 and image.shape[0] not in (1, 3):
            image = image[..., 0][None, ...]
        if self.pretrained:
            if image.shape[0] != 1:
                raise ValueError('Pretrained segmentation models expect a single-channel frame to repeat.')
            image = image.repeat(3, 1, 1)
        return image


def list_frames(root: str) -> list:
    """
    Args:
        root:  images directory with one subdirectory per clip stem.

    Returns:
        paths:  naturally sorted PNG paths.
    """
    paths = glob(os.path.join(root, '*', 'frame_*.png'))
    return natsorted(paths)


def checkpoint_pairs(experiment_dir: str) -> tuple:
    """
    Args:
        experiment_dir:  one model subdirectory.

    Returns:
        names:  checkpoint filenames. ensemble.pth is already averaged probabilities.
        settings_names:  matching experiment_settings pickle filenames.
    """
    filenames = natsorted(os.listdir(experiment_dir))
    settings_names = [name for name in filenames if name.startswith('experiment_settings') and name.endswith('.pkl')]
    if 'ensemble.pth' in filenames:
        if not settings_names:
            raise ValueError(f'{experiment_dir} has ensemble.pth and no experiment settings.')
        return ['ensemble.pth'], [settings_names[0]]
    names = [name for name in filenames if name.startswith('model') and name.endswith('.pth')]
    if not names or not settings_names:
        return [], []
    if len(settings_names) == 1 and len(names) > 1:
        settings_names = settings_names * len(names)
    if len(names) != len(settings_names):
        raise ValueError(
            f'{experiment_dir} has {len(names)} checkpoints and {len(settings_names)} settings files.'
        )
    return names, settings_names


def load_checkpoint(path: str, device: str) -> torch.nn.Module:
    """
    Args:
        path:  path to a torch-saved module.
        device:  'cuda' or 'cpu'.

    Returns:
        model:  evaluation-mode module on the requested device.
    """
    # register classes that the pickled modules refer to
    import models.unet  # noqa: F401
    import models.segmentation  # noqa: F401

    try:
        model = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        model = torch.load(path, map_location=device)
    model.eval()
    model.to(device)
    return model


def save_mask(probability: np.ndarray, path: str) -> None:
    """
    Args:
        probability:  foreground probability map.
        path:  PNG path to write.
    """
    mask = np.where(probability >= prob_threshold, 255, 0).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    imwrite(path, mask)


def segmentation_models() -> list:
    """
    Returns:
        names:  experiment folders that contain a segmentation checkpoint.
    """
    names = []
    for name in natsorted(os.listdir(model_dir)):
        experiment_dir = os.path.join(model_dir, name)
        if not os.path.isdir(experiment_dir):
            continue
        checkpoints, settings_names = checkpoint_pairs(experiment_dir)
        if not checkpoints:
            continue
        settings = pd.read_pickle(os.path.join(experiment_dir, settings_names[0]))
        if settings['task'] == 'segmentation':
            names.append(name)
    return names


def select_test_model(requested: str) -> str:
    """
    Args:
        requested:  experiment folder name, or None for the first single-checkpoint model.

    Returns:
        model_name:  one segmentation experiment.
    """
    available = segmentation_models()
    if not available:
        raise FileNotFoundError(f'No segmentation checkpoints under {model_dir}')
    if requested is not None:
        matches = [name for name in available if name == requested]
        if not matches:
            raise FileNotFoundError(f'{requested} is not a segmentation experiment in {model_dir}')
        return matches[0]

    for name in available:
        checkpoints, _ = checkpoint_pairs(os.path.join(model_dir, name))
        if len(checkpoints) == 1:
            return name
    return available[0]


def select_test_clip(rater: str, stem: str) -> tuple:
    """
    Args:
        rater:  annotator name, or None for the first annotator.
        stem:  clip stem, or None for the first clip that annotator labeled.

    Returns:
        rater, stem, frame_paths:  one annotated clip and its preprocessed frames.
    """
    masks_root = os.path.join(os.path.dirname(images_root), 'masks')
    if not os.path.isdir(masks_root):
        raise FileNotFoundError(f'Rater masks not found: {masks_root}')

    raters = natsorted([
        name for name in os.listdir(masks_root) if os.path.isdir(os.path.join(masks_root, name))
    ])
    if rater is not None:
        raters = [name for name in raters if name.lower() == rater.lower()]
        if not raters:
            raise FileNotFoundError(f'No masks for annotator {rater} under {masks_root}')

    for annotation_rater in raters:
        stems = natsorted(os.listdir(os.path.join(masks_root, annotation_rater)))
        if stem is not None:
            stems = [name for name in stems if name == stem]
        for clip in stems:
            frame_paths = natsorted(glob(os.path.join(images_root, clip, 'frame_*.png')))
            if frame_paths:
                return annotation_rater, clip, frame_paths

    raise FileNotFoundError('No clip with both a rater mask and preprocessed frames was found.')


def check_test_output(model_name: str, stem: str, frame_paths: list, destination: str) -> None:
    """
    Args:
        model_name:  experiment that was run.
        stem:  clip that was scored.
        frame_paths:  input frames.
        destination:  directory containing the test predictions.
    """
    prediction_dir = os.path.join(destination, model_name, stem)
    written = natsorted(glob(os.path.join(prediction_dir, 'frame_*.png')))
    if len(written) != len(frame_paths):
        raise RuntimeError(f'Expected {len(frame_paths)} prediction masks, found {len(written)}')

    mask = imageio.imread(written[0])
    if tuple(mask.shape) != (256, 384) or not set(np.unique(mask)).issubset({0, 255}):
        raise RuntimeError(f'Expected a binary uint8 (256, 384) mask, got {mask.shape} {np.unique(mask)}')

    print(f'Test passed for {model_name} on {stem}')
    print(f'{len(written)} prediction masks')
    print(f'Output: {prediction_dir}')


def run_experiment(model_name: str, frame_paths: list, device: str, output_root: str = None) -> bool:
    """
    Args:
        model_name:  experiment subdirectory name.
        frame_paths:  PNG paths to score.
        device:  'cuda' or 'cpu'.
        output_root:  directory under which {model_name}/{stem} is written.

    Returns:
        completed:  False when the experiment is skipped.
    """
    if output_root is None:
        output_root = predictions_root
    experiment_dir = os.path.join(model_dir, model_name)
    names, settings_names = checkpoint_pairs(experiment_dir)
    if not names:
        print(f'Skipping {model_name}: no checkpoints.')
        return False
    settings_list = [pd.read_pickle(os.path.join(experiment_dir, name)) for name in settings_names]
    if any(settings['task'] != 'segmentation' for settings in settings_list):
        print(f'Skipping {model_name}: not a segmentation experiment.')
        return False

    pretrained = bool(settings_list[0]['pretrained'])
    dataset = FrameDataset(frame_paths, pretrained)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    models = [load_checkpoint(os.path.join(experiment_dir, name), device) for name in names]

    output_dir = os.path.join(output_root, model_name)
    cursor = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=model_name):
            batch = batch.to(device)
            total = None
            for model, settings, name in zip(models, settings_list, names):
                prediction = get_prediction(model(batch))
                if settings['aux_classification']:
                    prediction = prediction[0]
                if name != 'ensemble.pth':
                    prediction = torch.softmax(prediction, dim=1)
                prediction = prediction[:, 1, ...]
                if prediction.shape[-2:] != batch.shape[-2:]:
                    prediction = F.interpolate(
                        prediction[:, None, ...],
                        size=batch.shape[-2:],
                        mode='bilinear',
                        align_corners=False,
                    )[:, 0, ...]
                total = prediction if total is None else total + prediction
            average = (total / len(models)).detach().cpu().numpy()

            for probability in average:
                source = frame_paths[cursor]
                stem = os.path.basename(os.path.dirname(source))
                filename = os.path.basename(source)
                save_mask(probability, os.path.join(output_dir, stem, filename))
                cursor += 1

    for model in models:
        del model
    if device == 'cuda':
        torch.cuda.empty_cache()
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description='Run segmentation inference on preprocessed Lahey frames.')
    parser.add_argument('--test', action='store_true', help='Run one model on one clip from one annotator and check the masks.')
    parser.add_argument('--model', default=test_model, help='Segmentation experiment to use with --test.')
    parser.add_argument('--rater', default=test_rater, help='Annotator to use with --test, for example andrew.')
    parser.add_argument('--stem', default=test_stem, help='Clip stem to use with --test.')
    args = parser.parse_args()
    run_test = args.test or test_one_clip

    if not os.path.isdir(images_root):
        raise FileNotFoundError(f'Preprocessed images not found: {images_root}')
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f'Model directory not found: {model_dir}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using {device} device')

    if run_test:
        model_name = select_test_model(args.model)
        rater, stem, frame_paths = select_test_clip(args.rater, args.stem)
        destination = os.path.join(predictions_root, 'test_one_clip')
        print(f'Test model {model_name}, annotator {rater}, clip {stem}, {len(frame_paths)} frames')
        if not run_experiment(model_name, frame_paths, device, destination):
            raise RuntimeError(f'{model_name} did not produce predictions.')
        check_test_output(model_name, stem, frame_paths, destination)
        return

    if model_names == 'all':
        selected = [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]
    elif isinstance(model_names, str):
        selected = [model_names]
    else:
        selected = list(model_names)

    frame_paths = list_frames(images_root)
    if not frame_paths:
        raise FileNotFoundError(f'No frames under {images_root}')
    print(f'{len(frame_paths)} frames, {len(selected)} models')
    os.makedirs(predictions_root, exist_ok=True)

    for model_name in natsorted(selected):
        run_experiment(model_name, frame_paths, device)

    print(f'Wrote masks to {predictions_root}')


if __name__ == '__main__':
    main()
