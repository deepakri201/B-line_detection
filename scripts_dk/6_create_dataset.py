"""
Copy one rater's preprocessed Lahey frames into the segmentation dataset layout.

A frame is positive when that rater's mask contains a B-line origin and negative
when the mask is empty. Filenames keep the two ids in the Lahey clip name so
ImageDataset can give every frame from one clip the same training weight.

Output layout (under project_root):
    intermediate/data/images/datasets/frames_1/{split}/{pos|neg}/BEDLUS_{id1}_{id2}_{frame}.tiff
"""

import os
import pickle
import argparse

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from natsort import natsorted


# directories produced by 1_preprocess_lahey.py and 5_create_splits.py
processed_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/processed'
project_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/lahey_training'
rater = None

dataset_name = 'frames_1'

# --------------------------------------------------------------------------------------------------


def load_splits(project: str) -> dict:
    """
    Args:
        project:  project tree that contains the split pickle.

    Returns:
        splits:  split name to study ids and clip stems.
    """
    path = os.path.join(project, 'intermediate', 'data', 'info', 'dataset_split_dictionary.pkl')
    if not os.path.isfile(path):
        raise FileNotFoundError(f'Split file not found: {path}. Run 5_create_splits.py first.')
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def stem_ids(stem: str) -> tuple:
    """
    Args:
        stem:  Lahey clip name.

    Returns:
        id1, id2:  the two ids used in the BEDLUS filename.
    """
    parts = stem.split('_')
    if len(parts) != 2 or not all(parts):
        raise ValueError(f'Clip {stem} must contain exactly two underscore-separated ids.')
    return parts[0], parts[1]


def frame_token(frame_name: str) -> str:
    """
    Args:
        frame_name:  processed frame filename, such as frame_000.png.

    Returns:
        token:  frame number used in the BEDLUS filename.
    """
    base = os.path.splitext(frame_name)[0]
    if not base.startswith('frame_'):
        raise ValueError(f'Unexpected frame filename: {frame_name}')
    return base.split('_', 1)[1]


def shared_frames(image_dir: str, mask_dir: str) -> list:
    """
    Args:
        image_dir:  processed frames for one clip.
        mask_dir:  one rater's masks for that clip.

    Returns:
        names:  frame filenames present as both an image and a mask.
    """
    image_names = {name for name in os.listdir(image_dir) if name.startswith('frame_') and name.endswith('.png')}
    mask_names = {name for name in os.listdir(mask_dir) if name.startswith('frame_') and name.endswith('.png')}
    return natsorted(image_names & mask_names)


def mask_is_positive(path: str) -> bool:
    """
    Args:
        path:  8-bit mask PNG.

    Returns:
        positive:  True when the mask contains a B-line origin.
    """
    mask = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return bool(np.any(mask > 0))


def write_frame(source: str, destination: str) -> None:
    """
    Args:
        source:  grayscale PNG.
        destination:  TIFF path to write.
    """
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    sitk.WriteImage(sitk.ReadImage(source), destination)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create the Lahey frames_1 segmentation dataset.')
    parser.add_argument('--processed-root', default=processed_root, help='Output directory of 1_preprocess_lahey.py.')
    parser.add_argument('--project-root', default=project_root, help='Project tree that training_run.py will use.')
    parser.add_argument('--rater', default=rater, required=rater is None, help='Annotator whose masks define positive frames.')
    args = parser.parse_args()

    images_root = os.path.join(args.processed_root, 'images')
    masks_root = os.path.join(args.processed_root, 'masks', args.rater)
    for path, label in ((images_root, 'images'), (masks_root, f'masks for {args.rater}')):
        if not os.path.isdir(path):
            raise FileNotFoundError(f'{label} directory not found: {path}')

    splits = load_splits(args.project_root)
    dataset_root = os.path.join(args.project_root, 'intermediate', 'data', 'images', 'datasets', dataset_name)
    written = {'pos': 0, 'neg': 0}

    for split_name, entry in splits.items():
        for stem in tqdm(entry['stems'], desc=split_name):
            image_dir = os.path.join(images_root, stem)
            mask_dir = os.path.join(masks_root, stem)
            if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
                print(f'Warning: skipping {stem}, which has no image or {args.rater} mask folder.')
                continue
            id1, id2 = stem_ids(stem)
            for frame_name in shared_frames(image_dir, mask_dir):
                kind = 'pos' if mask_is_positive(os.path.join(mask_dir, frame_name)) else 'neg'
                filename = f'BEDLUS_{id1}_{id2}_{frame_token(frame_name)}.tiff'
                write_frame(
                    os.path.join(image_dir, frame_name),
                    os.path.join(dataset_root, split_name, kind, filename),
                )
                written[kind] += 1

    print(f'Wrote {written["pos"]} positive and {written["neg"]} negative frames to {dataset_root}')


if __name__ == '__main__':
    main()
