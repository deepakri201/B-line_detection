"""
Create the 2-channel segmentation labels and the CSV read by training_run.py.

Positive labels are the rater's existing 4 mm origin masks, stored as
background and foreground channels. Every negative frame in a split shares one
empty label. Paths in the CSV are relative to intermediate/data/images.

Output layout (under project_root):
    intermediate/data/images/labelsets/frames_1_disk_4mm/{split}/pos/BEDLUS_{id1}_{id2}_{frame}_label.tiff
    intermediate/data/images/labelsets/frames_1_disk_4mm/{split}/neg/negative_label.tiff
    intermediate/data/images/labelsets/frames_1_disk_4mm/image_label_combinations.csv
"""

import os
import pickle
import argparse

import numpy as np
import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm
from natsort import natsorted


# directories produced by 1_preprocess_lahey.py and 5_create_splits.py
processed_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/processed'
project_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/lahey_training'
rater = None

dataset_name = 'frames_1'
labelset_name = 'frames_1_disk_4mm'

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


def read_mask(path: str) -> np.ndarray:
    """
    Args:
        path:  8-bit mask PNG.

    Returns:
        mask:  2D uint8 array.
    """
    mask = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if mask.ndim == 3:
        mask = mask[..., 0]
    return np.where(mask > 0, 255, 0).astype(np.uint8)


def add_background(foreground: np.ndarray) -> np.ndarray:
    """
    Args:
        foreground:  2D uint8 mask. Foreground pixels are 255.

    Returns:
        label:  uint8 array of shape (2, height, width), background channel first.
    """
    foreground = foreground[None, ...]
    return np.concatenate((255 - foreground, foreground), axis=0)


def write_label(label: np.ndarray, path: str) -> None:
    """
    Args:
        label:  2-channel uint8 label.
        path:  TIFF path to write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sitk.WriteImage(sitk.GetImageFromArray(label), path)


def relative_path(*parts: str) -> str:
    """
    Args:
        parts:  path pieces below intermediate/data/images.

    Returns:
        path:  forward-slash path stored in the training CSV.
    """
    return '/'.join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create Lahey segmentation labels and the training CSV.')
    parser.add_argument('--processed-root', default=processed_root, help='Output directory of 1_preprocess_lahey.py.')
    parser.add_argument('--project-root', default=project_root, help='Project tree that training_run.py will use.')
    parser.add_argument('--rater', default=rater, help='Annotator whose masks are the labels. Used when the split file does not record one.')
    args = parser.parse_args()

    images_root = os.path.join(args.processed_root, 'images')
    masks_root = os.path.join(args.processed_root, 'masks')
    for path, label in ((images_root, 'images'), (masks_root, 'masks')):
        if not os.path.isdir(path):
            raise FileNotFoundError(f'{label} directory not found: {path}')

    splits = load_splits(args.project_root)
    images_base = os.path.join(args.project_root, 'intermediate', 'data', 'images')
    label_root = os.path.join(images_base, 'labelsets', labelset_name)
    rows = {'split': [], 'b-lines_present': [], 'image_path': [], 'label_info': []}
    counts = {}

    for split_name, entry in splits.items():
        counts[split_name] = {'pos': 0, 'neg': 0}
        negative_label = None
        negative_relative = relative_path('labelsets', labelset_name, split_name, 'neg', 'negative_label.tiff')
        for stem in tqdm(entry['stems'], desc=split_name):
            chosen = entry.get('stem_raters', {}).get(stem, args.rater)
            image_dir = os.path.join(images_root, stem)
            mask_dir = os.path.join(masks_root, chosen, stem) if chosen else ''
            if not chosen or not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
                print(f'Warning: skipping {stem}, which has no image or mask folder.')
                continue
            id1, id2 = stem_ids(stem)
            for frame_name in shared_frames(image_dir, mask_dir):
                token = frame_token(frame_name)
                foreground = read_mask(os.path.join(mask_dir, frame_name))
                positive = bool(np.any(foreground > 0))
                kind = 'pos' if positive else 'neg'
                image_name = f'BEDLUS_{id1}_{id2}_{token}.tiff'
                image_relative = relative_path('datasets', dataset_name, split_name, kind, image_name)
                if positive:
                    label_name = f'BEDLUS_{id1}_{id2}_{token}_label.tiff'
                    label_relative = relative_path('labelsets', labelset_name, split_name, 'pos', label_name)
                    write_label(add_background(foreground), os.path.join(images_base, *label_relative.split('/')))
                else:
                    if negative_label is None:
                        negative_label = add_background(np.zeros_like(foreground))
                        write_label(negative_label, os.path.join(images_base, *negative_relative.split('/')))
                    label_relative = negative_relative
                rows['split'].append(split_name)
                rows['b-lines_present'].append(kind)
                rows['image_path'].append(image_relative)
                rows['label_info'].append(label_relative)
                counts[split_name][kind] += 1

    csv_path = os.path.join(label_root, 'image_label_combinations.csv')
    os.makedirs(label_root, exist_ok=True)
    pd.DataFrame(rows, columns=['split', 'b-lines_present', 'image_path', 'label_info']).to_csv(csv_path, index=False)

    for split_name, count in counts.items():
        print(f'{split_name}: {count["pos"]} positive, {count["neg"]} negative')
        if split_name != 'test' and count['pos'] == 0:
            print(f'Warning: training fold {split_name} has no positive frames.')
    print(f'Wrote {csv_path}')


if __name__ == '__main__':
    main()
