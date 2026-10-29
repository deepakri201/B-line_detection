"""
Assign Lahey clips to the folds expected by network_training/training_run.py.

Clips that share a Lahey patient id stay in the same split. A fraction of
patients is held out as the test split, and the rest are divided round-robin
across the training folds. Pass one rater, or --all-raters to include every
clip annotated by any rater.

Output layout (under project_root):
    intermediate/data/info/dataset_split_dictionary.pkl
    intermediate/data/info/dataset_split_dictionary.json
    intermediate/data/info/corner_points_dictionary.pkl
    intermediate/data/info/processing_dictionary.pkl

The last two files are empty. ImageDataset unpickles them on import and never
reads their contents during segmentation training.
"""

import os
import re
import json
import pickle
import argparse
from collections import defaultdict

import numpy as np
from natsort import natsorted


# directories produced by 1_preprocess_lahey.py, and the project tree training_run.py will use
processed_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/processed'
project_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/lahey_training'

# one rater supplies the labels, unless --all-raters is set
rater = None
all_raters = False

# same fold count and seed as dataset_generation/1_create_splits.py
n_folds = 5
test_fraction = 0.2
seed = 11

# --------------------------------------------------------------------------------------------------


def study_id(study: str, stem: str) -> str:
    """
    Args:
        study:  study folder name stored in the clip sidecar.
        stem:  clip name, used when the sidecar has no study.

    Returns:
        group:  id shared by every clip from one Lahey study.
    """
    match = re.search(r'Lahey_([^_]+)', study or '')
    if match:
        return match.group(1)
    if study:
        return study
    return stem


def has_frames(directory: str) -> bool:
    """
    Args:
        directory:  clip folder of images or masks.

    Returns:
        present:  True when the folder contains at least one frame PNG.
    """
    if not os.path.isdir(directory):
        return False
    return any(name.startswith('frame_') and name.endswith('.png') for name in os.listdir(directory))


def annotated_clips(images_root: str, masks_root: str, meta_root: str, raters: list) -> tuple:
    """
    Args:
        images_root:  processed image clips.
        masks_root:  parent directory of the rater mask folders.
        meta_root:  clip sidecars.
        raters:  annotators to consider, in preference order.

    Returns:
        groups:  patient id to clip stems.
        stem_raters:  clip stem to the rater whose mask is used for that clip.
    """
    groups = defaultdict(list)
    stem_raters = {}
    for stem in natsorted(os.listdir(images_root)):
        image_dir = os.path.join(images_root, stem)
        meta_path = os.path.join(meta_root, f'{stem}.json')
        if not has_frames(image_dir) or not os.path.isfile(meta_path):
            continue
        chosen = next((
            name for name in raters if has_frames(os.path.join(masks_root, name, stem))
        ), None)
        if chosen is None:
            continue
        with open(meta_path, 'r') as handle:
            meta = json.load(handle)
        groups[study_id(str(meta.get('study') or ''), stem)].append(stem)
        stem_raters[stem] = chosen
    return groups, stem_raters


def assign_splits(groups: dict, stem_raters: dict, folds: int, held_out: float, split_seed: int) -> dict:
    """
    Args:
        groups:  patient id to clip stems.
        stem_raters:  clip stem to the rater whose mask is used.
        folds:  number of training folds.
        held_out:  fraction of patients reserved for the test split.
        split_seed:  seed for the patient shuffle.

    Returns:
        splits:  split name to its patient ids, clip stems, and chosen raters.
    """
    if folds < 1:
        raise ValueError('The number of folds must be at least 1.')
    if held_out < 0 or held_out >= 1:
        raise ValueError('The test fraction must be at least 0 and less than 1.')

    studies = sorted(groups)
    rng = np.random.default_rng(split_seed)
    shuffled = [studies[index] for index in rng.permutation(len(studies))]
    n_test = int(round(len(shuffled) * held_out))
    if len(shuffled) > 1 and n_test >= len(shuffled):
        n_test = len(shuffled) - 1

    splits = {str(index): {'studies': [], 'stems': []} for index in range(folds)}
    splits['test'] = {'studies': [], 'stems': []}
    for study in shuffled[:n_test]:
        splits['test']['studies'].append(study)
        splits['test']['stems'].extend(groups[study])
    for index, study in enumerate(shuffled[n_test:]):
        fold = str(index % folds)
        splits[fold]['studies'].append(study)
        splits[fold]['stems'].extend(groups[study])

    for name in splits:
        splits[name]['studies'] = sorted(splits[name]['studies'])
        splits[name]['stems'] = natsorted(splits[name]['stems'])
        splits[name]['stem_raters'] = {
            stem: stem_raters[stem] for stem in splits[name]['stems']
        }
    return splits


def write_pickle(path: str, payload) -> None:
    """
    Args:
        path:  pickle path to write.
        payload:  object stored in the pickle.
    """
    with open(path, 'wb') as handle:
        pickle.dump(payload, handle)


def main() -> None:
    parser = argparse.ArgumentParser(description='Create study-level Lahey splits for segmentation training.')
    parser.add_argument('--processed-root', default=processed_root, help='Output directory of 1_preprocess_lahey.py.')
    parser.add_argument('--project-root', default=project_root, help='Project tree that training_run.py will use.')
    parser.add_argument('--rater', default=rater, help='Annotator whose masks define the dataset.')
    parser.add_argument('--all-raters', action='store_true', default=all_raters, help='Include every clip annotated by any rater. Each patient stays in one split.')
    parser.add_argument('--folds', type=int, default=n_folds, help='Number of training folds.')
    parser.add_argument('--test-fraction', type=float, default=test_fraction, help='Fraction of patients held out as test.')
    parser.add_argument('--seed', type=int, default=seed, help='Seed for the patient shuffle.')
    args = parser.parse_args()
    if args.all_raters and args.rater:
        parser.error('Pass either --rater or --all-raters.')
    if not args.all_raters and not args.rater:
        parser.error('Pass --rater, or --all-raters to use every annotator.')

    images_root = os.path.join(args.processed_root, 'images')
    masks_root = os.path.join(args.processed_root, 'masks')
    meta_root = os.path.join(args.processed_root, 'meta')
    for path, label in ((images_root, 'images'), (masks_root, 'masks'), (meta_root, 'meta')):
        if not os.path.isdir(path):
            raise FileNotFoundError(f'{label} directory not found: {path}')

    if args.all_raters:
        raters = natsorted([
            name for name in os.listdir(masks_root) if os.path.isdir(os.path.join(masks_root, name))
        ])
    else:
        raters = [args.rater]
        if not os.path.isdir(os.path.join(masks_root, args.rater)):
            raise FileNotFoundError(f'Masks for {args.rater} not found under {masks_root}')
    if not raters:
        raise FileNotFoundError(f'No rater folders under {masks_root}')

    groups, stem_raters = annotated_clips(images_root, masks_root, meta_root, raters)
    if not groups:
        raise FileNotFoundError('No clips with images, meta, and a mask from the selected raters.')

    splits = assign_splits(groups, stem_raters, args.folds, args.test_fraction, args.seed)
    info_dir = os.path.join(args.project_root, 'intermediate', 'data', 'info')
    os.makedirs(info_dir, exist_ok=True)
    write_pickle(os.path.join(info_dir, 'dataset_split_dictionary.pkl'), splits)
    with open(os.path.join(info_dir, 'dataset_split_dictionary.json'), 'w') as handle:
        json.dump(splits, handle, indent=2)
    write_pickle(os.path.join(info_dir, 'corner_points_dictionary.pkl'), {})
    write_pickle(os.path.join(info_dir, 'processing_dictionary.pkl'), {})

    source = 'all raters' if args.all_raters else args.rater
    print(f'{len(groups)} patients, {sum(len(stems) for stems in groups.values())} clips from {source}')
    for name in [str(index) for index in range(args.folds)] + ['test']:
        entry = splits[name]
        print(f'{name}: {len(entry["studies"])} patients, {len(entry["stems"])} clips')
        if name != 'test' and not entry['stems']:
            print(f'Warning: training fold {name} has no clips.')
    print(f'Wrote splits to {info_dir}')


if __name__ == '__main__':
    main()
