"""
Write one movie per clip comparing a rater's B-line mask with a model prediction.

Each frame is two panels of the same ultrasound image. The left panel overlays
the ground-truth mask in green. The right panel overlays the predicted mask in
orange.

Output layout (under movies_root):
    {model_name}/{rater}/{stem}.mp4
"""

import os
import argparse

import numpy as np
from glob import glob
from tqdm import tqdm
from natsort import natsorted
from PIL import Image, ImageDraw, ImageFont
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio


# directories produced by 1_preprocess_lahey.py and 2_run_inference.py
images_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/processed/images'
masks_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/processed/masks'
predictions_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/predictions'
movies_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/viz'

# playback and overlay appearance
frames_per_second = 10
overlay_alpha = 0.55
ground_truth_color = (0, 220, 70)
prediction_color = (255, 120, 0)
panel_gap = 8

# write one movie for one model, rater, and clip
# command line: python scripts_dk/4_make_clip_movies.py --test --rater andrew --model 0101_resnet18_unet_0
test_one_clip = False
test_model = None
test_rater = None
test_stem = None

# --------------------------------------------------------------------------------------------------


def read_gray(path: str) -> np.ndarray:
    """
    Args:
        path:  grayscale PNG.

    Returns:
        image:  2D uint8 array.
    """
    image = imageio.imread(path)
    if image.ndim == 3:
        image = image[..., 0]
    return image


def read_mask(path: str) -> np.ndarray:
    """
    Args:
        path:  8-bit mask PNG.

    Returns:
        mask:  boolean array, True where the PNG is nonzero.
    """
    return read_gray(path) > 0


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple) -> np.ndarray:
    """
    Args:
        image:  grayscale frame.
        mask:  boolean mask of the same height and width.
        color:  RGB color used for the mask.

    Returns:
        panel:  RGB image with the mask blended onto the frame.
    """
    rgb = np.repeat(image[..., None], 3, axis=2).astype(np.float32)
    blended = rgb.copy()
    blended[mask] = (1.0 - overlay_alpha) * rgb[mask] + overlay_alpha * np.array(color, dtype=np.float32)
    return np.clip(blended, 0, 255).astype(np.uint8)


def label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    """
    Args:
        panel:  RGB image.
        text:  short label drawn in the upper left.

    Returns:
        labeled:  copy of the panel with the label.
    """
    image = Image.fromarray(panel)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, 8 * len(text) + 8, 16), fill=(0, 0, 0))
    draw.text((4, 2), text, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def side_by_side(image: np.ndarray, ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """
    Args:
        image:  grayscale ultrasound frame.
        ground_truth:  boolean rater mask.
        prediction:  boolean predicted mask.

    Returns:
        frame:  RGB frame with the ground truth on the left and the prediction on the right.
    """
    left = label_panel(overlay_mask(image, ground_truth, ground_truth_color), 'Ground truth')
    right = label_panel(overlay_mask(image, prediction, prediction_color), 'Prediction')
    gap = np.full((left.shape[0], panel_gap, 3), 255, dtype=np.uint8)
    return np.concatenate((left, gap, right), axis=1)


def shared_frames(stem: str, rater: str, model_name: str) -> list:
    """
    Args:
        stem:  clip name.
        rater:  annotator name.
        model_name:  prediction experiment name.

    Returns:
        names:  frame filenames present as an image, a rater mask, and a prediction.
    """
    directories = (
        os.path.join(images_root, stem),
        os.path.join(masks_root, rater, stem),
        os.path.join(predictions_root, model_name, stem),
    )
    if not all(os.path.isdir(path) for path in directories):
        return []
    image_names, mask_names, prediction_names = (set(os.listdir(path)) for path in directories)
    names = [name for name in image_names & mask_names & prediction_names if name.startswith('frame_') and name.endswith('.png')]
    return natsorted(names)


def write_movie(frames: list, path: str) -> None:
    """
    Args:
        frames:  RGB frames in display order.
        path:  movie path to write.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    writer = imageio.get_writer(path, fps=frames_per_second, macro_block_size=1)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def make_clip_movie(model_name: str, rater: str, stem: str) -> str:
    """
    Args:
        model_name:  prediction experiment name.
        rater:  annotator name.
        stem:  clip name.

    Returns:
        path:  written movie path.
    """
    names = shared_frames(stem, rater, model_name)
    if not names:
        raise FileNotFoundError(f'No shared frames for {model_name}, {rater}, {stem}')

    frames = []
    for name in names:
        image = read_gray(os.path.join(images_root, stem, name))
        ground_truth = read_mask(os.path.join(masks_root, rater, stem, name))
        prediction = read_mask(os.path.join(predictions_root, model_name, stem, name))
        if ground_truth.shape != image.shape or prediction.shape != image.shape:
            raise ValueError(f'Shape mismatch for {stem} {name}')
        frames.append(side_by_side(image, ground_truth, prediction))

    path = os.path.join(movies_root, model_name, rater, f'{stem}.mp4')
    write_movie(frames, path)
    return path


def prediction_models() -> list:
    """
    Returns:
        names:  prediction folders that contain frame masks directly under each clip.
    """
    return natsorted([
        name for name in os.listdir(predictions_root)
        if glob(os.path.join(predictions_root, name, '*', 'frame_*.png'))
    ])


def select_test_case(model_name: str, rater: str, stem: str) -> tuple:
    """
    Args:
        model_name:  requested experiment, or None for the first available model.
        rater:  requested annotator, or None for the first annotator.
        stem:  requested clip, or None for the first clip with shared frames.

    Returns:
        model_name, rater, stem:  one movie to create.
    """
    models = prediction_models()
    if model_name is not None:
        models = [name for name in models if name == model_name]
    if not models:
        raise FileNotFoundError(f'No prediction folders with frame masks under {predictions_root}')

    raters = natsorted([
        name for name in os.listdir(masks_root) if os.path.isdir(os.path.join(masks_root, name))
    ])
    if rater is not None:
        raters = [name for name in raters if name.lower() == rater.lower()]
    if not raters:
        raise FileNotFoundError(f'No rater masks under {masks_root}')

    for selected_model in models:
        clips = natsorted(os.listdir(os.path.join(predictions_root, selected_model)))
        if stem is not None:
            clips = [name for name in clips if name == stem]
        for selected_rater in raters:
            for clip in clips:
                if shared_frames(clip, selected_rater, selected_model):
                    return selected_model, selected_rater, clip

    raise FileNotFoundError('No clip has an image, a rater mask, and a prediction for every requested input.')


def main() -> None:
    parser = argparse.ArgumentParser(description='Make side-by-side ground-truth and prediction movies.')
    parser.add_argument('--test', action='store_true', help='Write one movie for one model, rater, and clip.')
    parser.add_argument('--model', default=test_model, help='Prediction folder to use. Defaults to every available model.')
    parser.add_argument('--rater', default=test_rater, help='Annotator to use. Defaults to every rater.')
    parser.add_argument('--stem', default=test_stem, help='Clip stem to use. Defaults to every clip.')
    args = parser.parse_args()

    for path, label in ((images_root, 'images'), (masks_root, 'masks'), (predictions_root, 'predictions')):
        if not os.path.isdir(path):
            raise FileNotFoundError(f'{label} directory not found: {path}')

    if args.test or test_one_clip:
        model_name, rater, stem = select_test_case(args.model, args.rater, args.stem)
        path = make_clip_movie(model_name, rater, stem)
        print(f'Test movie: {path}')
        return

    models = prediction_models()
    if args.model is not None:
        models = [name for name in models if name == args.model]
    raters = natsorted([
        name for name in os.listdir(masks_root) if os.path.isdir(os.path.join(masks_root, name))
    ])
    if args.rater is not None:
        raters = [name for name in raters if name.lower() == args.rater.lower()]

    written = 0
    for model_name in models:
        clips = natsorted([
            name for name in os.listdir(os.path.join(predictions_root, model_name))
            if os.path.isdir(os.path.join(predictions_root, model_name, name))
        ])
        if args.stem is not None:
            clips = [name for name in clips if name == args.stem]
        for rater in raters:
            for stem in tqdm(clips, desc=f'{model_name}/{rater}'):
                if not shared_frames(stem, rater, model_name):
                    continue
                make_clip_movie(model_name, rater, stem)
                written += 1

    print(f'Wrote {written} movies to {movies_root}')


if __name__ == '__main__':
    main()
