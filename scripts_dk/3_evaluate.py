"""
Compare predicted B-line masks with each rater's masks.

Instance scores follow evaluation/segmentation/3_instance_detection.py:
connected components are reduced to centroids and matched within 0.5 cm.
Pixel Dice, IoU, precision, and recall are reported as well, because the
Lahey annotations are short strokes rather than origin points. A frame whose
prediction and reference are both empty counts as a perfect match.

One Excel file is written per model, with a summary sheet, a per-clip sheet,
and one frame-level sheet per rater.
"""

import os
import sys
import json
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import pandas as pd
import SimpleITK as sitk
from glob import glob
from natsort import natsorted

from utils.evaluation_utils import (
    separate_structures,
    get_centroids,
    evaluate_detections,
    get_detection_statistics,
)


# directories produced by 1_preprocess_lahey.py and 2_run_inference.py
processed_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data'
masks_root = os.path.join(processed_root, 'processed/masks')
predictions_root = os.path.join(processed_root, 'predictions')
meta_root = os.path.join(processed_root, 'processed/meta')
results_root = os.path.join(processed_root, 'evaluation')

# match distance used by evaluation/segmentation/3_instance_detection.py
max_physical_distance_cm = 0.50

# --------------------------------------------------------------------------------------------------


def read_mask(path: str) -> np.ndarray:
    """
    Args:
        path:  8-bit PNG mask.

    Returns:
        mask:  boolean array, True where the PNG is nonzero.
    """
    array = sitk.GetArrayFromImage(sitk.ReadImage(path))
    if array.ndim == 3:
        array = array[..., 0]
    return array > 0


def pixel_scores(prediction: np.ndarray, reference: np.ndarray) -> dict:
    """
    Args:
        prediction:  boolean predicted mask.
        reference:  boolean rater mask.

    Returns:
        scores:  overlap counts and per-frame Dice, IoU, precision, and recall.
    """
    intersection = int(np.logical_and(prediction, reference).sum())
    predicted_area = int(prediction.sum())
    reference_area = int(reference.sum())
    union = predicted_area + reference_area - intersection
    empty = predicted_area == 0 and reference_area == 0

    if empty:
        dice = iou = precision = recall = 1.0
    else:
        dice = (2 * intersection / (predicted_area + reference_area)) if (predicted_area + reference_area) else 0.0
        iou = (intersection / union) if union else 0.0
        precision = (intersection / predicted_area) if predicted_area else 0.0
        recall = (intersection / reference_area) if reference_area else 0.0

    return {
        'intersection': intersection,
        'predicted_area': predicted_area,
        'reference_area': reference_area,
        'dice': dice,
        'iou': iou,
        'pixel_precision': precision,
        'pixel_recall': recall,
    }


def component_centroids(mask: np.ndarray) -> list:
    """
    Args:
        mask:  boolean mask.

    Returns:
        centroids:  (x, y) centroid of each connected component.
    """
    if not np.any(mask):
        return []
    structures = separate_structures(mask.astype(np.uint8))
    if structures.shape[-1] == 0:
        return []
    return get_centroids(structures)


def max_pixel_distance(meta: dict) -> float:
    """
    Args:
        meta:  sidecar written by preprocessing, with spacing in mm and the resize scale.

    Returns:
        distance:  match radius in pixels of the resized frame.
    """
    spacing_mm = float(np.mean(meta['pixel_spacing_mm']))
    distance_mm = max_physical_distance_cm * 10.0
    return (distance_mm / spacing_mm) * float(meta['scaling_factor'])


def instance_counts(prediction: np.ndarray, reference: np.ndarray, distance_px: float) -> tuple:
    """
    Args:
        prediction:  boolean predicted mask.
        reference:  boolean rater mask.
        distance_px:  maximum centroid distance counted as a match.

    Returns:
        tp, fp, fn:  detection counts, with true positives counted once per reference component.
    """
    detection = evaluate_detections(
        component_centroids(prediction),
        component_centroids(reference),
        distance_px,
    )
    true_positive = len(set([point[1] for point in detection['TP']]))
    false_positive = len(detection['FP'])
    false_negative = len(detection['FN'])
    return true_positive, false_positive, false_negative


def pooled_pixel_scores(frame_df: pd.DataFrame) -> dict:
    """
    Args:
        frame_df:  frame rows that include intersection and area columns.

    Returns:
        scores:  micro-averaged overlap and macro-averaged Dice and IoU.
    """
    intersection = int(frame_df['intersection'].sum())
    predicted_area = int(frame_df['predicted_area'].sum())
    reference_area = int(frame_df['reference_area'].sum())
    union = predicted_area + reference_area - intersection
    if predicted_area == 0 and reference_area == 0:
        dice = iou = precision = recall = 1.0
    else:
        dice = (2 * intersection / (predicted_area + reference_area)) if (predicted_area + reference_area) else 0.0
        iou = (intersection / union) if union else 0.0
        precision = (intersection / predicted_area) if predicted_area else 0.0
        recall = (intersection / reference_area) if reference_area else 0.0
    return {
        'dice_micro': dice,
        'iou_micro': iou,
        'pixel_precision_micro': precision,
        'pixel_recall_micro': recall,
        'dice_macro': float(frame_df['dice'].mean()) if len(frame_df) else 0.0,
        'iou_macro': float(frame_df['iou'].mean()) if len(frame_df) else 0.0,
    }


def summarize_counts(frame_df: pd.DataFrame) -> dict:
    """
    Args:
        frame_df:  rows with TP, FP, and FN.

    Returns:
        summary:  pooled detection counts and precision, recall, and F1.
    """
    true_positive = int(frame_df['TP'].sum())
    false_positive = int(frame_df['FP'].sum())
    false_negative = int(frame_df['FN'].sum())
    precision, recall, f1_score = get_detection_statistics(true_positive, false_positive, false_negative)
    pixel = pooled_pixel_scores(frame_df)
    return {
        'n_frames': int(len(frame_df)),
        'TP': true_positive,
        'FP': false_positive,
        'FN': false_negative,
        'precision': precision,
        'recall': recall,
        'f1': f1_score,
        **pixel,
    }


def sheet_name(rater: str, used: set) -> str:
    """
    Args:
        rater:  rater identifier.
        used:  sheet names already chosen.

    Returns:
        name:  an Excel sheet name of at most 31 characters.
    """
    cleaned = ''.join('_' if character in '[]:*?/\\' else character for character in rater)[:31]
    if not cleaned:
        cleaned = 'rater'
    candidate = cleaned
    suffix = 2
    while candidate in used:
        tail = f'_{suffix}'
        candidate = cleaned[:31 - len(tail)] + tail
        suffix += 1
    used.add(candidate)
    return candidate


def load_meta() -> dict:
    """
    Returns:
        meta:  stem -> sidecar dictionary.
    """
    meta = {}
    for path in glob(os.path.join(meta_root, '*.json')):
        with open(path, 'r') as handle:
            payload = json.load(handle)
        meta[payload['stem']] = payload
    return meta


def evaluate_model(model_name: str, meta: dict) -> None:
    """
    Args:
        model_name:  prediction subdirectory.
        meta:  preprocessing sidecars keyed by stem.
    """
    prediction_dir = os.path.join(predictions_root, model_name)
    raters = natsorted([
        name for name in os.listdir(masks_root) if os.path.isdir(os.path.join(masks_root, name))
    ])
    frame_tables = {}
    clip_rows = []
    summary_rows = []

    for rater in raters:
        rows = []
        rater_dir = os.path.join(masks_root, rater)
        reference_paths = natsorted(glob(os.path.join(rater_dir, '*', 'frame_*.png')))
        for reference_path in reference_paths:
            stem = os.path.basename(os.path.dirname(reference_path))
            filename = os.path.basename(reference_path)
            prediction_path = os.path.join(prediction_dir, stem, filename)
            if not os.path.exists(prediction_path):
                print(f'Warning: missing prediction {model_name}/{stem}/{filename}')
                continue
            if stem not in meta:
                print(f'Warning: missing meta for {stem}')
                continue

            reference = read_mask(reference_path)
            prediction = read_mask(prediction_path)
            if prediction.shape != reference.shape:
                raise ValueError(f'Shape mismatch for {stem} {filename}: {prediction.shape} vs {reference.shape}')

            scores = pixel_scores(prediction, reference)
            true_positive, false_positive, false_negative = instance_counts(
                prediction, reference, max_pixel_distance(meta[stem])
            )
            rows.append({
                'rater': rater,
                'stem': stem,
                'study': meta[stem]['study'],
                'frame': os.path.splitext(filename)[0],
                'dice': scores['dice'],
                'iou': scores['iou'],
                'pixel_precision': scores['pixel_precision'],
                'pixel_recall': scores['pixel_recall'],
                'TP': true_positive,
                'FP': false_positive,
                'FN': false_negative,
                'intersection': scores['intersection'],
                'predicted_area': scores['predicted_area'],
                'reference_area': scores['reference_area'],
            })

        if not rows:
            continue
        frame_df = pd.DataFrame(rows)
        frame_tables[rater] = frame_df

        for stem, clip_df in frame_df.groupby('stem', sort=False):
            clip_summary = summarize_counts(clip_df)
            clip_rows.append({
                'rater': rater,
                'stem': stem,
                'study': clip_df['study'].iloc[0],
                **clip_summary,
            })
        rater_summary = summarize_counts(frame_df)
        rater_summary['n_clips'] = int(frame_df['stem'].nunique())
        summary_rows.append({'rater': rater, **rater_summary})

    if not summary_rows:
        print(f'No overlapping frames for {model_name}.')
        return

    os.makedirs(results_root, exist_ok=True)
    workbook_path = os.path.join(results_root, f'{model_name}.xlsx')
    used_names = {'summary', 'clips'}
    with pd.ExcelWriter(workbook_path) as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name='summary', index=False)
        pd.DataFrame(clip_rows).to_excel(writer, sheet_name='clips', index=False)
        for rater, frame_df in frame_tables.items():
            exported = frame_df.drop(columns=['intersection', 'predicted_area', 'reference_area'])
            exported.to_excel(writer, sheet_name=sheet_name(rater, used_names), index=False)
    print(f'Wrote {workbook_path}')


def main() -> None:
    if not os.path.isdir(masks_root):
        raise FileNotFoundError(f'Rater masks not found: {masks_root}')
    if not os.path.isdir(predictions_root):
        raise FileNotFoundError(f'Predictions not found: {predictions_root}')

    meta = load_meta()
    model_names = natsorted([
        name for name in os.listdir(predictions_root)
        if glob(os.path.join(predictions_root, name, '*', 'frame_*.png'))
    ])
    if not model_names:
        raise FileNotFoundError(f'No model folders under {predictions_root}')

    for model_name in model_names:
        evaluate_model(model_name, meta)


if __name__ == '__main__':
    main()
