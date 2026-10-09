"""
Convert Lahey lung-ultrasound DICOMs and JSON annotations into the image
format used by the segmentation models.

Each DICOM is a multi-frame YBR_FULL_422 sector image. Rater JSON files store
B-lines as two width points in millimeters from the top-left of the frame.
The midpoint of those points is projected onto the pleural line, and a disk is
drawn at that origin. This matches the BEDLUS labels, which mark B-line origins
rather than the whole sector. Frames are masked to the JSON fan, cropped,
padded, and resized to 256x384, matching utils.conversion_utils.crop_and_resize.
Images and masks are saved as 8-bit PNGs with intensities in 0-255. Inference
divides by 255 when loading.

Output layout (under output_root):
    images/{stem}/frame_{index:03d}.png
    masks/{rater}/{stem}/frame_{index:03d}.png
    meta/{stem}.json
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
from tqdm import tqdm
try:
    import imageio.v2 as imageio
except ImportError:
    import imageio
imwrite = imageio.imwrite
from skimage.morphology import disk
from skimage.transform import resize
from scipy.ndimage import binary_dilation

try:
    import pydicom
except ImportError as exc:
    raise ImportError(
        'This script requires pydicom, which is not in requirements.txt. '
        'Install it with: pip install pydicom'
    ) from exc


# define input and output directories
input_root = '/Users/dk422/Documents/Data/LUS/Group-001'
output_root = '/Users/dk422/Documents/LUS/AI4Ultrasound/B-LineDetection_Ruben/data/processed'

# network input size, height then width (same as data_preparation/5_get_processed_frames.py)
output_shape = (256, 384)

# disk diameter at each B-line origin, matching the BEDLUS 4 mm point labels
origin_diameter_mm = 4.0

# process one clip from one annotator and check the written files
# command line: python scripts_dk/1_preprocess_lahey.py --test --rater andrew
# test_one_clip = True
# test_rater = "andrew"
# test_stem = "5144160282_18161988"
test_one_clip = False
test_rater = None
test_stem = None

# --------------------------------------------------------------------------------------------------


def dicom_grayscale_frames(path: str) -> tuple:
    """
    Args:
        path:  path to a DICOM file.

    Returns:
        frames:  uint8 grayscale frames of shape (N, H, W).
        spacing_mm:  (row spacing, column spacing) in millimeters.
    """
    dataset = pydicom.dcmread(path)
    pixels = dataset.pixel_array
    photometric = str(getattr(dataset, 'PhotometricInterpretation', 'MONOCHROME2'))

    # pydicom >= 2.1 converts YBR to RGB inside pixel_array. Older versions do not.
    version = tuple(int(part) for part in pydicom.__version__.split('.')[:2])
    if photometric.startswith('YBR') and version < (2, 1):
        from pydicom.pixel_data_handlers.util import convert_color_space
        pixels = convert_color_space(pixels, photometric, 'RGB')
    if photometric.startswith('YBR') or photometric == 'RGB':
        photometric = 'RGB'

    if pixels.ndim == 2:
        pixels = pixels[None, ...]
    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        pixels = pixels[None, ...]

    if photometric == 'RGB':
        if pixels.ndim != 4 or pixels.shape[-1] < 3:
            raise ValueError(f'Expected color frames in {path}, got shape {pixels.shape}.')
        # same weights as skimage.color.rgb2gray, applied to the whole clip at once
        rgb = pixels[..., :3].astype(np.float32)
        gray = 0.2125 * rgb[..., 0] + 0.7154 * rgb[..., 1] + 0.0721 * rgb[..., 2]
        frames = np.clip(np.rint(gray), 0, 255).astype(np.uint8)
    else:
        if pixels.ndim != 3:
            raise ValueError(f'Expected grayscale frames in {path}, got shape {pixels.shape}.')
        if pixels.dtype == np.uint8:
            frames = pixels
        else:
            peak = float(np.max(pixels)) if pixels.size else 1.0
            if peak <= 0:
                peak = 1.0
            frames = np.clip(np.rint(pixels.astype(np.float64) / peak * 255), 0, 255).astype(np.uint8)

    spacing = getattr(dataset, 'PixelSpacing', None)
    if spacing is None:
        raise ValueError(f'{path} has no PixelSpacing, so annotation millimeters cannot be mapped.')
    spacing_mm = (float(spacing[0]), float(spacing[1]))
    return frames, spacing_mm


def fan_mask(shape: tuple, fan: dict) -> np.ndarray:
    """
    Args:
        shape:  (height, width) of the DICOM frame.
        fan:  center, radii, and boundary angles from the annotation JSON.

    Returns:
        mask:  boolean sector mask. Angle 90 degrees is straight down the image.
    """
    height, width = shape
    rows = np.arange(height, dtype=np.float64)[:, None]
    cols = np.arange(width, dtype=np.float64)[None, :]
    delta_row = rows - float(fan['center_rows_px'])
    delta_col = cols - float(fan['center_cols_px'])
    radius = np.hypot(delta_col, delta_row)
    angle = np.degrees(np.arctan2(delta_row, delta_col))

    angle_lo = min(float(fan['angle1']), float(fan['angle2']))
    angle_hi = max(float(fan['angle1']), float(fan['angle2']))
    # keep the wedge that contains the downward axis of a curvilinear sector
    if angle_lo <= 90.0 <= angle_hi:
        angle_ok = (angle >= angle_lo) & (angle <= angle_hi)
    else:
        angle_ok = (angle <= angle_lo) | (angle >= angle_hi)

    radius_ok = (radius >= float(fan['radius1'])) & (radius <= float(fan['radius2']))
    return radius_ok & angle_ok


def geometry_from_fan(mask: np.ndarray, shape: tuple) -> dict:
    """
    Crop, pad, and scale the same way as utils.conversion_utils.crop_and_resize.

    Args:
        mask:  boolean sector mask.
        shape:  output (height, width).

    Returns:
        geometry:  crop, padding, and resize scale applied to images and masks.
    """
    if not np.any(mask):
        raise ValueError('Fan mask is empty.')

    mip_horizontal = np.max(mask, axis=0)
    mip_vertical = np.max(mask, axis=1)
    top = len(mip_vertical) - len(np.trim_zeros(mip_vertical, trim='f'))
    bottom = len(np.trim_zeros(mip_vertical, trim='b'))
    left = len(mip_horizontal) - len(np.trim_zeros(mip_horizontal, trim='f'))
    right = len(np.trim_zeros(mip_horizontal, trim='b'))

    roi_height = bottom - top
    roi_width = right - left
    if (roi_height / shape[0]) * shape[1] - roi_width > 0:
        difference = np.ceil((roi_height / shape[0]) * shape[1] - roi_width)
        if difference % 2 == 1:
            add_left, add_right = int(difference // 2), int(difference // 2 + 1)
        else:
            add_left = add_right = int(difference // 2)
        padding = ((0, 0), (add_left, add_right))
    else:
        difference = np.ceil((roi_width / shape[1]) * shape[0] - roi_height)
        if difference % 2 == 1:
            add_top, add_bottom = int(difference // 2), int(difference // 2 + 1)
        else:
            add_top = add_bottom = int(difference // 2)
        padding = ((add_top, add_bottom), (0, 0))

    padded_height = roi_height + padding[0][0] + padding[0][1]
    scaling_factor = shape[0] / padded_height
    return {
        'crop': ((int(top), int(bottom)), (int(left), int(right))),
        'padding': padding,
        'scaling_factor': float(scaling_factor),
    }


def apply_geometry(array: np.ndarray, geometry: dict, shape: tuple, order: int) -> np.ndarray:
    """
    Args:
        array:  2D image or mask.
        geometry:  crop, padding, and scale from geometry_from_fan.
        shape:  output (height, width).
        order:  interpolation order (1 for images, 0 for masks).

    Returns:
        processed:  uint8 array of the requested shape.
    """
    (top, bottom), (left, right) = geometry['crop']
    cropped = array[top:bottom, left:right]
    padded = np.pad(cropped, pad_width=geometry['padding'], mode='constant')
    resized = resize(padded, shape, order=order, preserve_range=True, anti_aliasing=False)
    return np.clip(np.rint(resized), 0, 255).astype(np.uint8)


def annotation_point_to_pixel(point: list, spacing_mm: tuple) -> np.ndarray:
    """
    Args:
        point:  annotation coordinate (x, y, z) in millimeters.
        spacing_mm:  (row spacing, column spacing) in millimeters.

    Returns:
        pixel:  floating-point (column, row) coordinate.
    """
    row_spacing, col_spacing = spacing_mm
    return np.array([float(point[0]) / col_spacing, float(point[1]) / row_spacing])


def cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    """Return the scalar two-dimensional cross product."""
    return float(first[0] * second[1] - first[1] * second[0])


def ray_line_intersection(
    center: np.ndarray,
    direction: np.ndarray,
    line_start: np.ndarray,
    line_end: np.ndarray,
) -> tuple:
    """
    Intersect a radial ray with a pleural line. The pleural line may be
    extrapolated when its annotation does not quite span the sector boundary.

    Returns:
        radius:  distance from the fan center to the intersection.
        on_segment:  whether the intersection lies on the annotated segment.
    """
    line_direction = line_end - line_start
    denominator = cross_2d(direction, line_direction)
    if abs(denominator) < 1e-8:
        return None

    offset = line_start - center
    ray_scale = cross_2d(offset, line_direction) / denominator
    line_scale = cross_2d(offset, direction) / denominator
    if ray_scale <= 0:
        return None

    radius = ray_scale * float(np.linalg.norm(direction))
    return radius, 0.0 <= line_scale <= 1.0


def pleural_radius(
    angle: float,
    center: np.ndarray,
    pleural_segments: list,
) -> float:
    """
    Find where a radial boundary meets the annotated pleura. Prefer an exact
    segment intersection; otherwise use the nearest segment's extrapolated
    line. This handles adjacent pleural segments and short annotations.
    """
    direction = np.array([np.cos(angle), np.sin(angle)])
    candidates = []
    for line_start, line_end in pleural_segments:
        result = ray_line_intersection(center, direction, line_start, line_end)
        if result is None:
            continue
        radius, on_segment = result
        midpoint = (line_start + line_end) / 2.0
        midpoint_angle = np.arctan2(midpoint[1] - center[1], midpoint[0] - center[0])
        angular_distance = abs(np.arctan2(np.sin(angle - midpoint_angle), np.cos(angle - midpoint_angle)))
        candidates.append((not on_segment, angular_distance, radius))

    if not candidates:
        return None
    return min(candidates)[2]


def b_line_origin(
    b_line: dict,
    center: np.ndarray,
    pleural_segments: list,
    spacing_mm: tuple,
) -> np.ndarray:
    """
    Args:
        b_line:  one annotation with two width points.
        center:  fan center in pixels, (column, row).
        pleural_segments:  pleural lines in pixels.
        spacing_mm:  (row spacing, column spacing).

    Returns:
        origin:  (column, row) where the B-line center meets the pleura.
    """
    width_points = [
        annotation_point_to_pixel(point, spacing_mm)
        for point in b_line['line']['points'][:2]
    ]
    angles = [
        np.arctan2(point[1] - center[1], point[0] - center[0])
        for point in width_points
    ]
    angle_start, angle_end = sorted(angles)
    if angle_end - angle_start > np.pi:
        angle_start, angle_end = angle_end, angle_start + 2 * np.pi
    midpoint_angle = (angle_start + angle_end) / 2.0
    radius = pleural_radius(midpoint_angle, center, pleural_segments)
    if radius is None:
        return None
    direction = np.array([np.cos(midpoint_angle), np.sin(midpoint_angle)])
    return center + radius * direction


def rasterize_blines(
    annotations: list,
    pleura_lines: list,
    shape: tuple,
    spacing_mm: tuple,
    fan: dict,
) -> tuple:
    """
    Args:
        annotations:  b_lines list for one frame. Each item has two width points.
        pleura_lines:  line segments used to place each B-line origin.
        shape:  (height, width) of the DICOM frame.
        spacing_mm:  (row spacing, column spacing).
        fan:  fan center from the associated annotation JSON.

    Returns:
        mask:  uint8 mask with a disk at each B-line origin on the pleura.
        skipped:  number of B-lines that could not be placed.
    """
    height, width = shape
    canvas = np.zeros((height, width), dtype=bool)
    center = np.array([float(fan['center_cols_px']), float(fan['center_rows_px'])])
    segments = [
        (
            annotation_point_to_pixel(item['line']['points'][0], spacing_mm),
            annotation_point_to_pixel(item['line']['points'][1], spacing_mm),
        )
        for item in pleura_lines
    ]
    mean_spacing = (spacing_mm[0] + spacing_mm[1]) / 2.0
    radius_px = max(0, int(round((origin_diameter_mm / 2.0) / mean_spacing)))
    skipped = 0

    for b_line in annotations:
        if not segments:
            skipped += 1
            continue
        origin = b_line_origin(b_line, center, segments, spacing_mm)
        if origin is None:
            skipped += 1
            continue
        column = int(np.clip(np.rint(origin[0]), 0, width - 1))
        row = int(np.clip(np.rint(origin[1]), 0, height - 1))
        canvas[row, column] = True

    if np.any(canvas):
        canvas = binary_dilation(canvas, structure=disk(radius_px))
    return canvas.astype(np.uint8) * 255, skipped


def fan_from_json(data: dict) -> dict:
    """
    Args:
        data:  loaded annotation JSON.

    Returns:
        fan:  geometry fields used to build the sector mask.
    """
    keys = [
        'center_rows_px', 'center_cols_px', 'radius1', 'radius2', 'angle1', 'angle2',
    ]
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f'Annotation JSON is missing fan fields: {missing}')
    return {key: data[key] for key in keys}


def index_inputs(root: str) -> tuple:
    """
    Args:
        root:  Group folder containing one subdirectory per rater.

    Returns:
        dicoms:  stem -> list of DICOM paths.
        annotations:  stem -> list of (json path, rater name).
    """
    dicoms = defaultdict(list)
    annotations = defaultdict(list)
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            if filename.startswith('.'):
                continue
            path = os.path.join(dirpath, filename)
            if filename.endswith('.dcm'):
                dicoms[filename[:-4]].append(path)
            elif filename.endswith('.json'):
                base = filename[:-5]
                if '.' in base:
                    stem, rater = base.split('.', 1)
                else:
                    stem = base
                    rater = os.path.basename(os.path.dirname(dirpath))
                annotations[stem].append((path, rater))
    return dicoms, annotations


def serializable_geometry(geometry: dict, spacing_mm: tuple, study: str, dicom_path: str, stem: str) -> dict:
    """
    Args:
        geometry:  crop, padding, and scale.
        spacing_mm:  DICOM pixel spacing.
        study:  study folder name.
        dicom_path:  DICOM that was decoded.
        stem:  file stem shared by the DICOM and JSON.

    Returns:
        payload:  JSON-serializable sidecar used by evaluation to convert centimeters to pixels.
    """
    (top, bottom), (left, right) = geometry['crop']
    (pad_top, pad_bottom), (pad_left, pad_right) = geometry['padding']
    return {
        'stem': stem,
        'study': study,
        'dicom': dicom_path,
        'pixel_spacing_mm': [spacing_mm[0], spacing_mm[1]],
        'crop': [top, bottom, left, right],
        'padding': [pad_top, pad_bottom, pad_left, pad_right],
        'scaling_factor': geometry['scaling_factor'],
        'output_shape': list(output_shape),
        'annotation_mask': 'disks at B-line origins on the pleura',
        'origin_diameter_mm': origin_diameter_mm,
    }


def json_has_blines(path: str) -> bool:
    """
    Args:
        path:  rater annotation JSON.

    Returns:
        found:  True when at least one frame contains a B-line.
    """
    with open(path, 'r') as handle:
        data = json.load(handle)
    return any(frame.get('b_lines') for frame in data.get('frame_annotations') or [])


def select_test_case(dicoms: dict, annotations: dict, rater: str, stem: str) -> tuple:
    """
    Args:
        dicoms:  stem -> DICOM paths.
        annotations:  stem -> (json path, rater) pairs.
        rater:  requested annotator, or None for the first annotator with B-lines.
        stem:  requested clip stem, or None for the first suitable clip.

    Returns:
        stem, rater, dicom_paths, annotation_paths:  one clip and one annotator.
    """
    requested_rater = None if rater is None else rater.lower()
    candidates = [name for name in sorted(dicoms) if name in annotations]
    if stem is not None:
        if stem not in dicoms or stem not in annotations:
            raise FileNotFoundError(f'Test clip {stem} needs both a DICOM and a JSON annotation.')
        candidates = [stem]

    for candidate in candidates:
        matches = []
        for json_path, annotation_rater in annotations[candidate]:
            if requested_rater is not None and annotation_rater.lower() != requested_rater:
                continue
            if json_has_blines(json_path):
                matches.append((json_path, annotation_rater))
        if matches:
            chosen_rater = matches[0][1]
            return candidate, chosen_rater, dicoms[candidate], [matches[0]]

    raise FileNotFoundError('No clip with a DICOM, the requested annotator, and at least one B-line was found.')


def check_test_output(stem: str, rater: str, destination: str) -> None:
    """
    Args:
        stem:  clip that was processed.
        rater:  annotator that was processed.
        destination:  directory where the test files were written.
    """
    image_dir = os.path.join(destination, 'images', stem)
    mask_dir = os.path.join(destination, 'masks', rater, stem)
    meta_path = os.path.join(destination, 'meta', f'{stem}.json')
    image_paths = sorted(
        os.path.join(image_dir, name) for name in os.listdir(image_dir) if name.endswith('.png')
    )
    mask_paths = sorted(
        os.path.join(mask_dir, name) for name in os.listdir(mask_dir) if name.endswith('.png')
    )
    if not image_paths or not mask_paths or not os.path.exists(meta_path):
        raise RuntimeError(f'Test output is incomplete under {destination}')

    image = imageio.imread(image_paths[0])
    mask = imageio.imread(mask_paths[len(mask_paths) // 2])
    if tuple(image.shape) != output_shape or image.dtype != np.uint8:
        raise RuntimeError(f'Expected a uint8 {output_shape} image, got {image.shape} {image.dtype}')
    if tuple(mask.shape) != output_shape or not set(np.unique(mask)).issubset({0, 255}):
        raise RuntimeError(f'Expected a binary uint8 {output_shape} mask, got {mask.shape} {np.unique(mask)}')

    foreground_frames = 0
    for path in mask_paths:
        frame_mask = imageio.imread(path)
        if int((frame_mask > 0).sum()) > 0:
            foreground_frames += 1
    if foreground_frames == 0:
        raise RuntimeError('Test masks were written, but none contain a B-line origin.')

    print(f'Test passed for {stem}, annotator {rater}')
    print(f'{len(image_paths)} images, {len(mask_paths)} masks, {foreground_frames} masks with a B-line origin')
    print(f'Output: {destination}')


def process_stem(stem: str, dicom_paths: list, annotation_paths: list, destination: str = None) -> None:
    """
    Decode one clip, write its frames once, and write one mask per annotated frame and rater.

    Args:
        stem:  shared file stem.
        dicom_paths:  copies of the same DICOM (one per rater folder).
        annotation_paths:  (json path, rater) pairs.
        destination:  output directory. Defaults to output_root.
    """
    if destination is None:
        destination = output_root
    loaded = []
    for json_path, rater in annotation_paths:
        with open(json_path, 'r') as handle:
            data = json.load(handle)
        loaded.append((json_path, rater, data))

    fan_sources = [item for item in loaded if all(key in item[2] for key in ('radius1', 'angle1', 'center_rows_px'))]
    if not fan_sources:
        raise ValueError(f'{stem} has no JSON fan geometry.')
    fan = fan_from_json(fan_sources[0][2])
    for _, rater, data in fan_sources[1:]:
        other = fan_from_json(data)
        delta = max(
            abs(float(other[key]) - float(fan[key]))
            for key in ('center_rows_px', 'center_cols_px', 'radius1', 'radius2')
        )
        if delta > 2:
            print(f'Warning: {stem} fan geometry differs for rater {rater}; using the first JSON for the image crop.')

    dicom_path = sorted(dicom_paths)[0]
    frames, spacing_mm = dicom_grayscale_frames(dicom_path)
    mask = fan_mask(frames.shape[1:], fan)
    geometry = geometry_from_fan(mask, output_shape)

    image_dir = os.path.join(destination, 'images', stem)
    os.makedirs(image_dir, exist_ok=True)
    for index, frame in enumerate(frames):
        masked = frame.copy()
        masked[~mask] = 0
        processed = apply_geometry(masked, geometry, output_shape, order=1)
        imwrite(os.path.join(image_dir, f'frame_{index:03d}.png'), processed)

    seen_raters = set()
    for json_path, rater, data in loaded:
        frame_annotations = data.get('frame_annotations')
        if not frame_annotations:
            continue
        if rater in seen_raters:
            print(f'Warning: duplicate annotations for {stem} rater {rater}; skipping {json_path}.')
            continue
        seen_raters.add(rater)
        mask_dir = os.path.join(destination, 'masks', rater, stem)
        os.makedirs(mask_dir, exist_ok=True)
        rater_fan = fan_from_json(data)
        skipped_blines = 0
        for frame_info in frame_annotations:
            frame_number = int(frame_info['frame_number'])
            if frame_number < 0 or frame_number >= len(frames):
                print(f'Warning: {json_path} frame {frame_number} is outside the DICOM ({len(frames)} frames).')
                continue
            label, skipped = rasterize_blines(
                frame_info.get('b_lines') or [],
                frame_info.get('pleura_lines') or [],
                frames.shape[1:],
                spacing_mm,
                rater_fan,
            )
            skipped_blines += skipped
            processed_label = apply_geometry(label, geometry, output_shape, order=0)
            processed_label = np.where(processed_label >= 128, 255, 0).astype(np.uint8)
            imwrite(os.path.join(mask_dir, f'frame_{frame_number:03d}.png'), processed_label)
        if skipped_blines:
            print(
                f'Warning: skipped {skipped_blines} B-lines for {stem} rater {rater} '
                'because no usable pleural boundary was available.'
            )

    meta_dir = os.path.join(destination, 'meta')
    os.makedirs(meta_dir, exist_ok=True)
    study = os.path.basename(os.path.dirname(dicom_path))
    payload = serializable_geometry(geometry, spacing_mm, study, dicom_path, stem)
    with open(os.path.join(meta_dir, f'{stem}.json'), 'w') as handle:
        json.dump(payload, handle, indent=2)


def main() -> None:
    global input_root, output_root
    parser = argparse.ArgumentParser(description='Preprocess Lahey DICOMs and B-line annotations.')
    parser.add_argument('--input-root', default=input_root, help='Directory of rater folders containing DICOM and JSON files.')
    parser.add_argument('--output-root', default=output_root, help='Directory for images, masks, and meta.')
    parser.add_argument('--test', action='store_true', help='Process one clip from one annotator and check the outputs.')
    parser.add_argument('--rater', default=test_rater, help='Annotator to use with --test, for example andrew.')
    parser.add_argument('--stem', default=test_stem, help='Clip stem to use with --test. Defaults to the first clip with B-lines.')
    args = parser.parse_args()
    input_root = args.input_root
    output_root = args.output_root
    run_test = args.test or test_one_clip

    if not os.path.isdir(input_root):
        raise FileNotFoundError(f'Input directory does not exist: {input_root}')
    os.makedirs(output_root, exist_ok=True)

    dicoms, annotations = index_inputs(input_root)
    if run_test:
        destination = os.path.join(output_root, 'test_one_clip')
        os.makedirs(destination, exist_ok=True)
        stem, rater, dicom_paths, annotation_paths = select_test_case(dicoms, annotations, args.rater, args.stem)
        print(f'Test clip {stem}, annotator {rater}')
        process_stem(stem, dicom_paths, annotation_paths, destination)
        check_test_output(stem, rater, destination)
        return

    stems = sorted(set(dicoms) | set(annotations))
    print(f'{len(dicoms)} clips, {sum(len(v) for v in annotations.values())} JSON files')

    for stem in tqdm(stems):
        if stem not in dicoms:
            print(f'Warning: {stem} has JSON annotations and no DICOM.')
            continue
        if stem not in annotations:
            print(f'Warning: {stem} has a DICOM and no JSON.')
            continue
        process_stem(stem, dicoms[stem], annotations[stem])

    print(f'Wrote images, masks, and meta to {output_root}')


if __name__ == '__main__':
    main()
