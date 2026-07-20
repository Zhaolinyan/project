"""Pure helpers for deterministic offline semantic replay."""

import csv
import bisect
import json
import os

import numpy as np


FLOAT32 = 7
UINT16 = 4
UINT8 = 2
SEMANTIC_DTYPE = np.dtype([
    ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
    ('intensity', '<f4'), ('label', '<u4'),
])


def stamp_to_ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _field(msg, names):
    names = set(names)
    for field in msg.fields:
        if field.name.strip('\x00').strip() in names:
            return field
    return None


def _read_field(rows, field, dtype, width):
    values = rows[:, field.offset:field.offset + width].copy()
    return values.view(dtype).reshape(-1)


def parse_raw_cloud(msg):
    if msg.is_bigendian:
        raise ValueError('big-endian PointCloud2 is not supported')
    count = int(msg.width) * int(msg.height)
    if count == 0:
        raise ValueError('empty PointCloud2')
    point_step = int(msg.point_step)
    row_step = int(getattr(msg, 'row_step', 0)) or int(msg.width) * point_step
    data = bytes(msg.data)
    expected_bytes = int(msg.height) * row_step
    if len(data) < expected_bytes:
        raise ValueError(
            f'PointCloud2 data is truncated: expected={expected_bytes} actual={len(data)}')
    rows = np.ndarray(
        (int(msg.height), int(msg.width), point_step), dtype=np.uint8,
        buffer=data, strides=(row_step, point_step, 1)).copy().reshape(count, point_step)

    xyz = np.empty((count, 3), dtype=np.float32)
    for column, name in enumerate(('x', 'y', 'z')):
        field = _field(msg, [name])
        if field is None or field.datatype != FLOAT32:
            raise ValueError(f'missing FLOAT32 field {name!r}')
        xyz[:, column] = _read_field(rows, field, '<f4', 4)

    intensity_field = _field(msg, ['intensity', 'signal', 'reflectivity'])
    if intensity_field is None:
        intensity = np.zeros(count, dtype=np.float32)
    elif intensity_field.datatype == FLOAT32:
        intensity = _read_field(rows, intensity_field, '<f4', 4).astype(np.float32)
    elif intensity_field.datatype == UINT16:
        intensity = _read_field(rows, intensity_field, '<u2', 2).astype(np.float32)
    elif intensity_field.datatype == UINT8:
        intensity = _read_field(rows, intensity_field, np.uint8, 1).astype(np.float32)
    else:
        raise ValueError(f'unsupported intensity datatype {intensity_field.datatype}')

    valid = np.isfinite(xyz).all(axis=1)
    valid &= np.linalg.norm(xyz, axis=1) > 1e-3
    return xyz, intensity, valid


def _coordinate_lookup_labels(semantic_frame, tolerance):
    scale = 1.0 / tolerance
    xyz = np.column_stack((
        semantic_frame['x'], semantic_frame['y'], semantic_frame['z']))
    keys = np.rint(xyz * scale).astype(np.int64)
    labels = {}
    for key, label in zip(map(tuple, keys), semantic_frame['label']):
        labels.setdefault(key, int(label))
    return labels


def find_frame_for_stamp(frames, sorted_stamps, stamp_ns, tolerance_ns=0):
    if not sorted_stamps:
        return None, None
    if stamp_ns in frames:
        return stamp_ns, frames[stamp_ns]

    pos = bisect.bisect_left(sorted_stamps, stamp_ns)
    candidates = []
    if pos < len(sorted_stamps):
        candidates.append(sorted_stamps[pos])
    if pos > 0:
        candidates.append(sorted_stamps[pos - 1])
    if not candidates:
        return None, None

    best_stamp = min(candidates, key=lambda candidate: abs(candidate - stamp_ns))
    if abs(best_stamp - stamp_ns) > tolerance_ns:
        return None, None
    return best_stamp, frames[best_stamp]


def align_semantic_frame(
        msg, semantic_frame, allow_coordinate_lookup=False, tolerance=1e-3,
        minimum_coverage=0.0):
    xyz, intensity, valid = parse_raw_cloud(msg)
    valid_count = int(valid.sum())
    if valid_count != len(semantic_frame) and not allow_coordinate_lookup:
        raise ValueError(
            f'point count mismatch: raw-valid={valid_count} semantic={len(semantic_frame)}')

    aligned = np.zeros(len(xyz), dtype=SEMANTIC_DTYPE)
    aligned['x'] = xyz[:, 0]
    aligned['y'] = xyz[:, 1]
    aligned['z'] = xyz[:, 2]
    aligned['intensity'] = intensity

    semantic_xyz = np.column_stack((
        semantic_frame['x'], semantic_frame['y'], semantic_frame['z']))
    if valid_count == len(semantic_frame) and np.allclose(
            xyz[valid], semantic_xyz, rtol=0.0, atol=1e-4):
        aligned['label'][valid] = semantic_frame['label']
        return aligned

    if not allow_coordinate_lookup:
        if valid_count != len(semantic_frame):
            raise ValueError(
                f'point count mismatch: raw-valid={valid_count} semantic={len(semantic_frame)}')
        max_error = float(np.max(np.abs(xyz[valid] - semantic_xyz)))
        raise ValueError(f'semantic geometry does not match raw frame; max error={max_error}')

    labels_by_key = _coordinate_lookup_labels(semantic_frame, tolerance)
    scale = 1.0 / tolerance
    valid_indices = np.flatnonzero(valid)
    raw_keys = np.rint(xyz[valid] * scale).astype(np.int64)
    matched = 0
    for raw_index, key in zip(valid_indices, map(tuple, raw_keys)):
        label = labels_by_key.get(key)
        if label is not None:
            aligned['label'][raw_index] = label
            matched += 1

    if matched == 0:
        raise ValueError(
            f'no coordinate matches between raw-valid={valid_count} and semantic={len(semantic_frame)}')
    coverage = matched / valid_count if valid_count else 0.0
    if coverage < minimum_coverage:
        raise ValueError(
            f'semantic coordinate coverage {coverage:.3f} below required {minimum_coverage:.3f}')
    return aligned


def load_manifest(semantic_dir):
    manifest_path = os.path.join(semantic_dir, 'frames.csv')
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(
            f'missing {manifest_path}; regenerate bins and merge labels before replay')

    frames = {}
    with open(manifest_path, newline='', encoding='utf-8') as stream:
        reader = csv.DictReader(stream)
        required = {'filename', 'stamp_ns', 'point_count', 'original_point_count'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'invalid manifest columns in {manifest_path}')
        for row in reader:
            stamp_ns = int(row['stamp_ns'])
            if stamp_ns in frames:
                raise ValueError(f'duplicate timestamp {stamp_ns} in manifest')
            frames[stamp_ns] = {
                'path': os.path.join(semantic_dir, row['filename']),
                'point_count': int(row['point_count']),
                'original_point_count': int(row['original_point_count']),
            }
    if not frames:
        raise ValueError(f'empty manifest: {manifest_path}')
    return frames


def validate_inference_metadata(semantic_dir):
    metadata_path = os.path.join(semantic_dir, 'inference_metadata.json')
    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(
            f'missing {metadata_path}; regenerate Cylinder3D labels with the current code')
    with open(metadata_path, encoding='utf-8') as stream:
        metadata = json.load(stream)

    model_kind = metadata.get('model_kind')
    profile = metadata.get('preprocessing_profile')
    compatible = {
        'generic_pretrained_fallback': 'semantic_kitti_pretrained',
        'newer_college_finetuned': 'newer_college_finetuned',
    }
    if model_kind not in compatible or profile != compatible[model_kind]:
        raise ValueError(
            'semantic labels use an unknown or checkpoint-incompatible preprocessing '
            f'profile: model_kind={model_kind!r} profile={profile!r}; regenerate labels')
    return metadata
