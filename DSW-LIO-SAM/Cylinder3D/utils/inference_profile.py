"""Resolve a checkpoint together with the voxel space it was trained on."""

import hashlib
import os


SEMANTIC_KITTI_MIN_VOLUME = [0.0, -3.1415926, -4.0]
SEMANTIC_KITTI_MAX_VOLUME = [50.0, 3.1415926, 2.0]


def _absolute(project_root, path):
    if not path or os.path.isabs(path):
        return path
    return os.path.join(project_root, path)


def resolve_inference_profile(project_root, train_config, dataset_config):
    """Return checkpoint provenance and the compatible voxel-space settings."""
    fallback = _absolute(project_root, train_config['model_load_path'])
    preferred = _absolute(project_root, train_config.get('model_save_path', ''))

    if preferred and os.path.isfile(preferred):
        checkpoint = preferred
        model_kind = 'newer_college_finetuned'
        profile = 'newer_college_finetuned'
        min_volume = list(dataset_config['min_volume_space'])
        max_volume = list(dataset_config['max_volume_space'])
    else:
        checkpoint = fallback
        model_kind = 'generic_pretrained_fallback'
        profile = 'semantic_kitti_pretrained'
        min_volume = list(SEMANTIC_KITTI_MIN_VOLUME)
        max_volume = list(SEMANTIC_KITTI_MAX_VOLUME)

    if not checkpoint or not os.path.isfile(checkpoint):
        raise FileNotFoundError(f'Cylinder3D checkpoint not found: {checkpoint}')

    return {
        'checkpoint': os.path.abspath(checkpoint),
        'model_kind': model_kind,
        'preprocessing_profile': profile,
        'min_volume_space': min_volume,
        'max_volume_space': max_volume,
    }


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest().upper()
