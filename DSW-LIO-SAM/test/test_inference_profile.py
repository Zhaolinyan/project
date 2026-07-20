import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    'inference_profile', ROOT / 'Cylinder3D' / 'utils' / 'inference_profile.py')
PROFILE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROFILE)


def test_generic_checkpoint_uses_training_voxel_space(tmp_path):
    checkpoint = tmp_path / 'generic.pt'
    checkpoint.write_bytes(b'checkpoint')
    dataset = {
        'min_volume_space': [0, -3.14, -5],
        'max_volume_space': [50, 3.14, 15],
    }
    result = PROFILE.resolve_inference_profile(
        str(tmp_path),
        {'model_load_path': 'generic.pt', 'model_save_path': 'missing.pt'},
        dataset,
    )

    assert result['model_kind'] == 'generic_pretrained_fallback'
    assert result['min_volume_space'] == PROFILE.SEMANTIC_KITTI_MIN_VOLUME
    assert result['max_volume_space'] == PROFILE.SEMANTIC_KITTI_MAX_VOLUME


def test_finetuned_checkpoint_keeps_configured_voxel_space(tmp_path):
    (tmp_path / 'generic.pt').write_bytes(b'generic')
    (tmp_path / 'finetuned.pt').write_bytes(b'finetuned')
    dataset = {
        'min_volume_space': [0, -3.14, -5],
        'max_volume_space': [50, 3.14, 15],
    }
    result = PROFILE.resolve_inference_profile(
        str(tmp_path),
        {'model_load_path': 'generic.pt', 'model_save_path': 'finetuned.pt'},
        dataset,
    )

    assert result['model_kind'] == 'newer_college_finetuned'
    assert result['min_volume_space'] == dataset['min_volume_space']
    assert result['max_volume_space'] == dataset['max_volume_space']
