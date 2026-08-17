from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG_NAMES = (
    "params_original_lio_sam.yaml",
    "params_baseline.yaml",
    "params_pseudo_semantic.yaml",
    "params_true_semantic.yaml",
)
ROT = (
    (-1.0, 0.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
)
TRANS = (-0.006253, 0.011775, -0.028535)


def _parameters(name):
    path = ROOT / "config" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))["/**"]["ros__parameters"]


def test_all_experiments_use_newer_college_extrinsic():
    for name in CONFIG_NAMES:
        params = _parameters(name)
        assert tuple(params["extrinsicTrans"]) == TRANS
        assert tuple(params["extrinsicRot"]) == tuple(value for row in ROT for value in row)
        assert tuple(params["extrinsicRPY"]) == tuple(value for row in ROT for value in row)


def test_inverse_pose_translation_rotates_before_negation():
    # For T_LI=(R,t), T_IL has t_IL=-R^T*t, not simply -t.
    inverse_translation = tuple(
        -sum(ROT[j][i] * TRANS[j] for j in range(3)) for i in range(3)
    )
    assert inverse_translation == (-0.006253, 0.011775, 0.028535)
