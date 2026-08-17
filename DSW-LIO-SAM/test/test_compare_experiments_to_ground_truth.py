import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
SPEC = importlib.util.spec_from_file_location(
    "compare_experiments_to_ground_truth",
    ROOT / "tools" / "compare_experiments_to_ground_truth.py")
COMPARISON = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(COMPARISON)


def test_flatten_result_exposes_endpoint_ate_and_rpe():
    result = {
        "association": {"matched_poses": 10, "estimated_poses": 12},
        "endpoint": {
            "estimated_start_to_end_m": 3.0,
            "ground_truth_start_to_end_m": 2.5,
            "start_to_end_distance_error_m": 0.5,
            "aligned_start_error_m": 0.1,
            "aligned_end_error_m": 0.2,
            "relative_displacement_error_m": 0.3,
        },
        "ate_translation_m": {"rmse": 0.4, "max": 0.8},
        "rpe": {
            "pairs": 8,
            "translation_m": {"rmse": 0.6},
            "rotation_deg": {"rmse": 1.2},
        },
    }

    row = COMPARISON.flatten_result("original", result)

    assert row["experiment"] == "original"
    assert row["start_to_end_distance_error_m"] == 0.5
    assert row["ate_rmse_m"] == 0.4
    assert row["rpe_translation_rmse_m"] == 0.6
    assert row["rpe_rotation_rmse_deg"] == 1.2
