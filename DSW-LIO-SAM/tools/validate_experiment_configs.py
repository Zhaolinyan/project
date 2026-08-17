#!/usr/bin/env python3
"""Validate that the four experiment configs differ only by treatment fields."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "original": ROOT / "config" / "params_original_lio_sam.yaml",
    "baseline": ROOT / "config" / "params_baseline.yaml",
    "pseudo": ROOT / "config" / "params_pseudo_semantic.yaml",
    "true": ROOT / "config" / "params_true_semantic.yaml",
}
TREATMENT_FIELDS = {
    "savePCDDirectory",
    "originalLioSamMode",
    "semanticEnabled",
    "semanticWeightAlpha",
    "requireSemanticCloud",
    "minimumSemanticCoverage",
    "pseudoGroundMaxZ",
    "pseudoStructureMinZ",
    "pseudoCanopyMinZ",
}
NEWER_COLLEGE_EXT_TRANS = [-0.006253, 0.011775, -0.028535]
NEWER_COLLEGE_EXT_ROT = [
    -1.0, 0.0, 0.0,
    0.0, -1.0, 0.0,
    0.0, 0.0, 1.0,
]


def load_parameters(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["/**"]["ros__parameters"]


def main():
    configs = {name: load_parameters(path) for name, path in CONFIGS.items()}
    reference_keys = set(configs["baseline"])
    key_mismatches = []
    for name, config in configs.items():
        keys = set(config)
        if keys != reference_keys:
            key_mismatches.append(
                f"  {name}: missing={sorted(reference_keys - keys)} "
                f"extra={sorted(keys - reference_keys)}"
            )
    if key_mismatches:
        raise SystemExit("Experiment config key mismatch:\n" + "\n".join(key_mismatches))

    common_keys = set.intersection(*(set(values) for values in configs.values()))
    unexpected = []
    for key in sorted(common_keys - TREATMENT_FIELDS):
        values = {name: config[key] for name, config in configs.items()}
        if len({repr(value) for value in values.values()}) != 1:
            unexpected.append((key, values))

    if unexpected:
        details = "\n".join(f"  {key}: {values}" for key, values in unexpected)
        raise SystemExit(f"Unexpected experiment confounders:\n{details}")

    original, baseline, pseudo, true = (
        configs[name] for name in ("original", "baseline", "pseudo", "true")
    )
    assert original["originalLioSamMode"] is True
    assert original["semanticEnabled"] is False
    assert original["semanticWeightAlpha"] == 0.0
    assert original["requireSemanticCloud"] is False
    assert baseline["semanticEnabled"] is False
    assert baseline["semanticWeightAlpha"] == 0.0
    assert baseline["requireSemanticCloud"] is False
    assert baseline["originalLioSamMode"] is False
    assert pseudo["originalLioSamMode"] is False
    assert true["originalLioSamMode"] is False
    assert pseudo["semanticEnabled"] is True and true["semanticEnabled"] is True
    assert pseudo["semanticWeightAlpha"] == true["semanticWeightAlpha"]
    assert pseudo["degeneracyThreshold"] == true["degeneracyThreshold"]
    assert pseudo["minimumSemanticCoverage"] == true["minimumSemanticCoverage"]
    assert pseudo["requireSemanticCloud"] is True and true["requireSemanticCloud"] is True
    assert all(config["loopClosureEnableFlag"] is False for config in configs.values())
    for name, config in configs.items():
        assert config["extrinsicTrans"] == NEWER_COLLEGE_EXT_TRANS, name
        assert config["extrinsicRot"] == NEWER_COLLEGE_EXT_ROT, name
        assert config["extrinsicRPY"] == NEWER_COLLEGE_EXT_ROT, name
    print("Four experiment configs are controlled; no unexpected parameter differences.")


if __name__ == "__main__":
    main()
