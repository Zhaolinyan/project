#!/usr/bin/env python3
"""Validate that the three experiment configs differ only by treatment fields."""

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "baseline": ROOT / "config" / "params_baseline.yaml",
    "pseudo": ROOT / "config" / "params_pseudo_semantic.yaml",
    "true": ROOT / "config" / "params_true_semantic.yaml",
}
TREATMENT_FIELDS = {
    "savePCDDirectory",
    "semanticEnabled",
    "semanticWeightAlpha",
    "requireSemanticCloud",
    "minimumSemanticCoverage",
    "pseudoGroundMaxZ",
    "pseudoStructureMinZ",
    "pseudoCanopyMinZ",
}


def load_parameters(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data["/**"]["ros__parameters"]


def main():
    configs = {name: load_parameters(path) for name, path in CONFIGS.items()}
    common_keys = set.intersection(*(set(values) for values in configs.values()))
    unexpected = []
    for key in sorted(common_keys - TREATMENT_FIELDS):
        values = {name: config[key] for name, config in configs.items()}
        if len({repr(value) for value in values.values()}) != 1:
            unexpected.append((key, values))

    if unexpected:
        details = "\n".join(f"  {key}: {values}" for key, values in unexpected)
        raise SystemExit(f"Unexpected experiment confounders:\n{details}")

    baseline, pseudo, true = (configs[name] for name in ("baseline", "pseudo", "true"))
    assert baseline["semanticEnabled"] is False
    assert baseline["semanticWeightAlpha"] == 0.0
    assert baseline["requireSemanticCloud"] is False
    assert baseline["loopClosureEnableFlag"] is False
    assert pseudo["semanticEnabled"] is True and true["semanticEnabled"] is True
    assert pseudo["semanticWeightAlpha"] == true["semanticWeightAlpha"]
    assert pseudo["degeneracyThreshold"] == true["degeneracyThreshold"]
    assert pseudo["minimumSemanticCoverage"] == true["minimumSemanticCoverage"]
    assert pseudo["requireSemanticCloud"] is True and true["requireSemanticCloud"] is True
    assert pseudo["loopClosureEnableFlag"] is False
    assert true["loopClosureEnableFlag"] is False
    print("Experiment configs are controlled; no unexpected parameter differences.")


if __name__ == "__main__":
    main()
