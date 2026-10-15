from pathlib import Path

import yaml

from tools.config_file_generation.sft_config_dict import STFCONFIGDICT


def test_base_sft_config_does_not_expose_inference_pipeline_topology() -> None:
    assert "inference_pipeline" not in STFCONFIGDICT

    loaded = yaml.safe_load(Path("sft_config.yaml").read_text(encoding="utf-8"))

    assert "inference_pipeline" not in loaded


def test_mission_config_contains_no_inference_pipeline_overrides() -> None:
    loaded = yaml.safe_load(Path("mission_config.yaml").read_text(encoding="utf-8"))

    for entry in loaded:
        for implementations in entry.values():
            for implementation in implementations:
                for options in implementation.values():
                    assert all("inference_pipeline" not in option for option in options)
