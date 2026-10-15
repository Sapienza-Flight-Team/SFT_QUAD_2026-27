import pytest

from common.config import SFTConfig


@pytest.fixture
def config_obj() -> SFTConfig:
    """Returns a config object"""
    return SFTConfig("sft_config.yaml")


def test_load_config(config_obj: SFTConfig) -> None:
    """test_load_config Test that the config is loaded correctly"""
    assert config_obj.config_dict["logger"]["log_level"] is not None
