from pathlib import Path

try:
    import multiprocessing_logging
except Exception:
    multiprocessing_logging = None

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2]
MISSION_CONFIG_PATH = CONFIG_ROOT / "mission_config.yaml"
UTILITY_CONFIG_PATH = CONFIG_ROOT / "utility_config.yaml"


class SFTConfig:
    def __init__(self, config_yaml: str, setup_logger=True) -> None:
        """__init__ SFTConfig

        Parameters
        ----------
        config_yaml : str
            The path to the configuration file
        """

        # Allow concurrent logging in multiprocessing (optional)
        if multiprocessing_logging is not None:
            try:
                multiprocessing_logging.install_mp_handler()
            except Exception:
                # best-effort: if installation fails, continue without it
                pass

        self.config_yaml = config_yaml
        self.config_dict = self._load_config()

    def _load_config(self) -> dict:
        """_load_config Load the configuration file

        Returns
        -------
        dict
            The configuration file as a dictionary
        """
        with open(self.config_yaml, "r") as f:
            config_dict = yaml.safe_load(f)
        return config_dict


if __name__ == "__main__":
    config = SFTConfig("sft_config.yaml")
