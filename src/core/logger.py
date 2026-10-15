import logging
import os
from multiprocessing import get_start_method
from common.config import SFTConfig

try:
    import multiprocessing_logging
except Exception:
    multiprocessing_logging = None


class PackagePathFilter(logging.Filter):
    def filter(self, record):
        record.pathname = record.pathname.replace(os.getcwd(), "")
        return True


class SFTLogger:
    def __init__(self, config: dict):
        if multiprocessing_logging is not None and get_start_method() == "fork":
            multiprocessing_logging.install_mp_handler()
        self.config_dict = config
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """_setup_logger Setup the logger

        Sets up a logger object with the configs, outputs to file and console
        if log_console is True

        Returns
        -------
        logging.Logger
            The logger
        """

        logger_config = self.config_dict

        logger = logging.getLogger("sft_cv")
        logger.setLevel(logger_config["log_level"])
        file_handler = logging.FileHandler(logger_config["log_file"])
        file_handler.setLevel(logger_config["log_level"])
        formatter = logging.Formatter(
            logger_config["log_format"], datefmt="%Y-%m-%d %H:%M:%S"
        )

        if logger_config["log_console"]:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logger_config["log_level"])
            console_handler.setFormatter(formatter)
            console_handler.addFilter(PackagePathFilter())
            logger.addHandler(console_handler)

        file_handler.addFilter(PackagePathFilter())
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        return logger

    def __getattr__(self, name: str):
        """__getattr__ Get the logger attribute

        This is a wrapper for the logger object, so you can use it like a normal logger

        Parameters
        ----------
        name : str
            The name of the attribute to get

        Returns
        -------
        Any
            The attribute of the logger object
        """
        return getattr(self.logger, name)

    def __getstate__(self):
        """Prepare state for pickling: exclude the actual logging.Logger which
        is not picklable. We'll store only the config dict and recreate the
        logger in the child process during unpickling.
        """
        state = self.__dict__.copy()
        # remove the live logger instance from the pickled state
        state.pop("logger", None)
        return state

    def __setstate__(self, state):
        """Restore state after unpickling and recreate the logger object."""
        self.__dict__.update(state)
        # Recreate the logger in the new process
        try:
            self.logger = self._setup_logger()
            if multiprocessing_logging is not None and get_start_method() == "fork":
                try:
                    multiprocessing_logging.install_mp_handler()
                except Exception:
                    pass
        except Exception:
            # Fallback: create a basic logger
            import logging

            self.logger = logging.getLogger("sft_cv")
            self.logger.addHandler(logging.NullHandler())
