from abc import ABC, abstractmethod
from logging import Logger
from typing import Dict, Optional, Any
import numpy as np
from numpy import typing as np_typing
from logging import Logger
from typing import Dict, Optional, Any, Tuple
import numpy as np
from numpy import typing as np_typing
from sft_math.geo import GeoGPS
from logging import Logger
from typing import Dict, Optional, Any


class ClassHandlerTemplate(ABC):

    class Failure(Exception):
        pass

    @abstractmethod
    def __init__(self, logger: Logger, configs: Dict[str, Optional[Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(self, classes: np_typing.ArrayLike):
        """
        This method should be implemented by the derived class to handle the classes.
        """
        raise NotImplementedError


class GeolocTemplate(ABC):

    class Failure(Exception):
        pass

    @abstractmethod
    def __init__(self, logger: Logger, configs: Dict[str, Optional[Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def __call__(
        self,
        pixels_coordinates: np_typing.NDArray,
        gps_lat: float,
        gps_lon: float,
        height: float,
        roll: float,
        pitch: float,
        yaw: float,
        tmp_path: str,
    ) -> Tuple[np_typing.NDArray[np.float64], GeoGPS]:
        raise NotImplementedError

    @abstractmethod
    def dump_data(self, tmp_path: str) -> None:
        raise NotImplementedError


class DecisionTemplate(ABC):

    class Failure(Exception):
        pass

    @abstractmethod
    def __init__(self, logger: Logger, configs: Dict[str, Optional[Any]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def insert(
        self,
        points: np_typing.ArrayLike,
        covariances: np_typing.ArrayLike,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def fit(
        self,
        n_clusters: int,
    ) -> np_typing.NDArray[np.float64]:
        raise NotImplementedError
