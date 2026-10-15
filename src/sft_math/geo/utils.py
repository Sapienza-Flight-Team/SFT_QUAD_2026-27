from sft_math.geo.constants import MAGNETIC_DECLINATION
from sft_math.typing import Array, Float
from numpy import typing as np_typing
import numpy as np


def truenorth2magnorth(
    truenorth: np_typing.ArrayLike,
) -> Array:
    """
    Convert true north to magnetic north by adding the magnetic declination.

    Parameters:
    truenorth (Array): The value representing geographic north.

    Returns:
    Array: The value representing magnetic north.
    """
    np_truenorth = np.array(truenorth, dtype=Float)
    return np_truenorth + MAGNETIC_DECLINATION


def magnorth2truenorth(
    magnorth: np_typing.ArrayLike,
) -> Array:
    """
    Convert magnetic north to true north by subtracting the magnetic declination.

    Parameters:
    magnorth (Array): The value representing magnetic north.

    Returns:
    Array: The value representing true north.
    """
    np_magnorth = np.array(magnorth, dtype=Float)
    return np_magnorth - MAGNETIC_DECLINATION
