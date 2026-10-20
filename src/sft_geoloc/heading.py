"""Heading conversion in degrees, with explicit east-positive declination.

Supply declination for the flight location and date; no fixed site value is
assumed. Heading and declination arrays broadcast and results wrap to [0, 360).
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray

from . import check


def magnetic_to_true(
    heading_deg: ArrayLike, declination_deg: ArrayLike
) -> NDArray[np.float64]:
    """Add east-positive magnetic declination to a magnetic heading."""
    heading = np.asarray(heading_deg, dtype=np.float64)
    declination = np.asarray(declination_deg, dtype=np.float64)
    check.finite(heading, name="heading_deg")
    check.finite(declination, name="declination_deg")
    return np.asarray((heading + declination) % 360)


def true_to_magnetic(
    heading_deg: ArrayLike, declination_deg: ArrayLike
) -> NDArray[np.float64]:
    """Subtract east-positive magnetic declination from a true heading."""
    return magnetic_to_true(heading_deg, -np.asarray(declination_deg, dtype=np.float64))
