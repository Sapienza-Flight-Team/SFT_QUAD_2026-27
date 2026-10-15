from sft_math.geo.polar import GPS
from sft_math.typing import Array, Float
import numpy as np


def from_vec(vec: Array) -> GPS:
    """Creates a GPS coordinate from a vector

    Args:
        vec (Array): vector with shape (..., 3)

    Returns:
        GPS: GPS coordinate
    """
    vec = np.array(vec, dtype=Float)
    if vec.shape[-1] != 3:
        raise ValueError("The last dimension of the vector must be 3")
    return GPS(
        lat=np.arcsin(np.clip(vec[..., 2] / np.linalg.norm(vec, axis=-1), -1.0, 1.0)),
        lon=np.arctan2(vec[..., 1], vec[..., 0]),
        is_deg=False,
    )


def barycenter(gps_list: list[GPS]) -> GPS:
    """Computes the barycenter of a list of GPS coordinates

    Args:
        gps_list (list[GPS]): list of GPS coordinates

    Returns:
        GPS: the barycenter of the GPS coordinates
    """
    if len(gps_list) == 0:
        raise ValueError("at least one GPS coordinate is required")
    try:
        np.broadcast_shapes(*[gps.vec().shape for gps in gps_list])
    except ValueError:
        raise ValueError("The shapes of the GPS coordinates must be broadcastable")

    gpsArray: Array = np.sum([gps.vec() for gps in gps_list], axis=0)
    if np.linalg.norm(gpsArray) == 0:
        raise RuntimeError("The barycenter is the origin")
    return from_vec(gpsArray)
