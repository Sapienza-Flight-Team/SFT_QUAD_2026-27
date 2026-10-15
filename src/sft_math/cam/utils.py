import numpy as np
from sft_math.typing import Array, Float
from typing import Tuple
from numpy import typing as np_typing


def get_rpy(matrix: np_typing.ArrayLike) -> Tuple[Array, Array, Array]:
    matrix = np.array(np.array(matrix, dtype=Float), copy=False, subok=True, ndmin=2)
    assert matrix.shape[-2:] == (3, 3), "shape of matrix must be 3,3"
    assert np.allclose(
        np.einsum("...rk, ...ck -> ...rc", matrix, matrix),
        np.tile(np.eye(3), matrix.shape[:-2] + (1, 1)),
        atol=1e-6,
    ), "matrix must be orthonormal matrix"
    assert np.allclose(
        np.linalg.det(matrix), 1
    ), "matrix must be with positive determinant"
    sp = -matrix[..., 2, 0]
    cp = np.array(np.sqrt(matrix[..., 0, 0] ** 2 + matrix[..., 1, 0] ** 2), dtype=Float)
    # disattivo il warning per la divisione per 0
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = cp == 0
        sr = np.where(mask, np.zeros_like(mask), matrix[..., 2, 1] / cp)
        cr = np.where(mask, np.ones_like(mask), matrix[..., 2, 2] / cp)
        sy = np.where(mask, -matrix[..., 0, 1], matrix[..., 1, 0] / cp)
        cy = np.where(mask, matrix[..., 1, 1], matrix[..., 0, 0] / cp)
    roll = np.array(np.arctan2(sr, cr), dtype=Float)
    pitch = np.array(np.arctan2(sp, cp), dtype=Float)
    yaw = np.array(np.arctan2(sy, cy), dtype=Float)
    return (np.rad2deg(roll), np.rad2deg(pitch), np.rad2deg(yaw))


def get_mat(
    roll: np_typing.ArrayLike,
    pitch: np_typing.ArrayLike,
    yaw: np_typing.ArrayLike,
    is_deg: bool = True,
) -> Array:
    if is_deg:
        roll = np.array(np.deg2rad(roll), dtype=Float)
        pitch = np.array(np.deg2rad(pitch), dtype=Float)
        yaw = np.array(np.deg2rad(yaw), dtype=Float)
    cr = np.array(np.cos(roll), dtype=Float)
    sr = np.array(np.sin(roll), dtype=Float)
    cp = np.array(np.cos(pitch), dtype=Float)
    sp = np.array(np.sin(pitch), dtype=Float)
    cy = np.array(np.cos(yaw), dtype=Float)
    sy = np.array(np.sin(yaw), dtype=Float)

    # make matrix xyz2ned
    return np.stack(
        [
            np.stack(
                [cy * cp, cy * sp * sr - sy * cr, sy * sr + cy * sp * cr],
                dtype=Float,
                axis=-1,
            ),  # (..., ->3)
            np.stack(
                [sy * cp, cy * cr + sy * sp * sr, sy * sp * cr - cy * sr],
                dtype=Float,
                axis=-1,
            ),  # (..., ->3)
            np.stack([-sp, cp * sr, cp * cr], dtype=Float, axis=-1),  # (..., ->3)
        ],
        dtype=Float,
        axis=-2,
    )  # (..., ->3, 3)
