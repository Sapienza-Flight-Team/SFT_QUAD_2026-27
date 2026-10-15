from sft_math.typing import Float, Array
from sft_math.cam.affine import Affine, Point
from sft_math.class_templates import ClassBase
import numpy as np
from numpy import typing as np_typing
from typing import Tuple


class Camera(ClassBase):

    class Projection(ClassBase):

        ratio: Array
        shape: Tuple[int, ...]

        __constants__ = [
            (lambda x: 1000 * x, "ratio", "mm/px"),
        ]

        def __init__(
            self,
            ratio: np_typing.ArrayLike,  # rapporto grandezza dei pixel e distanza focale in mm/px
        ):
            self.ratio = np.array(
                np.atleast_1d(np.array(ratio, dtype=Float)) / 1000
            ).astype(Float)
            self.shape = self.ratio.shape
            if np.any(self.ratio <= 0):
                raise ValueError("The pixel size must be positive")

        def out_plane(self, pxSpace: Affine) -> Affine:
            try:
                shape = np.broadcast_shapes(pxSpace.shape, self.shape)
            except ValueError:
                raise ValueError(
                    "The shapes of pxSpace and Projection must be broadcastable"
                )
            else:
                pxSpace = (pxSpace * self.ratio).broadcast_to(shape)
                vector = np.pad(
                    pxSpace.vector,
                    tuple((0, 0) for _ in range(len(pxSpace.shape))) + ((0, 1),),
                    constant_values=1,
                )
                generator = np.pad(
                    pxSpace.orthCoos2globalCoos_matrix,
                    tuple((0, 0) for _ in range(len(pxSpace.shape))) + ((0, 1), (0, 0)),
                )
                generator = np.pad(
                    generator,
                    tuple((0, 0) for _ in range(len(pxSpace.shape))) + ((0, 0), (0, 1)),
                )
                generator[..., :, -1] = vector
                return Affine(np.zeros_like(vector), generator).squeeze()

        def in_plane(self, dirSpace: Affine) -> Affine:
            try:
                shape = np.broadcast_shapes(self.shape, dirSpace.shape)
            except:
                raise ValueError(
                    "The shapes of dirSpace and Projection must be broadcastable"
                )
            else:
                vector = np.zeros(shape + (dirSpace.dimension,))
                generator = np.zeros(
                    shape + (dirSpace.dimension, dirSpace.dimension - 1)
                )
                vector[..., -1] = 1
                generator[..., :, :] = np.eye(dirSpace.dimension)[:, :-1]
                interSpace = Affine(vector, generator) & dirSpace
                vector = interSpace.vector[..., :-1]
                generator = interSpace.orthCoos2globalCoos_matrix[..., :-1, :]
                return Affine(vector / self.ratio[..., None], generator).squeeze()

        def __getitem__(self, key: Tuple) -> "Camera.Projection":
            return Camera.Projection(self.ratio[key] * 1000)

        def reshape(self, shape: Tuple[int, ...]) -> "Camera.Projection":
            return Camera.Projection(self.ratio.reshape(shape) * 1000)

        def broadcast_to(self, shape: Tuple[int, ...]) -> "Camera.Projection":
            return Camera.Projection(np.broadcast_to(self.ratio * 1000, shape))

    class Configuration(ClassBase):

        position: Affine
        roll: Array
        pitch: Array
        yaw: Array
        shape: Tuple[int, ...]

        __constants__ = [
            (lambda x: x, "position", ""),
            (lambda x: np.rad2deg(x), "roll", "°"),
            (lambda x: np.rad2deg(x), "pitch", "°"),
            (lambda x: np.rad2deg(x), "yaw", "°"),
        ]

        def __init__(
            self,
            position: Affine,
            roll: np_typing.ArrayLike,
            pitch: np_typing.ArrayLike,
            yaw: np_typing.ArrayLike,
            is_deg: bool = False,
        ):
            nproll = np.atleast_1d(np.array(roll, dtype=Float))
            nppitch = np.atleast_1d(np.array(pitch, dtype=Float))
            npyaw = np.atleast_1d(np.array(yaw, dtype=Float))
            try:
                shape = np.broadcast_shapes(
                    nproll.shape, nppitch.shape, npyaw.shape, position.shape
                )
                nproll = np.broadcast_to(nproll, shape)
                nppitch = np.broadcast_to(nppitch, shape)
                npyaw = np.broadcast_to(npyaw, shape)
            except:
                raise ValueError("angles and position must be broadcastable")
            else:
                if is_deg:
                    nproll = np.deg2rad(nproll)
                    nppitch = np.deg2rad(nppitch)
                    npyaw = np.deg2rad(npyaw)
                self.roll = nproll
                self.pitch = nppitch
                self.yaw = npyaw
                self.position = position
                self.shape = shape

        def xyz2ned(self, xyzSpace: Affine) -> Affine:
            cr = np.cos(self.roll)
            sr = np.sin(self.roll)
            cp = np.cos(self.pitch)
            sp = np.sin(self.pitch)
            cy = np.cos(self.yaw)
            sy = np.sin(self.yaw)

            # make matrix xyz2ned
            matrix = np.stack(
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
                    np.stack(
                        [-sp, cp * sr, cp * cr], dtype=Float, axis=-1
                    ),  # (..., ->3)
                ],
                dtype=Float,
                axis=-2,
            )  # (..., ->3, 3)

            return self.position + xyzSpace.change_vec(matrix)

        def ned2xyz(self, nedSpace: Affine) -> Affine:
            cr = np.cos(self.roll)
            sr = np.sin(self.roll)
            cp = np.cos(self.pitch)
            sp = np.sin(self.pitch)
            cy = np.cos(self.yaw)
            sy = np.sin(self.yaw)

            # make matrix xyz2ned
            matrix = np.stack(
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
                    np.stack(
                        [-sp, cp * sr, cp * cr], dtype=Float, axis=-1
                    ),  # (..., ->3)
                ],
                dtype=Float,
                axis=-1,
            )  # (..., 3, ->3)

            return (nedSpace + (-self.position)).change_vec(matrix)

        def __getitem__(self, key: Tuple) -> "Camera.Configuration":
            return Camera.Configuration(
                self.position[key],
                self.roll[key],
                self.pitch[key],
                self.yaw[key],
                is_deg=False,
            )

        def reshape(self, shape: Tuple[int, ...]) -> "Camera.Configuration":
            return Camera.Configuration(
                self.position.reshape(shape),
                self.roll.reshape(shape),
                self.pitch.reshape(shape),
                self.yaw.reshape(shape),
                is_deg=False,
            )

        def broadcast_to(self, shape: Tuple[int, ...]) -> "Camera.Configuration":
            return Camera.Configuration(
                self.position.broadcast_to(shape),
                np.broadcast_to(self.roll, shape),
                np.broadcast_to(self.pitch, shape),
                np.broadcast_to(self.yaw, shape),
                is_deg=False,
            )

    projection: Projection
    image_shape: Tuple[Array, Array]
    optical_center: Point
    px2body_matrix: Array
    shape: Tuple[int, ...]

    __constants__ = [
        (lambda x: x, "projection", ""),
        (lambda x: x, "image_shape", "px"),
        (lambda x: x, "optical_center", "px"),
        (lambda x: x, "shape", ""),
    ]

    def __init__(
        self,
        focal_lenght: np_typing.ArrayLike,  # distanza focale in millimetri
        pixel_size: np_typing.ArrayLike,  # grandezza del sensore pixel in micrometri
        image_shape: Tuple[
            np_typing.ArrayLike, np_typing.ArrayLike
        ],  # grandezza dell'immagine secondo le coordinate pixel
        optical_center: np_typing.ArrayLike,  # centro ottico in coordinate pixel
        px2body_matrix: np_typing.ArrayLike,  # matrice che converte le coordinate pixel in coordinate corpo
    ):
        focal_lenght = np.atleast_1d(np.array(focal_lenght, dtype=Float))
        pixel_size = np.atleast_1d(np.array(pixel_size, dtype=Float))
        self.projection = Camera.Projection(np.divide(pixel_size, focal_lenght))
        self.image_shape = (
            np.atleast_1d(np.array(image_shape[0], dtype=Float)),
            np.atleast_1d(np.array(image_shape[1], dtype=Float)),
        )
        self.optical_center = Point(
            np.atleast_1d(np.array(optical_center, dtype=Float)), ((), ())
        )
        self.px2body_matrix = np.array(
            np.array(px2body_matrix, dtype=Float), copy=False, subok=True, ndmin=3
        )
        try:
            self.shape = np.broadcast_shapes(
                self.projection.shape,
                self.image_shape[0].shape,
                self.image_shape[1].shape,
                self.optical_center.shape,
                self.px2body_matrix.shape[:-2],
            )
        except:
            raise ValueError("input not broadcastable")
        else:
            if np.any(self.image_shape[0] <= 0) or np.any(self.image_shape[1] <= 0):
                raise ValueError("width and height must be positive")

    def px2ned(self, affine: Affine, configuration: Configuration) -> Affine:
        if affine.dimension != 2:
            raise ValueError("affine.dimension must be 2")
        try:
            np.broadcast_shapes(affine.shape, self.shape, configuration.shape)
        except:
            raise ValueError("affine.shape and self.shape must be broadcastable")
        else:
            affine = (affine - self.optical_center).change_vec(self.px2body_matrix)
            affine = self.projection.out_plane(affine)
            return configuration.xyz2ned(affine)

    def ned2px(self, affine: Affine, configuration: Configuration) -> Affine:
        if affine.dimension != 3:
            raise ValueError("affine.dimension must be 3")
        try:
            np.broadcast_shapes(affine.shape, self.shape)
        except:
            raise ValueError("affine.shape and self.shape must be broadcastable")
        else:
            affine = configuration.ned2xyz(affine)
            affine = self.projection.in_plane(affine)
            affine = affine.change_vec(
                np.moveaxis(self.px2body_matrix, (-1, -2), (-2, -1))
            )
            return affine + self.optical_center

    def __getitem__(self, key: Tuple) -> "Camera":
        return Camera(
            np.ones_like(self.projection.ratio[key]),
            self.projection.ratio[key] * 1000,
            (
                self.image_shape[0][key],
                self.image_shape[1][key],
            ),
            self.optical_center.vector[key],
            self.px2body_matrix[key],
        )

    def reshape(self, shape: Tuple[int, ...]) -> "Camera":
        return Camera(
            np.ones(shape),
            np.reshape(self.projection.ratio * 1000, shape),
            (
                np.reshape(self.image_shape[0], shape),
                np.reshape(self.image_shape[1], shape),
            ),
            np.reshape(self.optical_center.vector, shape + (2,)),
            np.reshape(
                self.px2body_matrix,
                shape + (self.px2body_matrix.shape[-2], self.px2body_matrix.shape[-1]),
            ),
        )

    def broadcast_to(self, shape: Tuple[int, ...]) -> "Camera":
        return Camera(
            np.ones(shape),
            np.broadcast_to(self.projection.ratio * 1000, shape),
            (
                np.broadcast_to(self.image_shape[0], shape),
                np.broadcast_to(self.image_shape[1], shape),
            ),
            np.broadcast_to(self.optical_center.vector, shape + (2,)),
            np.broadcast_to(
                self.px2body_matrix,
                shape + (self.px2body_matrix.shape[-2], self.px2body_matrix.shape[-1]),
            ),
        )
