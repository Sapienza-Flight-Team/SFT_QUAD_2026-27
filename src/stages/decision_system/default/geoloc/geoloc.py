import numpy as np
from numpy import typing as np_typing
from logging import Logger
from typing import Dict, Optional, Any, Tuple
from sft_math.cam.px2body_matrix import resolve_px2body_matrix
from stages.decision_system.default.class_templates import GeolocTemplate
from sft_math import GeoLoc
from sft_math.cam import Camera
from sft_math.geo import GeoGPS
from sft_math.geo.polar import GPS
from sft_math.cam.affine import Affine
from scipy.stats import chi2


class Geoloc(GeolocTemplate):

    logger: Logger
    cam: Camera
    steps: int
    pval: float
    rp_sigma2_error: float
    yaw_sigma2_error: float
    altitude_sigma2_error: float
    ne_sigma2_error: float
    gps: GeoGPS
    max_manifold_distance: float

    class Failure(Exception):
        pass

    def __init__(self, logger: Logger, configs: Dict[str, Optional[Any]]) -> None:
        self.logger = logger
        configs = dict(configs)
        configs["px2body_matrix"] = resolve_px2body_matrix(configs)

        # check if the configs are correct
        if "focal_length" not in configs:
            raise Geoloc.Failure("focal_length not in configs")
        if "pixel_size" not in configs:
            raise Geoloc.Failure("pixel_size not in configs")
        if "image_shape" not in configs:
            raise Geoloc.Failure("image_shape not in configs")
        if "px2body_matrix" not in configs:
            raise Geoloc.Failure("px2body_matrix not in configs")
        if "monte_carlo" not in configs:
            raise Geoloc.Failure("monte_carlo not in configs")
        if "p_value" not in configs:
            raise Geoloc.Failure("p_value not in configs")
        if "rp_error" not in configs:
            raise Geoloc.Failure("rp_error not in configs")
        if "rp_maxerror" not in configs:
            raise Geoloc.Failure("rp_maxerror not in configs")
        if "yaw_error" not in configs:
            raise Geoloc.Failure("yaw_error not in configs")
        if "yaw_maxerror" not in configs:
            raise Geoloc.Failure("yaw_maxerror not in configs")
        if "altitude_error" not in configs:
            raise Geoloc.Failure("altitude_error not in configs")
        if "altitude_maxerror" not in configs:
            raise Geoloc.Failure("altitude_maxerror not in configs")
        if "ne_error" not in configs:
            raise Geoloc.Failure("ne_error not in configs")
        if "ne_maxerror" not in configs:
            raise Geoloc.Failure("ne_maxerror not in configs")
        if "manifold_center" not in configs:
            raise Geoloc.Failure("manifold_center not in configs")
        if not type(configs["image_shape"]) is list:
            raise Geoloc.Failure(
                f"image_shape is not a list, but got {type(configs['image_shape'])}"
            )
        if not len(configs["image_shape"]) == 2:
            raise Geoloc.Failure(
                f"image_shape must have 2 elements, but got {len(configs['image_shape'])}"
            )
        if not type(configs["image_shape"][0]) is int:
            raise Geoloc.Failure(
                f"image_shape[0] is not an int, but got {type(configs['image_shape'][0])}"
            )
        if not type(configs["image_shape"][1]) is int:
            raise Geoloc.Failure(
                f"image_shape[1] is not an int, but got {type(configs['image_shape'][1])}"
            )
        if not type(configs["px2body_matrix"]) is list:
            raise Geoloc.Failure(
                f"px2body_matrix is not a list, but got {type(configs['px2body_matrix'])}"
            )
        if not len(configs["px2body_matrix"]) == 2:
            raise Geoloc.Failure(
                f"px2body matrix must have 2 elements, but got {len(configs['px2body_matrix'])}"
            )
        if not type(configs["px2body_matrix"][0]) is list:
            raise Geoloc.Failure(
                f"px2body_matrix[0] is not a list, but got {type(configs['px2body_matrix'][0])}"
            )
        if not type(configs["px2body_matrix"][1]) is list:
            raise Geoloc.Failure(
                f"px2body_matrix[1] is not a list, but got {type(configs['px2body_matrix'][1])}"
            )
        if not type(configs["px2body_matrix"][0][0]) is float:
            raise Geoloc.Failure(
                f"px2body_matrix[0][0] is not a float, but got {type(configs['px2body_matrix'][0][0])}"
            )
        if not type(configs["px2body_matrix"][0][1]) is float:
            raise Geoloc.Failure(
                f"px2body_matrix[0][1] is not a float, but got {type(configs['px2body_matrix'][0][1])}"
            )
        if not type(configs["px2body_matrix"][1][0]) is float:
            raise Geoloc.Failure(
                f"px2body_matrix[1][0] is not a float, but got {type(configs['px2body_matrix'][1][0])}"
            )
        if not type(configs["px2body_matrix"][1][1]) is float:
            raise Geoloc.Failure(
                f"px2body_matrix[1][1] is not a float, but got {type(configs['px2body_matrix'][1][1])}"
            )
        if not type(configs["focal_length"]) is float:
            raise Geoloc.Failure(
                f"focal_length is not a float, but got {type(configs['focal_length'])}"
            )
        if not type(configs["pixel_size"]) is float:
            raise Geoloc.Failure(
                f"pixel_size is not a float, but got {type(configs['pixel_size'])}"
            )
        if not type(configs["monte_carlo"]) is int:
            raise Geoloc.Failure(
                f"monte_carlo is not a int, but got {type(configs['monte_carlo'])}"
            )
        if not type(configs["p_value"]) is float:
            raise Geoloc.Failure(
                f"p_value is not a float, but got {type(configs['p_value'])}"
            )
        if not type(configs["rp_error"]) is float:
            raise Geoloc.Failure(
                f"rp_error is not a float, but got {type(configs['rp_error'])}"
            )
        if not type(configs["rp_maxerror"]) is float:
            raise Geoloc.Failure(
                f"rp_maxerror is not a float, but got {type(configs['rp_maxerror'])}"
            )
        if not type(configs["yaw_error"]) is float:
            raise Geoloc.Failure(
                f"yaw_error is not a float, but got {type(configs['yaw_error'])}"
            )
        if not type(configs["yaw_maxerror"]) is float:
            raise Geoloc.Failure(
                f"yaw_maxerror is not a float, but got {type(configs['yaw_maxerror'])}"
            )
        if not type(configs["altitude_error"]) is float:
            raise Geoloc.Failure(
                f"altitude_error is not a float, but got {type(configs['altitude_error'])}"
            )
        if not type(configs["altitude_maxerror"]) is float:
            raise Geoloc.Failure(
                f"altitude_maxerror is not a float, but got {type(configs['altitude_maxerror'])}"
            )
        if not type(configs["ne_error"]) is float:
            raise Geoloc.Failure(
                f"ne_error is not a float, but got {type(configs['ne_error'])}"
            )
        if not type(configs["ne_maxerror"]) is float:
            raise Geoloc.Failure(
                f"ne_maxerror is not a float, but got {type(configs['ne_maxerror'])}"
            )
        if not type(configs["manifold_center"]) is list:
            raise Geoloc.Failure(
                f"manifold_center is not a list, but got {type(configs['manifold_center'])}"
            )
        if not len(configs["manifold_center"]) == 2:
            raise Geoloc.Failure(
                f"manifold_center must have 2 elements, but got {len(configs['manifold_center'])}"
            )
        if not type(configs["manifold_center"][0]) is float:
            raise Geoloc.Failure(
                f"manifold_center[0] is not a float, but got {type(configs['manifold_center'][0])}"
            )
        if not type(configs["manifold_center"][1]) is float:
            raise Geoloc.Failure(
                f"manifold_center[1] is not a float, but got {type(configs['manifold_center'][1])}"
            )

        # inizializzo la fotocamera
        self.cam = Camera(
            configs["focal_length"],
            configs["pixel_size"],
            (configs["image_shape"][0], configs["image_shape"][1]),
            optical_center=[
                configs["image_shape"][0] / 2,
                configs["image_shape"][1] / 2,
            ],
            px2body_matrix=np.array(configs["px2body_matrix"]),
        )

        # gli errori sono visti come distribuzioni gaussiane centrate a varianza sferica
        # devo dedurre l'intensità della varianza usando il p_value e l'errore suggerito
        # osservo che la norma quadra di una normale multivariata standard ad n dimensioni è una chi-quadro ad n gradi di libertà
        # quindi la norma quadra di una normale multivariata ad n dimensioni con varianza sigma^2 sarà:
        #     chi2n * sigma^2
        # io non conosco sigma^2 ma so che (ad esempio con yaw):
        #     P[norm(yaw)^2 <= yaw_error] = p_value
        #     P[chi2n*sigma_yaw^2 <= yaw_error] = p_value
        #     P[chi2n <= yaw_error/sigma_yaw^2] = p_value
        #     yaw_error/sigma_yaw^2 = chi2n.ppf(p_value, n)
        #     sigma_yaw^2 = yaw_error/chi2n.ppf(p_value, n)
        self.steps = configs["monte_carlo"]
        self.pval = configs["p_value"]
        self.rp_sigma2_error = np.array(
            np.linalg.norm(configs["rp_error"]) ** 2 / chi2.ppf(self.pval / 100, 2)
        ).item()
        self.rp_maxerror = configs["rp_maxerror"]
        self.yaw_sigma2_error = np.array(
            np.linalg.norm(configs["yaw_error"]) ** 2 / chi2.ppf(self.pval / 100, 1)
        ).item()
        self.yaw_maxerror = configs["yaw_maxerror"]
        self.altitude_sigma2_error = np.array(
            np.linalg.norm(configs["altitude_error"]) ** 2
            / chi2.ppf(self.pval / 100, 1)
        ).item()
        self.altitude_maxerror = configs["altitude_maxerror"]
        self.ne_sigma2_error = np.array(
            np.linalg.norm(configs["ne_error"]) ** 2 / chi2.ppf(self.pval / 100, 2)
        ).item()
        self.ne_maxerror = configs["ne_maxerror"]

        # leggo il gps del manifold
        self.gps = GeoGPS(
            GPS(
                configs["manifold_center"][0],
                configs["manifold_center"][1],
            )
        )
        max_manifold_distance = configs.get("max_manifold_distance")
        self.max_manifold_distance = (
            50000.0 if max_manifold_distance is None else float(max_manifold_distance)
        )

    def __call__(
        self,
        pixels_coordinates: np_typing.ArrayLike,
        gps: GPS,
        altitude: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> Tuple[np_typing.NDArray[np.float64], np_typing.NDArray[np.float64], GeoGPS]:

        self.logger.info(
            f"Starting geolocalization using Monte Carlo method of photo with gps {np.rad2deg(gps.lat).tolist(), np.rad2deg(gps.lon).tolist()}..."
        )

        np_pxcoos = np.array(pixels_coordinates)
        assert (
            np_pxcoos.ndim == 2
        ), f"pixels_coordinates should be 2D, but got {np_pxcoos.ndim}D"
        assert (
            np_pxcoos.shape[1] == 2
        ), f"pixels_coordinates should have 2 columns, but got {np_pxcoos.shape[1]} columns"

        # calcolo i valori nominali
        ne_vec: GeoGPS.Vector = self.gps.map(gps)  # type: ignore
        nord = np.array(ne_vec.nord, dtype=np.float64).item()
        east = np.array(ne_vec.east, dtype=np.float64).item()
        distance_from_center = float((nord**2 + east**2) ** 0.5)
        if distance_from_center > self.max_manifold_distance:
            self.logger.warning(
                "Photo GPS is %.1f m from geoloc manifold_center; "
                "local N/E geolocation will be unreliable. Update "
                "decision_system.geoloc.manifold_center for this mission.",
                distance_from_center,
            )

        # aggiungo il noise
        nord_errors_noclipped = np.random.normal(
            0, self.ne_sigma2_error**0.5, self.steps
        )
        east_errors_noclipped = np.random.normal(
            0, self.ne_sigma2_error**0.5, self.steps
        )
        ne_errors_noclipped = np.concatenate(
            (nord_errors_noclipped[:, None], east_errors_noclipped[:, None]), axis=1
        )
        # 2D clip
        ne_errors = np.where(
            np.broadcast_to(
                np.linalg.norm(ne_errors_noclipped, axis=1, keepdims=True)
                > self.ne_maxerror,
                ne_errors_noclipped.shape,
            ),
            ne_errors_noclipped
            / np.linalg.norm(ne_errors_noclipped, axis=1)[:, None]
            * self.ne_maxerror,
            ne_errors_noclipped,
        )
        altitude_errors = np.clip(
            np.random.normal(0, self.altitude_sigma2_error**0.5, self.steps),
            -self.altitude_maxerror,
            self.altitude_maxerror,
        )
        roll_errors_noclipped = np.random.normal(
            0, self.rp_sigma2_error**0.5, self.steps
        )
        pitch_errors_noclipped = np.random.normal(
            0, self.rp_sigma2_error**0.5, self.steps
        )
        rp_errors_noclipped = np.concatenate(
            (roll_errors_noclipped[:, None], pitch_errors_noclipped[:, None]), axis=1
        )
        # 2D clip
        rp_errors = np.where(
            np.broadcast_to(
                np.linalg.norm(rp_errors_noclipped, axis=1, keepdims=True)
                > self.rp_maxerror,
                rp_errors_noclipped.shape,
            ),
            rp_errors_noclipped
            / np.linalg.norm(rp_errors_noclipped, axis=1)[:, None]
            * self.rp_maxerror,
            rp_errors_noclipped,
        )
        yaw_errors = np.clip(
            np.random.normal(0, self.yaw_sigma2_error**0.5, self.steps),
            -self.yaw_maxerror,
            self.yaw_maxerror,
        )

        self.logger.debug(
            f"nominal values: {nord=}, {east=}, {altitude=}, {roll=}°, {pitch=}°, {yaw=}°"
        )

        res_nord_east = np.array([nord, east]) + ne_errors
        res_altitude = altitude + altitude_errors
        res_roll_pitch = np.array([roll, pitch]) + rp_errors
        res_yaw = yaw + yaw_errors

        position = Affine(
            np.array([res_nord_east[:, 0], res_nord_east[:, 1], -res_altitude]).T,
            (
                (),
                (),
                (),
            ),
        )
        configuration = Camera.Configuration(
            position,
            res_roll_pitch[:, 0],
            res_roll_pitch[:, 1],
            res_yaw,
            is_deg=True,
        )
        geoloc = GeoLoc(self.cam, configuration, self.gps)
        geoloc = geoloc.broadcast_to((1, self.steps))

        points_vec: GeoGPS.Vector = self.gps.map(
            geoloc.to_gps(
                Affine(
                    np_pxcoos[:, None, :],
                    (
                        (
                            (),
                            (),
                        )
                    ),
                )
            ).gps
        )  # type: ignore
        points = np.array([points_vec.nord, points_vec.east]).T

        # calcolo la media e la matrice di covarianza
        mean: np_typing.NDArray[np.float64] = np.mean(points, axis=0, dtype=np.float64)
        centralized_points = points - mean
        covariance: np_typing.NDArray[np.float64] = np.einsum(
            "sci,scj->cij", centralized_points, centralized_points, dtype=np.float64
        ) / (self.steps - 1)

        self.logger.debug(f"Mean: {mean.tolist()}")

        self.logger.info("Geolocalization completed successfully.")

        return mean, covariance, self.gps

    def dump_data(self, tmp_path: str):
        raise NotImplementedError
