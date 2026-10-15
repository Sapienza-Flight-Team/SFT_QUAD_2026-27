from sft_math import GeoLoc, Maps
from sft_math.cam import Camera
from sft_math.cam.affine import Affine
from sft_math.geo import GeoGPS
from sft_math.geo.polar import GPS
from sft_math.cam.utils import get_rpy
from sft_math.cam.px2body_matrix import load_px2body_matrix_from_yaml
import numpy as np
from numpy import typing as np_typing
import pandas as pd
import os
from pathlib import Path
from typing import Tuple, List
import yaml


class BaseGenerator:

    # path di output
    data_path: str  # path dove salvare i dati
    metadata: pd.DataFrame  # metadata
    geoloc: GeoLoc  # geolocalizzazione per ogni scatto
    maps: Maps  # mappa
    camera: Camera  # fotocamera
    px_coordinates: np_typing.NDArray  # coordinate px della mappa
    time_step: float  # tempo tra un frame e l'altro in secondi

    def __init__(
        self,
        data_path: str,  # path dove salvare i dati
        #! specifiche strutturali
        maps_path: str,  # path con le indicazioni della mappa
        camera_path: str,  # path con le specifiche della fotocamera
        #! specifiche di simulazione e missione
        time_step: float,  # tempo tra un frame e l'altro in secondi
        lines_csv_path: str,  # file csv con le path
        #! dinamiche di volo
        ideal_speed: float,  # velocità ideale dell'UAV in m/s
        nominal_speed_noise: float,  # di quanto si sbaglia al massimo in m/s l'UAV
        nominal_trajectory_brownianNoise: float,  # di quanto si sbaglia l'UAV in m quando segue una traiettoria
        nominal_perspective_noise: float,  # errore massimo della gimbal in gradi
        nominal_height_brownianNoise: float,  # variazione massima di quota in un secondo di volo in metri
        nominal_yaw_brownianNoise: float,  # variazione massima di yaw in un secondo di volo in gradi
    ):

        lines_csv = pd.read_csv(lines_csv_path)
        assert len(lines_csv) > 0, "lines_csv_path is empty"

        lines: Tuple[GPS, GPS] = (
            GPS(
                lines_csv["start_lat"].to_numpy(),
                lines_csv["start_lon"].to_numpy(),
            ),
            GPS(
                lines_csv["end_lat"].to_numpy(),
                lines_csv["end_lon"].to_numpy(),
            ),
        )
        altitudes: List[float] = lines_csv["altitude"].to_numpy().tolist()
        maneuver_time = lines_csv["maneuver_time"].to_numpy().astype(int)

        # mi assicuro che esista la cartella di output
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"data_path {data_path} does not exist, please create it first"
            )

        # creazione della mappa
        with open(maps_path, "r") as f:
            info = yaml.safe_load(f)
            self.maps = Maps(
                GeoGPS(
                    GPS(
                        info["manifold_center"]["lat"],
                        info["manifold_center"]["lon"],
                    )
                ),  # center of the manifold
                (  # 3 reference points
                    GPS(
                        (
                            info["three_points"]["point1"]["lat"],
                            info["three_points"]["point2"]["lat"],
                            info["three_points"]["point3"]["lat"],
                        ),
                        (
                            info["three_points"]["point1"]["lon"],
                            info["three_points"]["point2"]["lon"],
                            info["three_points"]["point3"]["lon"],
                        ),
                    ),
                    (
                        (
                            info["three_points"]["point1"]["coox"],
                            info["three_points"]["point1"]["cooy"],
                        ),
                        (
                            info["three_points"]["point2"]["coox"],
                            info["three_points"]["point2"]["cooy"],
                        ),
                        (
                            info["three_points"]["point3"]["coox"],
                            info["three_points"]["point3"]["cooy"],
                        ),
                    ),
                ),
            )

        # creazione della fotocamera
        with open(camera_path, "r") as f:
            info = yaml.safe_load(f)
            px2body_matrix = load_px2body_matrix_from_yaml(
                Path(__file__).resolve().parents[3] / "sft_config.yaml"
            )
            self.camera = Camera(
                info["focal_length"],  # focal length
                info["sensor_size"],  # sensor size
                info["photo_shape"],  # photo shape
                info["optic_center"],  # px coordinates of the optic center
                px2body_matrix,  # pixel-to-body matrix
            )

        self.time_step = time_step

        # Brownian noise values
        def _simulation(
            start: GPS, end: GPS, altitude: float, initial_time: int
        ) -> Tuple[list[dict[str, float]], int]:
            start_vec = self.maps.manifold_center.map(start)
            end_vec = self.maps.manifold_center.map(end)
            length = float((end_vec - start_vec).norm())
            ideal_yaw = float(np.rad2deg((end_vec - start_vec).angle()))
            metadata: list[dict[str, float]] = []
            current_pos = (
                np.random.uniform(0.0, 1.0) * ideal_speed * self.time_step / 2
            )  # posizione del primo scatto
            current_error = np.zeros(
                2
            )  # errore Browniano della posizione (inizializzato a 0)
            current_height = altitude  # altezza ideale
            current_yaw = ideal_yaw  # yaw ideale
            current_time = float(initial_time)  # tempo iniziale
            while current_pos <= length:
                t = current_pos / length  # posizione lungo la linea teorica
                vec = (1 - t) * start_vec + t * end_vec  # vettore posizione nominale
                # aggiorno gli stati secondo moto browniano
                current_error += np.array(
                    [
                        np.random.uniform(-1, 1)
                        * nominal_trajectory_brownianNoise
                        * (self.time_step**0.5),
                        np.random.uniform(-1, 1)
                        * nominal_trajectory_brownianNoise
                        * (self.time_step**0.5),
                    ]
                )
                current_height += (
                    np.random.uniform(-1, 1)
                    * nominal_height_brownianNoise
                    * (self.time_step**0.5)
                )
                current_yaw += (
                    np.random.uniform(-1, 1)
                    * nominal_yaw_brownianNoise
                    * (self.time_step**0.5)
                )
                # aggiorno la posizione dell'UAV
                vec += GeoGPS.Vector(current_error[0], current_error[1])
                gps = self.maps.manifold_center.unmap(
                    vec
                )  # gps della posizione reale dell'uav
                metadata.append(
                    {
                        "lat": float(np.rad2deg(gps.lat)),
                        "lon": float(np.rad2deg(gps.lon)),
                        "h": float(current_height),
                        "psi": float(np.deg2rad(current_yaw)),
                        "teensy_timestamp": int(current_time),
                    }
                )
                # aggiorno il livello dello scatto in base alla velocità (che segue errore non browniano)
                current_pos += self.time_step * (
                    ideal_speed + np.random.uniform(-1, 1) * nominal_speed_noise
                )
                current_time += self.time_step * 1000
            return metadata, int(current_time)

        metadata_list: list[list[dict[str, float]]] = []
        current_time: int = 0
        for i in range(len(lines_csv)):
            metadata, current_time = _simulation(
                lines[0][(i,)], lines[1][(i,)], altitudes[i], current_time
            )
            metadata_list.append(metadata)
            current_time += maneuver_time[i]

        merged_metadata = sum(metadata_list[1:], metadata_list[0])

        # no-Brownian noise values
        for metadata_frame in merged_metadata:
            perspective_angle = float(
                np.random.uniform(0, 1) * nominal_perspective_noise
            )
            perspective_direction = float(
                np.random.uniform(-1, 1) * 180
            )  # angolo del versore che è asse di rotazione
            perspective_angle = np.deg2rad(perspective_angle)
            perspective_direction = np.deg2rad(perspective_direction)
            # calcolo il versore della rotazione
            v = np.array(
                [np.cos(perspective_direction), np.sin(perspective_direction), 0]
            )
            # inizializzo le matrici
            A = np.array(
                [
                    [v[1] ** 2, -v[0] * v[1], 0],
                    [-v[0] * v[1], v[0] ** 2, 0],
                    [0, 0, 1],
                ]
            )
            B = np.array(
                [
                    [0, 0, v[1]],
                    [0, 0, -v[0]],
                    [-v[1], v[0], 0],
                ]
            )
            C = np.einsum("r,c -> rc", v, v)
            M = C + np.cos(perspective_angle) * A + np.sin(perspective_angle) * B
            # roll, pitch e yaw della camera rispetto l'uav
            roll, pitch, _ = get_rpy(M)
            metadata_frame["phi"] = float(np.deg2rad(roll))
            metadata_frame["theta"] = float(np.deg2rad(pitch))

        # create the 'secret' metadata as fusion of the four metadata, it is a pandas dataframe
        self.metadata = pd.DataFrame(merged_metadata)

        # creo il geoloc per ogni foto
        ne_vec: GeoGPS.Vector = self.maps.manifold_center.map(
            GPS(self.metadata["lat"].to_numpy(), self.metadata["lon"].to_numpy())
        )  # type: ignore
        ned_vec = np.stack(
            [ne_vec.nord, ne_vec.east, -self.metadata["h"].to_numpy()], axis=-1  # type: ignore
        )
        config = Camera.Configuration(
            Affine(ned_vec, ((), (), ())),
            self.metadata["phi"].to_numpy(),
            self.metadata["theta"].to_numpy(),
            self.metadata["psi"].to_numpy(),
            is_deg=False,
        )

        geoloc = GeoLoc(self.camera, config, self.maps.manifold_center)
        self.geoloc = geoloc.broadcast_to((1,) + geoloc.shape)
        px_vertices = Affine(
            [
                [[0, 0]],
                [[0, self.camera.image_shape[1].item()]],
                [
                    [
                        self.camera.image_shape[0].item(),
                        self.camera.image_shape[1].item(),
                    ]
                ],
                [[self.camera.image_shape[0].item(), 0]],
            ],
            ((()), (())),
        )

        gps_vertices = self.geoloc.to_gps(
            px_vertices
        )  # trovo le coordinate gps dei vertici sul suolo
        self.px_coordinates = self.maps.map(
            gps_vertices.gps
        )  # traduco queste coordinate gps in px sulla mappa
        self.px_coordinates = np.moveaxis(self.px_coordinates, 0, 1)

        # save metadata
        self.data_path = data_path
        self.metadata.to_csv(
            os.path.join(self.data_path, "true_metadata.csv"),
            index=False,
            float_format="%.15f",
        )

    def map(self, x: np_typing.ArrayLike, y: np_typing.ArrayLike) -> np_typing.NDArray:
        """Returns the px coordinates of the map"""
        gps_coos = self.maps.unmap(np.stack([x, y], dtype=np.float64, axis=-1))
        px_coordinates = self.geoloc.from_gps(gps_coos)
        return px_coordinates.vector

    def copy(self) -> "BaseGenerator":
        new_object = object.__new__(type(self))
        new_object.__dict__.update(self.__dict__)
        return new_object

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, idx) -> "BaseGenerator":
        out = self.copy()
        out.maps = self.maps
        out.camera = self.camera
        out.data_path = self.data_path
        out.metadata = self.metadata.iloc[idx]
        out.px_coordinates = self.px_coordinates[idx]
        out.geoloc = self.geoloc[idx]
        return out


class SensorsGenerator(BaseGenerator):

    noisy_metadata: pd.DataFrame  # metadata con rumore

    def __init__(
        self,
        gps_noise: float,
        barometer_noise: float,
        compass_noise: float,
        gyroscope_noise: float,
        **kwarks,  # parametri per BaseGenerator
    ):
        super().__init__(**kwarks)

        self.noisy_metadata = self.metadata.copy()

        # make noise for gps
        pos_error = np.random.randn(len(self), 2)
        vec_error = GeoGPS.Vector(
            pos_error[:, 0] * gps_noise,
            pos_error[:, 1] * gps_noise,
        )
        # make noise for barometer
        barometer_error = np.random.randn(len(self)) * barometer_noise
        # make noise for compass
        compass_error = np.random.randn(len(self)) * compass_noise
        # make noise for gyroscope
        gyroscope_error = np.random.randn(len(self), 2) * gyroscope_noise

        # recupero tutti i gps
        gps = GeoGPS(
            GPS(
                self.metadata["lat"].to_numpy(),
                self.metadata["lon"].to_numpy(),
            )
        )
        gps_noisy = gps + vec_error.to_movement()
        self.noisy_metadata["lat"] = np.rad2deg(gps_noisy.lat)
        self.noisy_metadata["lon"] = np.rad2deg(gps_noisy.lon)

        # recupero tutti gli h
        h = self.metadata["h"].to_numpy()
        h_noisy = h + barometer_error
        self.noisy_metadata["h"] = h_noisy

        # recupero tutti gli psi
        psi = np.rad2deg(self.metadata["psi"].to_numpy())
        psi_noisy = psi + compass_error
        psi_noisy = (psi_noisy + 180) % 360 - 180
        self.noisy_metadata["psi"] = np.deg2rad(psi_noisy)

        # recupero tutti gli phi, theta
        phi = np.rad2deg(self.metadata["phi"].to_numpy())
        theta = np.rad2deg(self.metadata["theta"].to_numpy())
        phi_noisy = phi + gyroscope_error[:, 0]
        phi_noisy = (phi_noisy + 180) % 360 - 180
        theta_noisy = theta + gyroscope_error[:, 1]
        self.noisy_metadata["phi"] = np.deg2rad(phi_noisy)
        self.noisy_metadata["theta"] = np.deg2rad(theta_noisy)

        # save the noisy metadata
        self.noisy_metadata.to_csv(
            os.path.join(self.data_path, "noisy_metadata.csv"),
            index=False,
            float_format="%.15f",
        )

    def copy(self) -> "SensorsGenerator":
        new_object = object.__new__(type(self))
        new_object.__dict__.update(self.__dict__)
        return new_object

    def __getitem__(self, idx) -> "SensorsGenerator":
        out = self.copy()
        out.maps = self.maps
        out.camera = self.camera
        out.data_path = self.data_path
        out.metadata = self.metadata.iloc[idx]
        out.px_coordinates = self.px_coordinates[idx]
        out.geoloc = self.geoloc[idx]
        out.noisy_metadata = self.noisy_metadata.iloc[idx]
        return out
