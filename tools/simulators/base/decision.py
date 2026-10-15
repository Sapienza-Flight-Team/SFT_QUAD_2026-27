from .generator import SensorsGenerator
from numpy import typing as np_typing
import numpy as np
import yaml
from sft_math.geo import GeoGPS
from sft_math.geo.polar import GPS
import pandas as pd
import os
import cv2
from tqdm import tqdm
from typing import Any, Dict


class DecisionGenerator:

    LUpx: np.ndarray
    LDpx: np.ndarray
    RDpx: np.ndarray
    RUpx: np.ndarray

    def __init__(
        self,
        image_path: str,
        generator: SensorsGenerator,
        n_classes: int,
        good_targets: int,
        intra_distance: float,
        confusion_matrix: np_typing.NDArray[np.float64],
        fake_targets: int,
        p_fake_targets: float,
        noise: float,
        boundaries: str,
    ):
        # recupero i bordi dell'area di AirDrop
        with open(boundaries, "r") as f:
            area_dict = yaml.safe_load(f)
            gps_points = GeoGPS(
                GPS(
                    [
                        area_dict["drop_area"]["LU"]["lat"],
                        area_dict["drop_area"]["LD"]["lat"],
                        area_dict["drop_area"]["RD"]["lat"],
                        area_dict["drop_area"]["RU"]["lat"],
                    ],
                    [
                        area_dict["drop_area"]["LU"]["lon"],
                        area_dict["drop_area"]["LD"]["lon"],
                        area_dict["drop_area"]["RD"]["lon"],
                        area_dict["drop_area"]["RU"]["lon"],
                    ],
                ),
            )
        gps_center = generator.maps.manifold_center
        vec_LU = gps_center.map(gps_points[..., 0].gps)
        vec_LD = gps_center.map(gps_points[..., 1].gps)
        vec_RD = gps_center.map(gps_points[..., 2].gps)
        vec_RU = gps_center.map(gps_points[..., 3].gps)
        L_vec = vec_LD - vec_LU
        R_vec = vec_RD - vec_RU
        U_vec = vec_RU - vec_LU
        D_vec = vec_RD - vec_LD
        #! i target sono inseriti in un'area che permette
        #! di includere tutta la loro area accettabile nell'area di drop
        LR = (L_vec.norm() + R_vec.norm()) / 2
        UD = (U_vec.norm() + D_vec.norm()) / 2
        assert (
            LR > intra_distance and UD > intra_distance
        ), f"intra_distance must be more little beacause {LR=}, {UD=}."

        # con questi dati posso generare i target
        good_targets_points = np.random.rand(good_targets, 2) * np.array(
            [LR - intra_distance, UD - intra_distance]
        ).reshape(1, 2)
        self_distances = np.linalg.norm(
            good_targets_points[:, None, :] - good_targets_points[None, :, :], axis=-1
        )
        self_distances[np.arange(good_targets), np.arange(good_targets)] = np.inf
        while np.min(self_distances) < intra_distance:
            good_targets_points = np.random.rand(good_targets, 2) * np.array(
                [LR - intra_distance, UD - intra_distance]
            ).reshape(1, 2)
            self_distances = np.linalg.norm(
                good_targets_points[:, None, :] - good_targets_points[None, :, :],
                axis=-1,
            )
            self_distances[np.arange(good_targets), np.arange(good_targets)] = np.inf
        fake_targets_points = np.random.rand(fake_targets, 2) * np.array(
            [LR, UD]
        ).reshape(1, 2)
        # calcolo le coordinate gps dei target good/fake
        good_targets_points += np.array([intra_distance / 2, intra_distance / 2])
        good_targets_gps = gps_center.unmap(
            vec_LU
            + L_vec * (good_targets_points[:, 0] / LR)
            + U_vec * (good_targets_points[:, 1] / UD)
        )
        fake_targets_gps = gps_center.unmap(
            vec_LU
            + L_vec * (fake_targets_points[:, 0] / LR)
            + U_vec * (fake_targets_points[:, 1] / UD)
        )

        # inizializzo le classi
        good_classes = np.random.choice(
            np.arange(n_classes), good_targets, replace=False
        )
        fake_classes = np.random.choice(
            np.arange(n_classes), fake_targets, replace=True
        )

        # calcolo le coordinate px dei target nella foto di base
        good_targets_bpx = generator.maps.map(good_targets_gps)
        good_targets_px = generator.map(
            good_targets_bpx[:, [0]], good_targets_bpx[:, [1]]
        )
        if fake_targets > 0:
            fake_targets_bpx = generator.maps.map(fake_targets_gps)
            fake_targets_px = generator.map(
                fake_targets_bpx[:, [0]], fake_targets_bpx[:, [1]]
            )
        else:
            fake_targets_bpx = np.array

        # considero solo i target interamente contenuti nell'immagine
        w, h = (
            int(generator.camera.image_shape[0].item()),
            int(generator.camera.image_shape[1].item()),
        )
        mask_good = (
            (good_targets_px[..., 0] >= 0)
            & (good_targets_px[..., 0] < w)
            & (good_targets_px[..., 1] >= 0)
            & (good_targets_px[..., 1] < h)
        )
        mask_fake = (
            (fake_targets_px[..., 0] >= 0)
            & (fake_targets_px[..., 0] < w)
            & (fake_targets_px[..., 1] >= 0)
            & (fake_targets_px[..., 1] < h)
        )
        # ogni componente comunica se un dato target compare in una data foto

        metadata: list[dict[str, Any]] = []
        bar = tqdm(range(len(generator)), "DecisionGenerator", leave=True)
        # per ogni colonna di generator.noisy_metadata mi segno il tipo
        all_frames: list[Dict[str, Any]] = []
        for i in range(len(generator.noisy_metadata)):
            all_frames.append(
                {
                    col: generator.noisy_metadata[col][i]
                    for col in generator.noisy_metadata.columns
                }
            )
        for i in bar:
            for j in range(len(good_targets_px)):
                if mask_good[j, i]:
                    frame = all_frames[i].copy()
                    frame["photo"] = i
                    frame["true_class"] = good_classes[j]
                    frame["x"] = int(good_targets_px[j, i, 0])
                    frame["y"] = int(good_targets_px[j, i, 1])
                    frame["target"] = "good"
                    frame["classification"] = np.random.choice(
                        np.arange(n_classes), p=confusion_matrix[good_classes[j]]
                    )
                    metadata.append(frame)
            for j in range(len(fake_targets_px)):
                if mask_fake[j, i]:
                    if np.random.rand() < p_fake_targets:
                        frame = all_frames[i].copy()
                        frame["photo"] = i
                        frame["true_class"] = fake_classes[j]
                        frame["x"] = int(fake_targets_px[j, i, 0])
                        frame["y"] = int(fake_targets_px[j, i, 1])
                        frame["target"] = "fake"
                        frame["classification"] = np.random.choice(
                            np.arange(n_classes), p=confusion_matrix[fake_classes[j]]
                        )
                        metadata.append(frame)
            # aggiungo il rumore
            indices = np.argwhere(np.random.rand(w, h) < noise**2)
            for j in range(len(indices)):
                frame = all_frames[i].copy()
                frame["photo"] = i
                frame["true_class"] = -1
                frame["x"] = indices[j][0]
                frame["y"] = indices[j][1]
                frame["target"] = "noise"
                frame["classification"] = np.random.randint(0, n_classes)
                metadata.append(frame)

        self.metadata = pd.DataFrame(metadata)
        self.metadata.to_csv(
            os.path.join(generator.data_path, "decision_metadata.csv"), index=False
        )

        # acquisizione delle foto come png
        LUpx = generator.maps.map(gps_points[..., 0].gps).astype(int)
        LDpx = generator.maps.map(gps_points[..., 1].gps).astype(int)
        RDpx = generator.maps.map(gps_points[..., 2].gps).astype(int)
        RUpx = generator.maps.map(gps_points[..., 3].gps).astype(int)

        # prima mostro una foto della mappa con i target buoni (in verde) e fake (in rosso)
        # con rispettiva classificazione base
        photo = cv2.imread(image_path)
        # aggiungo un rettangolo blu fine che indica il bordo della mappa
        photo = cv2.line(
            photo, tuple(LUpx[0].tolist()), tuple(LDpx[0].tolist()), (255, 0, 0), 10
        )
        photo = cv2.line(
            photo, tuple(RUpx[0].tolist()), tuple(RDpx[0].tolist()), (255, 0, 0), 10
        )
        photo = cv2.line(
            photo, tuple(LUpx[0].tolist()), tuple(RUpx[0].tolist()), (255, 0, 0), 10
        )
        photo = cv2.line(
            photo, tuple(LDpx[0].tolist()), tuple(RDpx[0].tolist()), (255, 0, 0), 10
        )
        # inserisco i good_targets
        good_targets_px = generator.maps.map(good_targets_gps).astype(int)
        for i in range(good_targets):
            photo = cv2.circle(
                photo,
                tuple(good_targets_px[i, :].tolist()),
                20,
                (0, 255, 0),
                -1,
            )
            # scrivo il target
            cv2.putText(
                photo,
                str(good_classes[i]),
                tuple((good_targets_px[i, :] + 30).tolist()),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        fake_targets_px = generator.maps.map(fake_targets_gps).astype(int)
        for i in range(fake_targets):
            photo = cv2.circle(
                photo,
                tuple(fake_targets_px[i, :].tolist()),
                20,
                (0, 0, 255),
                -1,
            )
            # scrivo il target a fianco del cerchio
            cv2.putText(
                photo,
                str(fake_classes[i]),
                tuple((fake_targets_px[i, :] + 30).tolist()),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )
        # save image
        cv2.imwrite(f"{generator.data_path}/map.png", photo)

        # get an example of a photo with at least one good target
        # read self.metadata and find all photos with good targets
        indices = self.metadata[self.metadata["target"] == "good"]["photo"].unique()
        # get a random photo
        index = np.random.choice(indices)
        # rendering the photo
        photo = cv2.imread(image_path)

        four_vertices_px = generator.px_coordinates[:, index, :]
        w, h = (
            generator.camera.image_shape[0].item(),
            generator.camera.image_shape[1].item(),
        )
        pts1 = np.array(four_vertices_px, dtype=np.float32)
        pts2 = np.array(
            [[0.0, 0.0], [0.0, h], [w, h], [w, 0.0]],
            dtype=np.float32,
        )
        M = cv2.getPerspectiveTransform(pts1, pts2)
        photo = cv2.warpPerspective(photo, M, (5328, 3040))

        # segno tutti i dati letti per la foto index-esima leggendo il metadata
        photo_metadata = self.metadata[self.metadata["photo"] == index]
        for i in range(len(photo_metadata)):
            if photo_metadata.iloc[i]["target"] == "good":
                color = (0, 255, 0)
            elif photo_metadata.iloc[i]["target"] == "fake":
                color = (0, 0, 255)
            else:
                color = (128, 128, 128)
            photo = cv2.circle(
                photo,
                tuple(photo_metadata.iloc[i][["x", "y"]].astype(int).tolist()),
                20,
                color,
                -1,
            )
            # scrivo il target a fianco del cerchio
            cv2.putText(
                photo,
                str(photo_metadata.iloc[i]["true_class"]),
                tuple((photo_metadata.iloc[i][["x", "y"]] + 30).astype(int).tolist()),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )

        cv2.imwrite(f"{generator.data_path}/inference_example.png", photo)
