from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import cv2
from sft_math.cam.cam import Camera
from sft_math.cam.affine import Affine
from sft_math.geoloc import GeoLoc
from sft_math.geo import GeoGPS
from sft_math.geo.polar import GPS
from sft_math.cam.px2body_matrix import load_px2body_matrix_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML_PATH = REPO_ROOT / "sft_config.yaml"


class Target:
    def __init__(self, image_path, side, center_gps_coords):
        self.image_path = image_path
        self.name = image_path.split("/")[-1].split(".")[0].split("_")[-1]
        self.side = side
        self.center_gps_coords = center_gps_coords
        px2body_matrix = load_px2body_matrix_from_yaml(CONFIG_YAML_PATH)
        self.camera = Camera(
            16,  # distanza focale in millimetri
            2.74,  # grandezza del sensore in micrometri
            (5320, 3032),  # shape della foto
            (5320 / 2, 3032 / 2),  # coordinate pixel del centro ottico
            px2body_matrix,
        )

        self.image = cv2.imread(self.image_path)
        if self.image is None:
            print(f"Error loading image {self.image_path}")
        self.vertexes = {}
        self.calculate_vertexes_gps()

    def load_image(self) -> None:
        try:
            self.image = cv2.imread(self.image_path)
        except Exception as e:
            print(f"Error loading image {self.image_path}: {e}")

    def calculate_vertexes_gps(self) -> None:
        # trovo la posizione del centro del target in coordinate GPS
        center_gps = GeoGPS(GPS(self.center_gps_coords[0], self.center_gps_coords[1]))
        # calcolo la posizione dei 4 vertici del target in coordinate GPS spostandomi di self.size metri in ogni direzione
        # per fare questo, uso la classe Movement del modulo GeoGPS
        movement = GeoGPS.Movement(
            yaw=[
                45,
                135,
                225,
                315,
            ],  # spostamento in 4 direzioni (nord, est, sud, ovest)
            distance=[((self.side / 2) ** 2 + (self.side / 2) ** 2) ** 0.5]
            * 4,  # spostamento di self.size/2 metri
        )
        vertexes_gps = center_gps + movement
        self.vertexes["gps"] = vertexes_gps

    def get_homography(self, metadata) -> np.ndarray | None:
        # do in input le coordinate GPS dei vertici di un immagine e calcolo le coordinate pixel corrispondenti usando la classe GeoLoc
        # per fare questo, uso la classe GeoLoc del modulo geoloc, che prende in input una camera, una configurazione e un GPS
        # la configurazione è data dalla posizione della camera (che è data dalle coordinate GPS del centro del target) e dall'orientamento della camera (che è dato dagli angoli di yaw, pitch e roll)
        configuration = self.camera.Configuration(
            Affine(((0), (0), (-metadata["altitude"])), ((), (), ())),
            metadata["yaw"],
            metadata["pitch"],
            metadata["roll"],
        )

        geoloc = GeoLoc(
            self.camera,
            configuration,
            GeoGPS(GPS(metadata["latitude"], metadata["longitude"])),
        )
        vertexes_px = geoloc.from_gps(self.vertexes["gps"])

        # ora calcolo: se almeno uno dei vertexes_px è dentro l'immagine, allora calcolo le coordinate pixel dei vertici e le salvo in self.vertexes["px"]
        is_in = False
        # rendo vertexes_px un array numpy
        vertexes_px = np.array(vertexes_px)

        for vertex in vertexes_px:
            if 0 <= vertex[0] < 5320 and 0 <= vertex[1] < 3032:
                is_in = True
                break
        if is_in:
            vertexes = np.array(
                [
                    [0, 0],
                    [self.image.shape[1], 0],
                    [self.image.shape[1], self.image.shape[0]],
                    [0, self.image.shape[0]],
                ],
                dtype=np.float32,
            )
            homography, _ = cv2.findHomography(vertexes, vertexes_px)
            return homography
        print(f"Target {self.name} is not visible in the image {self.image_path}")
        return None


def load_metadata(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


# creo un target di test con centro in (lat, lon) e lato 10 metri
target = Target(
    image_path="src/sft_cv/target_simulator/assets/target_bullseye.png",
    side=5.0,
    center_gps_coords=(18.1, 42.1),
)


# todo teorema di pitagora
def add_targets_to_images(metadata_file):
    metadata = load_metadata(metadata_file)

    for image_data in metadata:

        homography = target.get_homography(image_data)
        if homography is not None:
            # sovrappongo l'immagine del target sull'immagine originale usando la matrice di omografia
            warped_target = cv2.warpPerspective(target.image, homography, (5320, 3032))
            original_image = cv2.imread(image_data["image_path"])
            if original_image is not None:
                blended_image = cv2.addWeighted(
                    original_image, 0.7, warped_target, 0.3, 0
                )
                cv2.imwrite(
                    f"output/{image_data['image_path'].split('/')[-1]}", blended_image
                )
            else:
                print(f"Error loading original image {image_data['image_path']}")
        else:
            print(
                f"Target {target.name} is not visible in the image {image_data['image_path']}"
            )
