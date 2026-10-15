import cv2
import os
import csv
import numpy as np
from pathlib import Path
from sft_math import GeoLoc
from typing import Optional

# from sft_cv.common.config import SFTConfig
from sft_math.cam import Camera
from sft_math.cam.affine import Affine, Point
from sft_math.geo import GeoGPS
from sft_math.geo.polar import GPS
from common.config import SFTConfig
from sft_math.cam.px2body_matrix import load_px2body_matrix_from_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_YAML_PATH = REPO_ROOT / "sft_config.yaml"


class errorEstimator:
    """
    A class that estimates the error in the GPS coordinates of the selected point across multiple images.
    """

    def __init__(self, folder_path: str, csv_path: str, obj_coords: GPS):
        """
        Initializes the folderScanner class with the given folder path and csv path.

        Parameters:
        -----------
        folder_path : str
            Path to the folder containing the images.
        csv_path : str
            Path to the csv file containing the metadata.
        """
        self.folder_path = folder_path
        self.csv_path = csv_path
        self.images = os.listdir(folder_path)
        self.allGPS = []
        self.obj_coords = obj_coords

    def scan_folder(self) -> None:
        """
        Scans the folder for images and creates checkCoordinates objects for each image.
        """
        for image in self.images:
            if image.endswith(".png"):
                image_path = os.path.join(self.folder_path, image)
                check = checkCoordinates(image_path, self.csv_path, 1080, 1920)
                check.open_image()
                self.allGPS.append((check.GPScoordinates, check.heading, check.speed))

    def estimate_error(self):
        """
        Estimate error
        """

        delays = []
        distance_error = []

        for gps, heading, speed in self.allGPS:
            movement = gps - self.obj_coords
            vector = movement.to_vector()
            ned_error = np.array([vector.nord.item(), vector.east.item()])
            hor_heading = np.array([np.cos(heading), np.sin(heading), 0])
            down = np.array([0, 0, 1])
            dx_heading = -np.cross(hor_heading, down)
            # Matrix converting heading coordinates to NED coordinates (3x3)
            heading2ned = np.stack([hor_heading, dx_heading, down], axis=-1)
            # Its inverse is simply its transpose since it is orthonormal.
            ned2heading = heading2ned.T

            # Convert ned error (extended to 3D with z=0) into heading coordinates.
            ned_error_3d = np.append(ned_error, 0)
            heading_error = ned2heading.dot(ned_error_3d)
            # The first component represents the error along the heading direction.
            distance_error.append(heading_error[0])
            delay_sec = heading_error[0] / speed
            delays.append(delay_sec)

        delays = np.array(delays)
        print("Delay meters per photo: ", distance_error)
        print("Mean delay meters per photo:", np.mean(distance_error))
        print("Delay seconds per photo:", delays)
        print("Mean delay seconds per photo:", np.mean(delays))


class checkCoordinates:
    """
    A class to check coordinates on an image using OpenCV.

    Attributes:
    -----------
    image_path : str
        Path to the image file.
    window_h : int
        Height of the display window.
    window_w : int
        Width of the display window.
    """

    def __init__(self, image_path: str, csv_path: str, window_h: int, window_w: int):
        """
        Initializes the checkCoordinates class with the given image path and window dimensions.

        Parameters:
        -----------
        image_path : str
            Path to the image file.
        window_h : int
            Height of the display window.
        window_w : int
            Width of the display window.
        """
        self.image_path = image_path
        self.csv_path = csv_path
        self.image = cv2.imread(image_path)
        self.h = self.image.shape[0]
        self.w = self.image.shape[1]
        self.window_h = window_h
        self.window_w = window_w
        self.x_offset = 0
        self.y_offset = 0
        self.clicks = []
        self.configs = SFTConfig("/home/sft/sft_cv_2024_2025/sft_config.yaml")
        self.GPScoordinates: Optional[GeoGPS] = None

    def _get_coordinates(self, event, x, y, flags, param):
        """
        Callback function to get coordinates of the selected point with respect to the center of the image.

        Parameters:
        -----------
        event : int
            Type of the mouse event.
        x : int
            X-coordinate of the mouse event.
        y : int
            Y-coordinate of the mouse event.
        flags : int
            Any relevant flags passed by OpenCV.
        param : any
            Any extra parameters supplied by OpenCV.
        """
        if event == cv2.EVENT_RBUTTONDOWN:
            x = x + self.x_offset
            y = y + self.y_offset
            print(
                f"Image coordinates of point {len(self.clicks) + 1} before conversion: {x}, {y}"
            )
            x_center = self.w // 2
            y_center = self.h // 2
            x_shifted = -(y - y_center)
            y_shifted = x - x_center

            self.clicks.append((x_shifted, y_shifted))

            print(
                f"Image coordinates of point {len(self.clicks)}: {x_shifted}, {y_shifted}"
            )
            cv2.circle(
                self.image,
                (x, y),
                5,
                (0, 0, 255),
                -1,
            )

            coords = np.array([[x_shifted, y_shifted]], dtype=np.float64)
            metadata = self.get_metadata()

            px2body_matrix = load_px2body_matrix_from_yaml(CONFIG_YAML_PATH)
            camera = Camera(
                16,  # distanza focale in millimetri
                2.74,  # grandezza del sensore in micrometri
                (5320, 3032),  # shape della foto
                (5320 / 2, 3032 / 2),  # coordinate pixel del centro ottico
                px2body_matrix,
            )

            affine: Point = Affine(coords, ((), ()))

            physic_conf = camera.Configuration(
                Affine(((0), (0), (-metadata[0, 2])), ((), (), ())),
                metadata[0, 3],
                metadata[0, 4],
                metadata[0, 5],
            )

            geoGPS = GeoGPS(GPS(metadata[0, 0], metadata[0, 1]))

            geoloc = GeoLoc(camera, physic_conf, geoGPS)

            coordinates = geoloc.to_gps(affine)
            self.GPScoordinates = coordinates

            print(f"GPS coordinates of point {len(self.clicks)}: {repr(coordinates)}")

            img = self.image[
                self.y_offset : self.y_offset + self.window_h,
                self.x_offset : self.x_offset + self.window_w,
            ]

            cv2.imshow(f"{self.image_path}", img)

    def open_image(self):
        """
        Opens the image in an OpenCV window and allows the user to navigate and select points on the image.

        Only allows one click per image

        The user can navigate the image using arrow keys and select points by right-clicking.
        """
        cv2.namedWindow(f"{self.image_path}", cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(f"{self.image_path}", self.window_w, self.window_h)
        cv2.setMouseCallback(f"{self.image_path}", self._get_coordinates)

        while True:
            img = self.image[
                self.y_offset : self.y_offset + self.window_h,
                self.x_offset : self.x_offset + self.window_w,
            ]
            cv2.imshow(f"{self.image_path}", img)

            key = cv2.waitKey(1000)

            if len(self.clicks) > 0:
                break

            if key == 27 or key == ord("q"):  # ESC or q key to exit
                break
            elif key == 81:  # Left arrow key
                self.x_offset = max(self.x_offset - 50, 0)
            elif key == 82:  # Up arrow key
                self.y_offset = max(self.y_offset - 50, 0)
            elif key == 83:  # Right arrow key
                self.x_offset = min(self.x_offset + 50, self.w - self.window_w)
            elif key == 84:  # Down arrow key
                self.y_offset = min(self.y_offset + 50, self.h - self.window_h)

        cv2.destroyAllWindows()

    def get_metadata(self) -> np.ndarray:
        image_name = os.path.basename(self.image_path)
        image_name = image_name.replace(".png", ".npy")
        csvfile = csv.DictReader(open(self.csv_path))
        metadata = None
        for row in csvfile:
            if image_name in row["filename"]:
                metadata = row
                break
        if metadata is None:
            raise ValueError("Metadata not found for image")

        self.heading = np.deg2rad(float(metadata["heading"]))
        self.speed = float(metadata["vmod"])

        return np.array(
            [
                [
                    float(metadata["lat"]),
                    float(metadata["lon"]),
                    float(metadata["h"]),
                    float(metadata["phi"]),
                    float(metadata["theta"]),
                    float(metadata["psi"]),
                    float(metadata["heading"]),
                    float(metadata["vmod"]),
                ]
            ],
            dtype=np.float64,
        )


if __name__ == "__main__":
    obj_coords = GPS(41.83189, 12.33337)
    csv_path = "/home/sft/pdv73_2/metadata_20250307-171114.csv"
    folder_path = "/home/sft/pdv73_2/png"
    scan = errorEstimator(folder_path, csv_path, obj_coords)
    scan.scan_folder()
    scan.estimate_error()

    print("Done")
