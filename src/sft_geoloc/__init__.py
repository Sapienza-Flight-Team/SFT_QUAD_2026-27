"""Camera/GPS geometry using OpenCV, SciPy, and pyproj. See README for examples."""

from .camera import CameraModel


def __getattr__(name: str):
    # Camera-only tools work with SciPy versions predating RigidTransform.
    # Load GPS/uncertainty dependencies only when their API is requested.
    if name == "Geolocator":
        from .geolocator import Geolocator

        globals()[name] = Geolocator
        return Geolocator
    if name in {"GeolocationEstimate", "UncertaintyModel"}:
        from .old_uncertainty import GeolocationEstimate, UncertaintyModel

        globals().update(
            GeolocationEstimate=GeolocationEstimate, UncertaintyModel=UncertaintyModel
        )
        return globals()[name]
    if name in {"Data", "CentroidAnalyzer"}:
        from .old_clustering import Data, CentroidAnalyzer

        globals().update(Data=Data, CentroidAnalyzer=CentroidAnalyzer)
        return globals()[name]
    if name in {"CameraParams", "Transect"}:
        from .old_survey import CameraParams, Transect

        globals().update(CameraParams=CameraParams, Transect=Transect)
        return globals()[name]
    if name == "LocalMap":
        from .geodesy import LocalMap

        globals()[name] = LocalMap
        return LocalMap
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CameraModel",
    "CameraParams",
    "CentroidAnalyzer",
    "Data",
    "LocalMap",
    "Transect",
    "GeolocationEstimate",
    "Geolocator",
    "UncertaintyModel",
]
