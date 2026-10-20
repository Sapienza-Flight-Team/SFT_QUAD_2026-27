# sft_geoloc: complete usage guide

`sft_geoloc` converts camera pixels into ground positions, estimates their
uncertainty, groups repeated detections, and provides survey and heading helpers.
Its `.py` files are library modules: **import them from your application** rather
than running them individually. For runnable programs, use the
[geolocation tools guide](../../tools/README.md).

## Contents

- [Setup and first projection](#setup-and-first-projection)
- [Which file to use](#which-file-to-use)
- [Units, axes, shapes, and validity](#units-axes-shapes-and-validity)
- [Camera calibration and rays: camera.py](#camera-calibration-and-rays-camerapy)
- [Pixel/GPS projection: geolocator.py](#pixelgps-projection-geolocatorpy)
- [Maps and Earth coordinates: geodesy.py](#maps-and-earth-coordinates-geodesypy)
- [Ray intersections: ground.py](#ray-intersections-groundpy)
- [Uncertainty: uncertainty.py](#uncertainty-uncertaintypy)
- [Clustering: clustering.py](#clustering-clusteringpy)
- [Survey planning: survey.py](#survey-planning-surveypy)
- [Heading conversion: heading.py](#heading-conversion-headingpy)
- [Mounting configuration: px2body_matrix.py](#mounting-configuration-px2body_matrixpy)
- [Validation: check.py](#validation-checkpy)
- [Package exports: __init__.py](#package-exports-__init__py)
- [Pipeline integration, troubleshooting, and verification](#pipeline-integration-troubleshooting-and-verification)

## Setup and first projection

Use the repository's Python **3.11–3.13** environment and SciPy **1.17.x**,
with dependencies from `pyproject.toml`. NumPy, OpenCV, SciPy, pyproj, PyYAML, and
Shapely serve different parts of this package; matplotlib is used by clustering
plots. From the repository root:

```sh
pdm install
pdm run python your_script.py
```

PDM's scripts set `PYTHONPATH=src`. If you use an existing compatible environment
directly on Linux/macOS, use `PYTHONPATH=src python your_script.py`.

This example needs no images, devices, YAML, or saved telemetry:

```python
import numpy as np
from sft_geoloc import CameraModel, Geolocator

origin = np.array([42.0, 12.0])  # latitude, longitude in degrees
camera = CameraModel(
    intrinsic_matrix=[[800, 0, 320], [0, 800, 240], [0, 0, 1]],
    image_size_px=(640, 480),
    # Downward camera; image right is aircraft right, image top is forward.
    camera_to_body=[[0, -1, 0], [1, 0, 0], [0, 0, 1]],
)
locator = Geolocator(camera, mission_origin_latlon_deg=origin)
pixels = np.array([[320, 240], [420, 240]])

latlon, valid = locator.pixels_to_gps(
    pixels, observer_latlon_deg=origin, altitude_m=80, rpy_deg=[0, 0, 0]
)
print(latlon[valid])

ne, valid_ne = locator.pixels_to_ne(pixels, origin, 80, [0, 0, 0])
print(ne[valid_ne])  # centre near [0, 0]; second point about 10 m east
```

Examples below reuse `camera`, `locator`, `origin`, and `pixels` from this block
unless they import and construct their own objects.

## Which file to use

| File | Use it for | Main entry points |
| --- | --- | --- |
| [camera.py](camera.py) | Intrinsics, distortion, mounting, attitude, and rays | `CameraModel`, `body_to_ned` |
| [geolocator.py](geolocator.py) | Projecting pixels to GPS or ground and GPS back to pixels | `Geolocator` |
| [geodesy.py](geodesy.py) | Converting GPS, north/east map positions, and ECEF coordinates | `LocalMap`, `SphericalEarth`, `ned_to_ecef` |
| [ground.py](ground.py) | Intersecting custom rays with a sphere or plane | `intersect_sphere`, `intersect_plane` |
| [uncertainty.py](uncertainty.py) | Estimating position distributions from telemetry errors or samples | `UncertaintyModel`, `estimate_from_samples`, `GeolocationEstimate` |
| [clustering.py](clustering.py) | Combining repeated detections using their covariances | `Data`, `CentroidAnalyzer`, `plot` |
| [survey.py](survey.py) | Camera footprint spacing, clipped flight strips, and waypoint payloads | `CameraParams`, `camera_calc`, `build_transects`, `single_waypoint` |
| [heading.py](heading.py) | Converting magnetic and true headings | `magnetic_to_true`, `true_to_magnetic` |
| [px2body_matrix.py](px2body_matrix.py) | Resolving mounting configuration and correction angles | `resolve_px2body_matrix`, `load_px2body_matrix_from_yaml` |
| [check.py](check.py) | Central input validation | Domain-specific validators |
| [__init__.py](__init__.py) | Convenient package-level imports | Exported classes listed below |

Functions whose names start with `_` are implementation helpers. Use the public
entry points in this guide when writing application code.

## Units, axes, shapes, and validity

| Value | Order / shape | Units |
| --- | --- | --- |
| GPS position | `(..., 2)` → `[latitude, longitude]` | Degrees |
| Attitude | `(..., 3)` → `[roll, pitch, yaw]` | Degrees |
| Image point | `(..., 2)` → `[x, y]`; x right, y down | Pixels |
| Image size | `(width, height)` | Pixels; NumPy images instead have `(height, width, channels)` |
| Body vector | `(..., 3)` → forward, right, down | Metres, or dimensionless for rays |
| NED vector | `(..., 3)` → north, east, down | Metres, or dimensionless for rays |
| Mission-map position | `(..., 2)` → north, east | Metres |
| ECEF point | `(..., 3)` → Earth-centred Cartesian x, y, z | Metres |
| Position covariance | `(N, 2, 2)` in north/east order | Metres squared |

The camera frame follows OpenCV: x right, y down, z forward along the optical
axis. `camera_to_body` transforms that frame into aircraft forward/right/down.
Identity mounting equates camera and body axes; supply your actual mounting.
Attitude maps body into NED using extrinsic `xyz` rotations. At zero attitude,
body forward is north, right is east, and down is down. A +90° yaw turns forward
toward east. Positive down means that a camera 80 m above a local ground plane
has NED position `[0, 0, -80]`.

The Earth model is a **sphere**, with default radius `6_371_008.8` metres.
Geolocation altitude is height above its model surface. The package does not
convert terrain, geoid, or mean-sea-level datums. Supply heights consistent with
that surface, or use the local-plane API for a flat-ground approximation.

Leading dimensions broadcast using NumPy/SciPy rules. An `(N, 2)` pixel array
with one pose produces `N` results. To evaluate **every pixel at every pose**,
insert a singleton dimension explicitly; equally sized batches otherwise pair
entries rather than making a Cartesian product:

```python
attitudes = np.array([[0, 0, 0], [0, 0, 90]])
positions, hit = locator.pixels_to_gps(
    pixels[None, :, :], origin, 80, attitudes[:, None, :]
)
assert positions.shape == (2, 2, 2)  # poses, pixels, latitude/longitude
assert hit.shape == (2, 2)
```

A correctly shaped ray can still miss the surface. Projection methods returning
`(values, valid)` put NaNs at invalid entries; filter with `valid` before using
results. `body_to_pixels` and `gps_to_pixels` return only the pixel array: test
`np.isfinite(result).all(axis=-1)`. Finite projected pixels can lie outside the
image; check image bounds separately.

## Camera calibration and rays: camera.py

### Build from measured intrinsics

```python
from sft_geoloc.camera import CameraModel, body_to_ned

calibrated = CameraModel(
    intrinsic_matrix=[[800, 0, 320], [0, 900, 240], [0, 0, 1]],
    image_size_px=(640, 480),
    camera_to_body=np.eye(3),
    distortion_coefficients=[0.05, -0.01, 0, 0, 0],
)
rays = calibrated.pixel_to_body_rays([[320, 240], [400, 250]])
assert rays.shape == (2, 3)
projected = calibrated.body_to_pixels(rays * 10)
assert np.allclose(projected, [[320, 240], [400, 250]], atol=1e-6)

rotation = body_to_ned([0, 0, 90])
assert np.allclose(rotation.apply([1, 0, 0]), [0, 1, 0], atol=1e-12)
```

Intrinsics must be a finite, zero-skew pinhole matrix with positive `fx`, `fy`.
Mounting must be an orthogonal `3×3` matrix; reflections are supported. OpenCV
distortion accepts 4, 5, 8, 12, or 14 finite coefficients. The common five-value
order is `[k1, k2, p1, p2, k3]`. Calibration arrays are copied and read-only.

### Build from physical parameters or configuration

`from_parameters(focal_length_mm, pixel_size_um, image_size_px,
optical_center_px, px2body_matrix)` accepts an **already corrected** `2×2`
mounting. It sets equal pixel focal lengths, `focal_mm * 1000 / pixel_size_um`.

```python
physical_camera = CameraModel.from_parameters(
    focal_length_mm=8,
    pixel_size_um=10,
    image_size_px=(640, 480),
    optical_center_px=[320, 240],
    px2body_matrix=[[0, -1], [1, 0]],
)

camera_config = {
    "focal_length": 8,       # millimetres
    "pixel_size": 10,        # micrometres
    "image_shape": [640, 480],
    "optical_center": [320, 240],
    "px2body_matrix": [[0, -1], [1, 0]],
    "px2body_error_angle_deg": 2,
}
configured_camera = CameraModel.from_config(camera_config)
```

`from_config(geoloc_config, camera_config=None)` takes **configuration
subsections**, preferring non-None geolocation settings over camera settings.
Required settings are `focal_length` (alias `focal_length_mm`), `pixel_size`,
`image_shape`, and `px2body_matrix`. Centre aliases are `optical_center` and
`optic_center`; the default is half the image size. Optional settings are
`px2body_error_angle_deg` and `distortion_coefficients`.

This factory reads `image_shape`, not the pipeline's `camera.resolution` alias.
Use `Geolocator.from_config` with a geolocation subsection containing
`image_shape`, or adapt resolution explicitly when building your own camera.

| Method | Inputs | Output / behaviour |
| --- | --- | --- |
| `pixel_to_body_rays(pixels_px)` | `(..., 2)` pixels | Unit body-frame rays `(..., 3)`; nonfinite pixels produce NaNs |
| `body_to_pixels(points_body)` | `(..., 3)` camera-relative body vectors | `(..., 2)` pixels; on/behind-camera or nonfinite points produce NaNs |
| `rays_ned(pixels_px, rpy_deg)` | Pixels and broadcast attitudes | Unit NED rays `(..., 3)` |
| `body_to_ned(rpy_deg)` | `(..., 3)` attitudes | A SciPy `Rotation`; use `.apply()` or `.as_matrix()` |

## Pixel/GPS projection: geolocator.py

Construct `Geolocator(camera, mission_origin_latlon_deg, radius_m=6_371_008.8)`.
The mission origin fixes the north/east map used by this object. The observer is
the aircraft's **current** GPS location; it can differ from the mission origin.
Properties `mission_map` and `earth` expose the associated mapping objects.

| Method | Return value | Use |
| --- | --- | --- |
| `pixels_to_gps(pixels, observer, altitude, rpy)` | `(latlon_deg, valid)` | Intersect camera rays with the spherical Earth |
| `pixels_to_ne(pixels, observer, altitude, rpy)` | `(ne_m, valid)` | The same hits in the fixed mission map |
| `gps_to_pixels(targets, observer, altitude, rpy, target_altitude_m=0)` | Pixel array | Project GPS targets, rejecting behind-camera and Earth-occluded positions |
| `pixels_to_ground(pixels, altitude, rpy)` | `(ned_m, valid)` | Flat `down=0` plane beneath a camera at `[0,0,-altitude]` |
| `ground_homography(altitude, rpy, meters_per_pixel)` | `(H, (width, height))` | Map source pixels into an east-right, north-up ground image |

All method argument names include unit suffixes in the code. Heights in these
methods must be finite and nonnegative. The plane result is relative to the
camera's horizontal position; it is not automatically a mission-map position.

```python
targets = locator.mission_map.ne_to_gps([[0, 0], [0, 10]])
image_points = locator.gps_to_pixels(targets, origin, 80, [0, 0, 0])
visible = np.isfinite(image_points).all(axis=-1)
w, h = camera.image_size_px
inside = visible & (image_points[:, 0] >= 0) & (image_points[:, 0] < w)
inside &= (image_points[:, 1] >= 0) & (image_points[:, 1] < h)

flat_ned, ground_valid = locator.pixels_to_ground(pixels, 80, [0, 0, 0])
assert np.allclose(flat_ned[ground_valid, 2], 0)
```

### Read the full repository configuration

```python
from pathlib import Path
import yaml

config = yaml.safe_load(Path("sft_config.yaml").read_text())
configured_locator = Geolocator.from_config(config)
```

Unlike the camera factory, `Geolocator.from_config` takes the **full** config.
It reads `decision_system.geoloc`, `camera`, `manifold_center`, and optional
`decision_system.geoloc.earth_radius_m`. It does not load a YAML file itself.

### Make a ground image

```python
import cv2

image = np.zeros((480, 640, 3), dtype=np.uint8)
H, output_size = locator.ground_homography(80, [0, 0, 0], meters_per_pixel=0.2)
north_up = cv2.warpPerspective(image, H, output_size)
assert north_up.shape[:2] == (output_size[1], output_size[0])
```

Homography requires one attitude `(3,)`, one positive altitude, image dimensions
of at least two pixels, no nonzero lens distortion, and all four corner rays
hitting the ground. Undistort first and use a camera with matching undistorted
intrinsics and `distortion_coefficients=None` if the source is distorted. A
single homography describes a plane, not curved Earth or varying terrain.

## Maps and Earth coordinates: geodesy.py

`LocalMap(origin_latlon_deg, radius_m=6_371_008.8)` uses a spherical
azimuthal-equidistant map. Coordinates are north/east metres, based on distance
and bearing **from the origin**. Keep one origin for all mission detections.

```python
from sft_geoloc.geodesy import LocalMap, SphericalEarth, ned_to_ecef

local = LocalMap([42, 12])
gps = local.ne_to_gps([[0, 0], [100, 50]])
assert np.allclose(local.gps_to_ne(gps), [[0, 0], [100, 50]], atol=1e-6)

earth = SphericalEarth()
xyz = earth.geodetic_to_ecef(gps, altitude_m=[0, 80])
latlon_height = earth.ecef_to_geodetic(xyz)
assert np.allclose(latlon_height[:, :2], gps, atol=1e-10)
assert np.allclose(latlon_height[:, 2], [0, 80], atol=1e-6)

ned_rotation = ned_to_ecef([0, 0])
assert np.allclose(ned_rotation.apply([1, 0, 0]), [0, 0, 1], atol=1e-12)
```

| Entry point | Input | Output |
| --- | --- | --- |
| `LocalMap.gps_to_ne(latlon_deg)` | `(...,2)` degrees | `(...,2)` north/east metres |
| `LocalMap.ne_to_gps(ne)` | `(...,2)` north/east metres | `(...,2)` latitude/longitude degrees |
| `SphericalEarth.geodetic_to_ecef(latlon_deg, altitude_m=0)` | GPS and broadcast radial heights | `(...,3)` ECEF metres |
| `SphericalEarth.ecef_to_geodetic(xyz)` | `(...,3)` ECEF metres | `(...,3)` latitude, longitude, radial height |
| `ned_to_ecef(latlon_deg)` | GPS | SciPy `Rotation` mapping local NED vectors to ECEF |

Origins and target batches can broadcast. Map inversion is unique for distances
less than `pi * radius_m`; use a local mission origin rather than mapping a
worldwide dataset onto one mission plane. `SphericalEarth` accepts finite radial
heights, including negative heights; the higher-level geolocator requires
heights on or above its surface. This model is not the WGS84 ellipsoid.

## Ray intersections: ground.py

Import these helpers when you already have origins and direction vectors.
Directions need not have unit length. Leading dimensions broadcast; results
are `(points, valid)`, with NaNs for missed, backward, or zero-direction rays.
Zero-distance hits are accepted.

```python
from sft_geoloc.ground import intersect_plane, intersect_sphere

plane_hits, plane_valid = intersect_plane(
    origins=[0, 0, -80], directions=[[0, 0, 1], [0, 0, -1]], down_m=0
)
assert plane_valid.tolist() == [True, False]
assert np.allclose(plane_hits[0], [0, 0, 0])

sphere_hits, sphere_valid = intersect_sphere(
    origins=[6_371_008.8 + 80, 0, 0],
    directions=[-1, 0, 0],
    radius_m=6_371_008.8,
)
assert bool(sphere_valid)
```

`intersect_plane(origins, directions, down_m=0)` uses NED metres and a plane of
constant down coordinate; parallel rays are invalid. `intersect_sphere(origins,
directions, radius_m)` uses ECEF metres and returns the nearest forward hit on
an origin-centred sphere. These low-level functions do not apply camera mounting
or aircraft attitude for you.

## Uncertainty: uncertainty.py

### Generate noisy telemetry for one exposure

`UncertaintyModel` converts confidence radii into Gaussian scales. Position and
roll/pitch perturbations are clipped radially; altitude and yaw are clipped per
axis. Supply error radii at `confidence_percent`, maximum deviations, and an
integer `samples >= 2`. Length errors are metres; angular errors are degrees.

```python
from sft_geoloc import UncertaintyModel

uncertainty = UncertaintyModel(
    samples=200,
    confidence_percent=95,
    position_error_m=1,
    position_max_error_m=3,
    altitude_error_m=0.5,
    altitude_max_error_m=2,
    roll_pitch_error_deg=0.3,
    roll_pitch_max_error_deg=1,
    yaw_error_deg=0.5,
    yaw_max_error_deg=2,
)
estimate = uncertainty.estimate(
    locator, pixels, origin, 80, [0, 0, 0], rng=np.random.default_rng(7)
)
print(estimate.mean_ne_m[estimate.valid])
print(estimate.covariance_ne_m2[estimate.valid])
```

`estimate` requires one observer `(2,)`, one attitude `(3,)`, one nonnegative
height, and target pixels `(N,2)`. A seeded generator makes the result repeatable.

| `GeolocationEstimate` field | Shape | Meaning |
| --- | --- | --- |
| `mean_ne_m` | `(N,2)` | Mean target position in the locator's mission map |
| `covariance_ne_m2` | `(N,2,2)` | Sample covariance with denominator `S-1` |
| `valid` | `(N,)` | True only if **every** sample hits the model surface |
| `valid_samples` | `(N,)` | Number of surface-hit samples per target |

A below-surface altitude sample or a missed ray invalidates the whole target
distribution; its mean and covariance are NaN. The implementation does not
quietly discard those samples. Increasing sample count reduces Monte Carlo
sampling variation but does not change the chosen telemetry-error model.

### Use telemetry samples you already have

```python
from sft_geoloc.uncertainty import estimate_from_samples

sample_estimate = estimate_from_samples(
    locator,
    pixels_px=pixels,                             # (N,2)
    observer_latlon_deg=np.tile(origin, (3, 1)),   # (S,2)
    altitude_m=[79, 80, 81],                      # (S,)
    rpy_deg=[[0, 0, -0.5], [0, 0, 0], [0, 0, 0.5]],  # (S,3)
)
assert sample_estimate.mean_ne_m.shape == (2, 2)
```

All arrays must be finite and have the listed sample shapes. Identical samples
produce zero covariance. A valid estimate can still have a singular covariance;
that matters when passing it into `Data`, which requires positive definiteness.

### Configuration mapping

`UncertaintyModel.from_config(config)` takes the **geolocation subsection**:

| Config key | Model field |
| --- | --- |
| `monte_carlo` | `samples` |
| `p_value` | `confidence_percent` (percentage, not fraction) |
| `ne_error`, `ne_maxerror` | `position_error_m`, `position_max_error_m` |
| `altitude_error`, `altitude_maxerror` | `altitude_error_m`, `altitude_max_error_m` |
| `rp_error`, `rp_maxerror` | `roll_pitch_error_deg`, `roll_pitch_max_error_deg` |
| `yaw_error`, `yaw_maxerror` | `yaw_error_deg`, `yaw_max_error_deg` |

All error and maximum-error values must be finite and nonnegative;
`confidence_percent` must be strictly between 0 and 100.

## Clustering: clustering.py

### Store detections

`Data(points, covariances)` accepts `(N,D)` points with symmetric,
positive-definite `(N,D,D)` covariances. Use a common mission origin, units, and
coordinate order for every detection. A single `(D,)` point and `(D,D)`
covariance are also accepted.

```python
from sft_geoloc import Data, CentroidAnalyzer

data = Data(
    points=[[0, 0], [0.5, 0.2], [-0.3, -0.1], [20, 20], [20.2, 19.8]],
    covariances=np.repeat(np.eye(2)[None, :, :], 5, axis=0),
)
assert data[0].points.shape == (1, 2)
combined = data[:2].append(data[2:])
assert len(combined) == 5
```

`points`, `covariances` (also available as `covs`), `covs_inv`, `scale` (Cholesky
factor), and `scale_inv` are read-only arrays. `n_samples`, `n_dimensions`, and
`len(data)` describe the batch. Indexing preserves the sample axis for an integer
index. `append` returns new data and requires matching coordinate dimensions.
An empty batch must retain explicit shapes, such as `(0,2)` and `(0,2,2)`.

### Estimate candidate centroids

```python
analyzer = CentroidAnalyzer(horizon=0.95, filter=0.1)
analyzer(data, batch_size=5, passages=4, rng=np.random.default_rng(42))
print(analyzer.centroids)  # (C,D)
print(analyzer.probs)      # (C,)

# Optional seeds from an earlier fit:
analyzer(data, batch_size=5, passages=4, feedback=analyzer.centroids,
         rng=np.random.default_rng(43))
```

`horizon` is a fraction in `[0,1]` controlling the chi-square threshold for squared
Mahalanobis distance. `filter` is a fraction in `[0,1)` setting the minimum
neighbourhood population relative to the densest neighbourhood. `batch_size`
and `passages` are positive integers. Feedback has shape `(C,D)`; some earlier
centroids are reused while fresh seeds keep the search moving.

Calling the analyzer updates its attributes and returns `None`. The `radius`
property is the **squared Mahalanobis** threshold for the current dimensionality.
Without `rng`, sampling uses `np.random`, preserving `np.random.seed` behaviour.
Neighbourhood means are weighted by inverse covariance. Empty input returns
empty centroids and probabilities.

Candidates can repeat or remain close together. This analyzer does not choose a
mission-specific number of targets or assign semantic identities. `probs` ranks
candidate consistency; it is not a calibrated probability that a target exists.
The decision pipeline performs further selection.

### Plot results

```python
from sft_geoloc.clustering import plot

# Run on a desktop with a display; selected must have shape (K,2).
plot(data, selected=analyzer.centroids[:1], centroids=analyzer.centroids, covs_p=0.8)
```

Plotting requires two-dimensional data and `0 < covs_p < 1`. It displays input
positions, covariance ellipses, candidates, and selected points. Matplotlib is
loaded when the plot function is called.

## Survey planning: survey.py

Use this module for geometry and payload construction. It performs no vehicle
connection, mission upload, or mission start.

```python
from sft_geoloc import LocalMap
from sft_geoloc.survey import CameraParams, camera_calc, build_transects, single_waypoint

survey_camera = CameraParams(
    sensor_width_mm=12,
    sensor_height_mm=8,
    image_width_px=1200,
    image_height_px=800,
    focal_length_mm=8,
    frontal_overlap=0.5,
    side_overlap=0.2,
    altitude_m=40,
    landscape=True,
)
spacing = camera_calc(survey_camera)
assert spacing["adjusted_footprint_side_m"] == 48
assert spacing["adjusted_footprint_frontal_m"] == 20

survey_map = LocalMap([42, 12])
boundary = survey_map.ne_to_gps([[0, 0], [0, 180], [120, 180], [120, 0]])
transects, info = build_transects(boundary, survey_camera, grid_angle_deg=30)
for transect in transects:
    start_lat, start_lon = transect.start[0]
    end_lat, end_lon = transect.end[0]
    print(start_lat, start_lon, end_lat, end_lon, transect.altitude_m)

waypoint_items = single_waypoint(42, 12, altitude_m=40)
assert waypoint_items[0]["command"] == 16
```

Camera sensor dimensions, focal length, and altitude must be positive. Image
sizes must be positive whole pixel counts. Overlaps are fractions in `[0,1)`;
`landscape` is boolean. Both physical sensor dimensions determine footprint
width and height: `altitude * sensor_dimension / focal_length`. Landscape places
sensor width across the flight path; portrait swaps side and frontal roles.
Image resolution is validated but does not change the physical footprint.

`build_transects(polygon, camera, grid_angle_deg=0)` accepts at least three
latitude/longitude vertices. The projected polygon must be simple and have
positive area. Concave boundaries are supported; holes and multi-polygons are
not accepted by this vertex-list interface. `grid_angle_deg` is counterclockwise
from **east**, unlike aircraft yaw measured from north.

Lanes are centred in equal-width strips no farther apart than the adjusted side
footprint and alternate direction. Disconnected intersections yield separate
segments. Segment endpoints retain the mission API's single-element list of
`(latitude, longitude)` tuples. Connecting legs between segments, terrain
clearance, and complete mission sequencing are the caller's responsibility.

`info` contains both footprint spacings, `n_transects`, and `trigger_distance_m`
(the adjusted frontal spacing). `single_waypoint` accepts nonnegative relative
altitude and returns a one-item list: MAVLink frame 3, command 16, acceptance
radius 2 m, and unspecified yaw (`param4=NaN`). Upload this payload separately.

## Heading conversion: heading.py

Supply magnetic declination for the flight **location and date**. The convention
is east-positive: true heading equals magnetic heading plus declination.

```python
from sft_geoloc.heading import magnetic_to_true, true_to_magnetic

magnetic = np.array([359, 90])
true = magnetic_to_true(magnetic, declination_deg=5)
assert np.allclose(true, [4, 95])
assert np.allclose(true_to_magnetic(true, 5), magnetic)
```

Both arguments are finite degrees and can broadcast. Results wrap to `[0,360)`;
scalar inputs produce zero-dimensional NumPy arrays, for which `.item()` gives a
Python scalar. These helpers do not obtain declination from a magnetic model or
modify telemetry automatically.

## Mounting configuration: px2body_matrix.py

```python
from sft_geoloc.px2body_matrix import resolve_px2body_matrix, load_px2body_matrix_from_yaml

mounting = resolve_px2body_matrix(
    geoloc_config={"px2body_error_angle_deg": 2},
    camera_config={"px2body_matrix": [[0, -1], [1, 0]]},
)
assert np.asarray(mounting).shape == (2, 2)

repository_mounting = load_px2body_matrix_from_yaml("sft_config.yaml")
```

Resolution is independent for each key: a non-None value from geolocator config
wins over camera config. The angle defaults to zero. The result is a Python list
of two rows, calculated as `Rz(error_angle_deg) @ px2body_matrix`. The resolver
checks shape and finiteness; `CameraModel` additionally checks orthogonality.
The YAML loader reads `decision_system.geoloc` and `camera` itself.

**Apply the correction once.** `CameraModel.from_config` already applies
`px2body_error_angle_deg`. If you use the resolved matrix, pass it to
`from_parameters`, or clear the correction angle when constructing the camera.

## Validation: check.py

`check.py` is the central error manager. Public APIs call it automatically after
converting their inputs to arrays; application code normally catches errors from
the API rather than invoking every check itself. Checks return `None` on success
and raise descriptive `ValueError`s on invalid inputs. A missing mounting in the
resolver raises `KeyError`; missing required config keys can also raise
`KeyError`. File-loading failures remain ordinary file/YAML exceptions.

```python
from sft_geoloc import check

check.vector_shape(np.zeros((5, 2)), size=2, name="pixels")
check.finite([42, 12], name="observer")
try:
    check.estimate_altitude(-1)
except ValueError as error:
    print(error)
```

| Area | Validators |
| --- | --- |
| Common shapes and numbers | `vector_shape`, `finite`, `finite_scalar`, `finite_nonnegative`, `positive_scalar`, `positive_integer` |
| Camera calibration | `intrinsic_matrix`, `image_size`, `camera_to_body`, `distortion_coefficients`, `camera_config`, `focal_length_pixel_size`, `image_shape`, `optical_center`, `px2body_matrix`, `mounting_required` |
| Geography and rays | `latitude`, `radius_scalar`, `radius`, `surface_altitude`, `ray_shape` |
| Ground images | `meters_per_pixel`, `homography_pose`, `homography_altitude`, `homography_distortion`, `homography_image_size`, `homography_corners` |
| Uncertainty | `pixels_shape`, `altitude_samples`, `sample_shapes`, `finite_samples`, `sample_count`, `confidence_percent`, `estimate_pose`, `estimate_altitude` |
| Clustering | `clustering_data`, `clustering_dimensions`, `centroid_options`, `centroid_feedback`, `clustering_plot` |
| Survey and waypoints | `survey_camera`, `survey_vertices`, `survey_polygon`, `waypoint_position` |

Shape validators generally expect already-converted NumPy arrays. Add new
explicit input validation here when extending the package, and call checks
directly instead of using `assert`; they remain active under `python -O`.
Impossible geometric hits use validity masks rather than input-validation errors.

## Package exports: `__init__.py`

These imports are available at package level:

```python
from sft_geoloc import (
    CameraModel, CameraParams, CentroidAnalyzer, Data, GeolocationEstimate,
    Geolocator, LocalMap, Transect, UncertaintyModel,
)
```

`CameraModel` is loaded immediately; other exported classes are loaded lazily
when requested. Import standalone functions from their defining module, such as
`from sft_geoloc.heading import magnetic_to_true`. There is no package CLI in
`__init__.py`; launch `tools/geoloc.py` for interactive tooling.

## Pipeline integration, troubleshooting, and verification

The application adapter in `src/sft_cv/decision_system/geoloc/geoloc.py` uses this
package to return target means, covariances, and a `LocalMap`. Convert its
north/east outputs with `LocalMap.ne_to_gps`. The runtime target simulator uses
`gps_to_pixels` to project configured target corners into images. Tool workflows
share these conventions through the same package.

For repeated detections, the usual sequence is:

1. Build one calibrated camera and one locator with a fixed mission origin.
2. Use one exposure's GPS, surface-relative altitude, and degree attitude.
3. Project pixels directly, or call an uncertainty estimator.
4. Filter invalid estimates and ensure covariances meet the clustering contract.
5. Put detections into `Data`, estimate candidates, and perform mission-specific selection.
6. Convert selected north/east positions back to GPS with the same mission map.

| Symptom | Check / action |
| --- | --- |
| Import fails for `RigidTransform` or batched rotation operations | Recreate the environment using project Python/SciPy requirements |
| Missing focal length, image shape, or mounting | Pass the correct subsection and provide the required camera settings |
| Unexpected 90° turns, reflections, or huge attitude changes | Check mounting, degree units, and north/east versus east/north order; use `geoloc debug` |
| Projection returns NaNs | Inspect the validity mask, altitude, mounting, camera-facing direction, and Earth occlusion |
| Finite pixel is outside the image | Apply explicit width/height bounds after projection |
| Homography rejects distortion or a corner | Undistort with matching intrinsics, use positive altitude, and ensure all corners hit the plane |
| Uncertainty result is invalid despite some valid samples | Every sample must hit; inspect `valid_samples` and the telemetry-error bounds |
| `Data` rejects covariance | Check matching dimensions, finiteness, symmetry, and strictly positive eigenvalues; zero-noise estimates may be singular |
| Heading result is offset | Supply declination using east-positive degrees and the correct location/date |
| Survey grid direction differs from aircraft yaw | Grid angle starts at east and increases counterclockwise |

Run the focused library checks from the repository root:

```sh
pdm run python -m pytest -q tests/test_sft_geoloc.py tests/test_geoloc_utilities.py tests/test_px2body_matrix.py tests/test_geoloc_migration.py
```

Geometry regression outputs are stored in
`tests/assets/geolocation_reference.json`; the package does not require the
removed `sft_math` implementation.
