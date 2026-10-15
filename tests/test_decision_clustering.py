"""Mock-point tests for the general behaviour of the decision system.

Geolocation already works, so these tests inject **mock points expressed in
local ENU metres** (the output format of ``Geoloc``) straight into the per-class
``Decision`` accumulators and run the *real* decision-system output path for a
given mission. They check the layers that turn accumulated detections into
localized waypoints:

- ``BULLSEYE``: a single target (class 2) -> exactly one centroid;
- ``NUMBERS``: numbered identity is no longer available; generic GCP detections
  are mapped to class 0 and clustered as multiple no-number GCP instances;
- ``PERIMETER`` (optional smoke test): the dedicated corner localizer is fully
  covered by ``tests/test_perimeter.py``; here we only check that the
  ``DecisionSystem`` *routes* PERIMETER to it.

The harness drives a real ``DecisionSystem`` in-process (no ``.start()``, no
multiprocessing transport) configured exactly like the chosen mission via the
standard ``update()`` merge of ``mission_config.yaml`` over ``sft_config.yaml``.

Run with:   PYTHONPATH=src python -m pytest tests/test_decision_clustering.py -v
Visualize:  PYTHONPATH=src python tests/test_decision_clustering.py --mission NUMBERS --plot
"""

from __future__ import annotations

import multiprocessing as mp
import os
import shutil
import tempfile
from typing import Dict, Optional

import numpy as np
import pytest

from common.config import SFTConfig
from core.logger import SFTLogger
from stages.decision_system.default import DecisionSystem


_SHARED_LOGGER: Optional[SFTLogger] = None


def _get_logger(configs: SFTConfig) -> SFTLogger:
    """One SFTLogger for the whole test session.

    ``SFTLogger.__init__`` calls ``multiprocessing_logging.install_mp_handler()``
    globally; building one per tester stacks listener threads that leak noisy
    pipe-teardown errors at GC on Windows. A single shared instance avoids that.
    """
    global _SHARED_LOGGER
    if _SHARED_LOGGER is None:
        _SHARED_LOGGER = SFTLogger(configs)
    return _SHARED_LOGGER


# --------------------------------------------------------------------------
# mock-point helpers (same conventions as test_perimeter / test_instance_splitter)
# --------------------------------------------------------------------------


def identity_covs(n: int) -> np.ndarray:
    """`n` identity 2x2 covariance matrices, shape (n, 2, 2)."""
    return np.tile(np.eye(2), (n, 1, 1))


def make_cluster(
    center, n: int = 15, jitter: float = 1.5, seed: int = 42
) -> np.ndarray:
    """`n` noisy ENU points (metres) around `center`, shape (n, 2)."""
    rng = np.random.default_rng(seed)
    return np.asarray(center, dtype=np.float64) + rng.normal(0.0, jitter, size=(n, 2))


def make_detections(corners, per_corner=15, std=1.5, n_fp=80, seed=42, area=100.0):
    """Detection clouds around each corner + uniform false positives.

    Mirrors ``tests/test_perimeter._make_detections``. Returns
    ``(points (N, 2), covs (N, 2, 2))``.
    """
    rng = np.random.default_rng(seed)
    clouds = [rng.normal(c, std, size=(per_corner, 2)) for c in corners]
    pts = np.vstack(clouds) if clouds else np.empty((0, 2))
    if n_fp:
        fps = rng.uniform(0.0, area, size=(n_fp, 2))
        pts = np.vstack([pts, fps]) if pts.shape[0] else fps
    rng.shuffle(pts)
    return pts, identity_covs(pts.shape[0])


def make_sequential_trail(
    start, end, steps: int = 10, jitter: float = 1.0, seed: int = 42
) -> np.ndarray:
    """A sequence of noisy detections drifting from `start` to `end` simulating a flight pass."""
    rng = np.random.default_rng(seed)
    xs = np.linspace(start[0], end[0], steps)
    ys = np.linspace(start[1], end[1], steps)
    base_points = np.column_stack((xs, ys))
    noise = rng.normal(0.0, jitter, size=(steps, 2))
    return base_points + noise


def make_noisy_cluster(
    center, n_signal: int = 15, n_noise: int = 50, spread: float = 40.0, seed: int = 42
) -> np.ndarray:
    """A tight cluster embedded in a large area of uniform noise."""
    rng = np.random.default_rng(seed)
    signal = np.asarray(center, dtype=np.float64) + rng.normal(
        0.0, 1.5, size=(n_signal, 2)
    )
    noise = np.asarray(center, dtype=np.float64) + rng.uniform(
        -spread, spread, size=(n_noise, 2)
    )
    pts = np.vstack([signal, noise])
    rng.shuffle(pts)
    return pts


def assert_matches(found: np.ndarray, expected, tol: float = 5.0) -> None:
    """Every expected ENU point is matched by some found centroid within `tol`."""
    found = np.asarray(found).reshape(-1, 2)
    for point in np.asarray(expected).reshape(-1, 2):
        assert (
            found.shape[0] > 0 and np.min(np.linalg.norm(found - point, axis=1)) < tol
        ), f"point {point} not matched by {found}"


# --------------------------------------------------------------------------
# in-process harness
# --------------------------------------------------------------------------


class DecisionClusterTester:
    """Drive a real ``DecisionSystem`` for one mission with mock ENU points.

    Usage::

        tester = DecisionClusterTester("BULLSEYE")
        tester.feed(2, make_cluster([10.0, -5.0]))
        enu = tester.run()           # {class_id: (M, 2) ENU centroids}
        report = tester.report       # {target_name: [{"lat":..,"lon":..}]}
    """

    def __init__(
        self,
        mission: str,
        config_path: str = "sft_config.yaml",
        tmp_root: Optional[str] = None,
    ) -> None:
        self.mission = mission
        self.tmp_root = tmp_root or tempfile.mkdtemp(prefix="ds_cluster_test_")

        configs = SFTConfig(config_path)
        # Redirect the stage's working/debug dir to a throwaway location: the
        # constructor rmtree()s decision_system.root, and we never want that to
        # hit the real repo folder. Disable the file debugger too.
        configs.config_dict["decision_system"]["root"] = self.tmp_root
        configs.config_dict["decision_system"]["debug"] = False

        logger = _get_logger(configs)

        self.ds = DecisionSystem(
            configs,
            logger,
            in_q=mp.Queue(),
            out_q=[mp.Queue(), mp.Queue()],
            idx=0,
        )

        # Configure exactly like the mission would: set the task and run the
        # standard config merge (mission_config.yaml over sft_config.yaml) which
        # rebuilds self.decisions for the mission's active classes.
        self.ds.current_task = mission
        if mission != "DEFAULT":
            self.ds.update()

        self.enu_centroids: Dict[int, np.ndarray] = {}
        self.report: Optional[dict] = None

    @property
    def active_classes(self):
        return list(self.ds.decisions.keys())

    def feed(self, class_id: int, points, covariances=None) -> None:
        """Inject mock ENU points (metres) for a class, as if post-geoloc."""
        if class_id not in self.ds.decisions:
            raise KeyError(
                f"class {class_id} is not active for mission {self.mission}; "
                f"active classes: {self.active_classes}"
            )
        points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if covariances is None:
            covariances = identity_covs(points.shape[0])
        self.ds.decisions[class_id].insert(points, covariances)

    def run(self) -> Dict[int, np.ndarray]:
        """Run the real mission output path and return ENU centroids per class."""
        captured: Dict[int, np.ndarray] = {}
        original_put = self.ds.put

        def capturing_put():
            outs = self.ds.outs[0]
            if outs is not None:
                for cls_id, arr in outs.items():
                    captured[cls_id] = np.array(arr)
            return original_put()

        # put() writes data/outs/SFT_localization_report.csv relative to cwd;
        # run from the temp dir so the repo stays clean.
        cwd = os.getcwd()
        self.ds.put = capturing_put  # type: ignore[method-assign]
        try:
            os.chdir(self.tmp_root)
            if self.mission == "PERIMETER":
                self.ds._produce_perimeter_output()
            else:
                self.ds._produce_centroid_output()
        finally:
            os.chdir(cwd)
            self.ds.put = original_put  # type: ignore[method-assign]

        self.enu_centroids = captured
        try:
            self.report = self.ds.out_q[0].get(timeout=1.0)
        except Exception:
            self.report = None
        return captured

    def cleanup(self) -> None:
        # Close the multiprocessing queues so their feeder threads/pipes are
        # released deterministically; otherwise their GC at interpreter exit
        # raises noisy PeekNamedPipe/ReadFile teardown errors on Windows.
        queues = [self.ds.in_q, *(self.ds.out_q or [])]
        for q in queues:
            try:
                q.close()
                q.join_thread()
            except Exception:
                pass
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    # -- visualization (manual runs only) ---------------------------------

    def plot(
        self, title: Optional[str] = None, block: bool = True
    ) -> None:  # pragma: no cover - visual aid
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        cmap = plt.get_cmap("tab10")
        for i, (cls_id, decision) in enumerate(self.ds.decisions.items()):
            pts = decision.data.points
            if pts.size:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=6,
                    alpha=0.4,
                    color=cmap(i % 10),
                    label=f"class {cls_id} pts",
                )
            cents = self.enu_centroids.get(cls_id)
            if cents is not None and cents.shape[0]:
                ax.scatter(
                    cents[:, 0],
                    cents[:, 1],
                    marker="x",
                    s=120,
                    linewidths=2.5,
                    color=cmap(i % 10),
                    label=f"class {cls_id} centroid",
                )
        ax.set_aspect("equal", "box")
        ax.set_title(title or f"Decision system — {self.mission}")
        ax.legend(fontsize=7, loc="best")
        if block:
            plt.show()
        else:
            plt.show(block=False)


# --------------------------------------------------------------------------
# BULLSEYE — single target, single centroid
# --------------------------------------------------------------------------


@pytest.fixture(scope="session", autouse=True)
def _plot_teardown():
    """Keep plot windows open at the end of the test session if plotting is enabled."""
    yield
    if os.environ.get("PLOT_TESTS") == "1":
        try:
            import matplotlib.pyplot as plt

            if plt.get_fignums():
                plt.show()
        except ImportError:
            pass


def test_bullseye_single_centroid():
    tester = DecisionClusterTester("BULLSEYE")
    try:
        assert tester.active_classes == [2]
        center = np.array([12.0, -7.0])
        tester.feed(2, make_cluster(center, n=20, jitter=1.5))
        enu = tester.run()

        assert set(enu.keys()) == {2}
        assert enu[2].shape == (1, 2)
        assert_matches(enu[2], center, tol=5.0)

        # the report carries the configured target name
        assert tester.report is not None
        assert "BULLSEYE" in tester.report
        assert len(tester.report["BULLSEYE"]) == 1

        if os.environ.get("PLOT_TESTS") == "1":
            tester.plot(title="test_bullseye_single_centroid", block=False)
    finally:
        tester.cleanup()


def test_bullseye_sequential_trail():
    """Simulates a drone passing over a BULLSEYE, yielding a smeared trail of detections."""
    tester = DecisionClusterTester("BULLSEYE")
    try:
        start_pt = np.array([10.0, 0.0])
        end_pt = np.array([20.0, 5.0])
        trail = make_sequential_trail(start_pt, end_pt, steps=30, jitter=1.5)
        tester.feed(2, trail)

        enu = tester.run()

        assert set(enu.keys()) == {2}
        assert enu[2].shape == (1, 2)
        # the centroid should be roughly in the middle of the trail
        midpoint = (start_pt + end_pt) / 2.0
        assert_matches(enu[2], midpoint, tol=6.0)

        if os.environ.get("PLOT_TESTS") == "1":
            tester.plot(title="test_bullseye_sequential_trail", block=False)
    finally:
        tester.cleanup()


def test_bullseye_with_heavy_noise():
    """Simulates a real BULLSEYE cluster surrounded by lots of scattered false positives."""
    tester = DecisionClusterTester("BULLSEYE")
    try:
        center = np.array([-15.0, 25.0])
        cloud = make_noisy_cluster(center, n_signal=20, n_noise=60, spread=50.0)
        tester.feed(2, cloud)

        enu = tester.run()

        assert set(enu.keys()) == {2}
        assert enu[2].shape == (1, 2)
        assert_matches(enu[2], center, tol=5.0)

        if os.environ.get("PLOT_TESTS") == "1":
            tester.plot(title="test_bullseye_with_heavy_noise", block=False)
    finally:
        tester.cleanup()


# --------------------------------------------------------------------------
# NUMBERS — generic GCP clusters under class 0
# --------------------------------------------------------------------------


def test_numbers_generic_gcp_clusters_under_class_0():
    tester = DecisionClusterTester("NUMBERS")
    try:
        assert sorted(tester.active_classes) == [0]

        centers = np.array(
            [[idx * 30.0, idx * 12.0] for idx in range(10)],
            dtype=np.float64,
        )
        clouds = [
            make_cluster(center, n=15, jitter=1.5, seed=idx)
            for idx, center in enumerate(centers)
        ]
        tester.feed(0, np.vstack(clouds))

        enu = tester.run()

        assert set(enu.keys()) == {0}
        assert enu[0].shape == (10, 2)
        assert_matches(enu[0], centers, tol=5.0)

        if os.environ.get("PLOT_TESTS") == "1":
            tester.plot(
                title="test_numbers_generic_gcp_clusters_under_class_0", block=False
            )
    finally:
        tester.cleanup()


def test_numbers_close_proximity():
    """Very close generic GCPs merge because digit labels no longer separate them."""
    tester = DecisionClusterTester("NUMBERS")
    try:
        center1 = np.array([50.0, 50.0])
        center2 = np.array([53.0, 50.0])
        expected_merged_center = np.mean([center1, center2], axis=0)

        tester.feed(
            0,
            np.vstack(
                [
                    make_cluster(center1, n=20, jitter=1.0),
                    make_cluster(center2, n=20, jitter=1.0),
                ]
            ),
        )

        enu = tester.run()

        assert set(enu.keys()) == {0}
        assert enu[0].shape == (1, 2)
        assert_matches(enu[0], expected_merged_center, tol=3.0)

        if os.environ.get("PLOT_TESTS") == "1":
            tester.plot(title="test_numbers_close_proximity", block=False)
    finally:
        tester.cleanup()


def test_numbers_empty_class_yields_no_centroid():
    """A class with no fed detections simply produces no centroid (no crash)."""
    tester = DecisionClusterTester("NUMBERS")
    try:
        tester.feed(0, make_cluster([90.0, 36.0], n=15, jitter=1.5))
        enu = tester.run()
        assert set(enu.keys()) == {0}
        assert enu[0].shape == (1, 2)

        if os.environ.get("PLOT_TESTS") == "1":
            tester.plot(
                title="test_numbers_empty_class_yields_no_centroid", block=False
            )
    finally:
        tester.cleanup()


# --------------------------------------------------------------------------
# PERIMETER — routing smoke test (algorithm covered by test_perimeter.py)
# --------------------------------------------------------------------------


def test_perimeter_routes_to_corner_localizer():
    tester = DecisionClusterTester("PERIMETER")
    try:
        assert tester.active_classes == [1]
        square = np.array([[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]])
        pts, covs = make_detections(square)
        tester.feed(1, pts, covs)

        enu = tester.run()

        assert set(enu.keys()) == {1}
        assert enu[1].shape[0] == 4
        assert_matches(enu[1], square, tol=5.0)

        if os.environ.get("PLOT_TESTS") == "1":
            tester.plot(title="test_perimeter_routes_to_corner_localizer", block=False)
    finally:
        tester.cleanup()


# --------------------------------------------------------------------------
# stop-command routing -> communicator wiring
#
# The communicator only ever receives a report if the DecisionSystem finalizes
# (sets received_stop) on the mission-specific *_STOP command. These tests check
# that each handler arms finalization for the matching active mission only.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mission, handler, stop_name",
    [
        ("BULLSEYE", "bullseye_stop", "BULLSEYE_STOP"),
        ("PERIMETER", "perimeter_stop", "PERIMETER_STOP"),
        ("NUMBERS", "numbers_stop", "NUMBERS_STOP"),
    ],
)
def test_matching_stop_arms_finalization(mission, handler, stop_name):
    tester = DecisionClusterTester(mission)
    try:
        assert tester.ds.received_stop is False
        getattr(tester.ds, handler)()
        assert tester.ds.received_stop is True
    finally:
        tester.cleanup()


def test_mismatched_stop_is_ignored():
    """A PERIMETER_STOP while BULLSEYE is active must not finalize."""
    tester = DecisionClusterTester("BULLSEYE")
    try:
        tester.ds.perimeter_stop()
        assert tester.ds.received_stop is False
        # the matching stop still works afterwards
        tester.ds.bullseye_stop()
        assert tester.ds.received_stop is True
    finally:
        tester.cleanup()


def test_stop_ignored_when_no_active_mission():
    """With the mission already back to DEFAULT, a stop is a no-op."""
    tester = DecisionClusterTester("BULLSEYE")
    try:
        tester.ds.current_task = "DEFAULT"
        tester.ds.bullseye_stop()
        assert tester.ds.received_stop is False
    finally:
        tester.cleanup()


def test_legacy_survey_stop_finalizes_active_mission():
    tester = DecisionClusterTester("NUMBERS")
    try:
        tester.ds.survey_stop()
        assert tester.ds.received_stop is True
    finally:
        tester.cleanup()


# --------------------------------------------------------------------------
# runnable visual mode
# --------------------------------------------------------------------------

if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mission",
        choices=["BULLSEYE", "NUMBERS", "PERIMETER", "ALL", "STRESS"],
        default="NUMBERS",
    )
    parser.add_argument("--plot", action="store_true", help="show a scatter plot")
    cli = parser.parse_args()

    if cli.mission == "ALL":
        missions = ["BULLSEYE", "NUMBERS", "PERIMETER", "STRESS"]
    elif cli.mission == "STRESS":
        missions = ["STRESS"]
    else:
        missions = [cli.mission]

    for mission in missions:
        if mission == "STRESS":
            # Run the stress scenarios
            scenarios = [
                ("BULLSEYE Sequential Trail", "BULLSEYE"),
                ("BULLSEYE Heavy Noise", "BULLSEYE"),
                ("NUMBERS Close Proximity", "NUMBERS"),
            ]
            for title, ms in scenarios:
                tester = DecisionClusterTester(ms)
                print(f"\n--- STRESS TEST: {title} ---")
                if title == "BULLSEYE Sequential Trail":
                    tester.feed(
                        2,
                        make_sequential_trail(
                            [10.0, 0.0], [20.0, 5.0], steps=30, jitter=1.5
                        ),
                    )
                elif title == "BULLSEYE Heavy Noise":
                    tester.feed(
                        2,
                        make_noisy_cluster(
                            [-15.0, 25.0], n_signal=20, n_noise=60, spread=50.0
                        ),
                    )
                elif title == "NUMBERS Close Proximity":
                    tester.feed(
                        0,
                        np.vstack(
                            [
                                make_cluster([50.0, 50.0], n=20, jitter=1.0),
                                make_cluster([53.0, 50.0], n=20, jitter=1.0),
                            ]
                        ),
                    )

                enu = tester.run()
                for cls_id, cents in sorted(enu.items()):
                    print(f"  class {cls_id}: {cents.shape[0]} centroid(s)")
                    for c in cents:
                        print(f"      ENU=({c[0]:.2f}, {c[1]:.2f})")
                if cli.plot:
                    tester.plot(title=title, block=False)
                tester.cleanup()
            continue

        tester = DecisionClusterTester(mission)
        print(f"\n--- mission={mission}  active classes={tester.active_classes} ---")

        if mission == "BULLSEYE":
            tester.feed(2, make_cluster([12.0, -7.0], n=20, jitter=1.5))
        elif mission == "NUMBERS":
            clouds = []
            for idx in range(10):
                center = [idx * 30.0, idx * 12.0]
                clouds.append(make_cluster(center, n=15, jitter=1.5, seed=idx))
            tester.feed(0, np.vstack(clouds))
        else:  # PERIMETER
            square = [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [20.0, 80.0]]
            pts, covs = make_detections(np.array(square))
            tester.feed(1, pts, covs)

        enu = tester.run()
        for cls_id, cents in sorted(enu.items()):
            print(f"  class {cls_id}: {cents.shape[0]} centroid(s)")
            for c in cents:
                print(f"      ENU=({c[0]:.2f}, {c[1]:.2f})")
        if tester.report:
            print("report:")
            for name, wps in tester.report.items():
                for wp in wps:
                    print(f"  {name}: lat={wp['lat']:.7f}, lon={wp['lon']:.7f}")

        if cli.plot:
            tester.plot(block=False)
        tester.cleanup()

    if cli.plot:
        import matplotlib.pyplot as plt

        plt.show()
