"""Tests for decision system bounding box size filtering.

Tests the _filter_bboxes_by_size() method to ensure detections are correctly
rejected based on bounding box dimensions. The minimum threshold is 70 pixels
for both width and height.

Run with:   PYTHONPATH=src python -m pytest tests/test_decision_system_bbox_filtering.py -v
"""

import numpy as np
import pytest

from stages.decision_system.default import DecisionSystem
from common.structs import PhotoExif
from common.mavlink_structs import PixData


class TestBboxFiltering:
    """Test suite for bounding box size filtering."""

    def test_empty_bbox_list(self):
        """Empty bbox list returns empty arrays."""
        coords, classes = DecisionSystem._filter_bboxes_by_size([], min_size=70)
        assert coords.shape == (0, 2)
        assert classes.shape == (0,)

    def test_all_valid_bboxes(self):
        """All bboxes above threshold are kept."""
        bboxes = [
            {"bbox": [0, 0, 100, 100], "class": 0},
            {"bbox": [0, 0, 80, 80], "class": 1},
            {"bbox": [0, 0, 70, 70], "class": 2},  # exactly at threshold
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 3
        assert np.allclose(classes, [0, 1, 2])

    def test_all_invalid_bboxes(self):
        """All bboxes below threshold are dropped."""
        bboxes = [
            {"bbox": [0, 0, 50, 50], "class": 0},
            {"bbox": [0, 0, 69, 69], "class": 1},  # just below threshold
            {"bbox": [0, 0, 30, 30], "class": 2},
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 0
        assert classes.shape[0] == 0

    def test_mixed_valid_invalid_bboxes(self):
        """Only bboxes above threshold are kept in mixed scenario."""
        bboxes = [
            {"bbox": [0, 0, 100, 100], "class": 0},      # valid
            {"bbox": [0, 0, 50, 50], "class": 1},        # invalid: too small
            {"bbox": [0, 0, 70, 100], "class": 2},       # valid
            {"bbox": [0, 0, 60, 80], "class": 3},        # invalid: width too small
            {"bbox": [0, 0, 100, 60], "class": 4},       # invalid: height too small
            {"bbox": [0, 0, 70, 70], "class": 5},        # valid: exactly at threshold
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 3
        # Check that only the valid classes are kept
        assert np.allclose(np.sort(classes), [0, 2, 5])

    def test_width_below_threshold_rejected(self):
        """Bboxes with width below 70 are rejected."""
        bboxes = [
            {"bbox": [0, 0, 69, 100], "class": 0},  # width = 69, should be rejected
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 0

    def test_height_below_threshold_rejected(self):
        """Bboxes with height below 70 are rejected."""
        bboxes = [
            {"bbox": [0, 0, 100, 69], "class": 0},  # height = 69, should be rejected
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 0

    def test_centroid_computation(self):
        """Centroids are correctly computed for valid bboxes."""
        bboxes = [
            {"bbox": [10, 20, 110, 120], "class": 0},  # centroid should be (60, 70)
            {"bbox": [0, 0, 70, 70], "class": 1},      # centroid should be (35, 35)
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 2
        assert np.allclose(coords[0], [60, 70])
        assert np.allclose(coords[1], [35, 35])

    def test_class_labels_preserved(self):
        """Class labels are correctly preserved after filtering."""
        bboxes = [
            {"bbox": [0, 0, 100, 100], "class": 10},
            {"bbox": [0, 0, 50, 50], "class": 11},  # rejected
            {"bbox": [0, 0, 80, 80], "class": 12},
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert np.allclose(classes, [10, 12])

    def test_custom_threshold(self):
        """Custom threshold can be specified."""
        bboxes = [
            {"bbox": [0, 0, 100, 100], "class": 0},
            {"bbox": [0, 0, 50, 50], "class": 1},
            {"bbox": [0, 0, 75, 75], "class": 2},
        ]
        # With threshold=100, only first bbox should pass
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=100)
        assert coords.shape[0] == 1
        assert classes[0] == 0

    def test_negative_coordinates(self):
        """Bboxes with negative coordinates are handled correctly."""
        bboxes = [
            {"bbox": [-50, -50, 50, 50], "class": 0},   # valid: 100x100
            {"bbox": [-20, -20, 40, 40], "class": 1},   # valid: 60x60... wait no that's invalid
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 1
        assert classes[0] == 0

    def test_floating_point_bbox_coordinates(self):
        """Floating point bbox coordinates are handled correctly."""
        bboxes = [
            {"bbox": [10.5, 20.3, 80.7, 90.5], "class": 0},  # 70.2 x 70.2, valid
            {"bbox": [0.0, 0.0, 69.5, 69.5], "class": 1},    # rejected
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 1
        assert classes[0] == 0

    def test_large_scale_bboxes(self):
        """Large scale bboxes are handled correctly."""
        bboxes = [
            {"bbox": [0, 0, 10000, 8000], "class": 0},
            {"bbox": [100, 200, 300, 400], "class": 1},  # 200x200, valid
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 2

    def test_boundary_exactly_70_pixels(self):
        """Bboxes with exactly 70 pixels on both sides are kept."""
        bboxes = [
            {"bbox": [0, 0, 70, 70], "class": 0},
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 1
        assert classes[0] == 0

    def test_boundary_69_pixels_rejected(self):
        """Bboxes with exactly 69 pixels are rejected."""
        bboxes = [
            {"bbox": [0, 0, 69, 69], "class": 0},
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 0

    def test_multiple_classes_same_detection(self):
        """Multiple detections of different classes are correctly filtered."""
        bboxes = [
            {"bbox": [0, 0, 100, 100], "class": 0},    # valid
            {"bbox": [10, 10, 50, 50], "class": 1},    # invalid
            {"bbox": [0, 0, 75, 75], "class": 2},      # valid
            {"bbox": [0, 0, 70, 80], "class": 3},      # valid
            {"bbox": [0, 0, 40, 40], "class": 4},      # invalid
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.shape[0] == 3
        valid_classes = np.sort(classes)
        assert np.allclose(valid_classes, [0, 2, 3])

    def test_output_dtypes(self):
        """Output arrays have correct dtypes."""
        bboxes = [
            {"bbox": [0, 0, 100, 100], "class": 0},
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes, min_size=70)
        assert coords.dtype == np.float64
        assert classes.dtype == np.int64

    def test_coords_shape(self):
        """Output coordinates have correct shape (N, 2)."""
        bboxes = [
            {"bbox": [0, 0, 100, 100], "class": 0},
            {"bbox": [0, 0, 80, 80], "class": 1},
            {"bbox": [0, 0, 70, 70], "class": 2},
        ]
        coords, classes = DecisionSystem._filter_bboxes_by_size(bboxes)
        assert coords.shape == (3, 2)
        assert classes.shape == (3,)

class TestDynamicBboxScaling:
    """Test suite for dynamic scaling of bounding box size threshold based on altitude."""

    def _create_mock_photo(self, altitude: float, bboxes: list) -> PhotoExif:
        pix_data = PixData(
            alt_rel=altitude,
            lat=41.0,
            lon=12.0,
            roll=0.0,
            pitch=0.0,
            yaw=0.0,
        )
        photo = PhotoExif(
            filename="test.jpg",
            timestamp=0.0,
            exposure_time=1.0,
            gain=1.0,
            black_level=0,
            bsl_color_space="sRGB",
            gamma=1,
            light_source_preset="Daylight",
            pixel_format="RGB8",
            pix_data=pix_data,
        )
        # Attach bboxes dynamically as the pipeline normally does
        photo.bboxes = bboxes
        return photo

    def test_dynamic_scaling_at_reference_altitude(self):
        """At reference altitude, the threshold is exactly min_size."""
        # reference altitude is 50m, min_size is 584px
        bboxes = [
            {"bbox": [0, 0, 600, 600], "class": 0},  # valid (>584)
            {"bbox": [0, 0, 500, 500], "class": 1},  # invalid (<584)
        ]
        photo = self._create_mock_photo(altitude=50.0, bboxes=bboxes)
        
        res = DecisionSystem.input_format(photo, min_size=584.0, ref_alt=50.0)
        coords = res[4]
        classes = res[5]
        
        assert coords.shape[0] == 1
        assert classes[0] == 0

    def test_dynamic_scaling_at_lower_altitude(self):
        """At lower altitude, the target appears larger, so threshold increases."""
        # reference altitude is 50m, min_size is 584px
        # at 25m, the threshold should be 584 * (50 / 25) = 1168px
        bboxes = [
            {"bbox": [0, 0, 1200, 1200], "class": 0},  # valid (>1168)
            {"bbox": [0, 0, 1000, 1000], "class": 1},  # invalid (<1168)
        ]
        photo = self._create_mock_photo(altitude=25.0, bboxes=bboxes)
        
        res = DecisionSystem.input_format(photo, min_size=584.0, ref_alt=50.0)
        coords = res[4]
        classes = res[5]
        
        assert coords.shape[0] == 1
        assert classes[0] == 0

    def test_dynamic_scaling_at_higher_altitude(self):
        """At higher altitude, the target appears smaller, so threshold decreases."""
        # reference altitude is 50m, min_size is 584px
        # at 100m, the threshold should be 584 * (50 / 100) = 292px
        bboxes = [
            {"bbox": [0, 0, 350, 350], "class": 0},  # valid (>292)
            {"bbox": [0, 0, 250, 250], "class": 1},  # invalid (<292)
        ]
        photo = self._create_mock_photo(altitude=100.0, bboxes=bboxes)
        
        res = DecisionSystem.input_format(photo, min_size=584.0, ref_alt=50.0)
        coords = res[4]
        classes = res[5]
        
        assert coords.shape[0] == 1
        assert classes[0] == 0

    def test_zero_altitude_fallback(self):
        """If altitude is zero, it should fallback safely without division by zero."""
        # if altitude is 0, scaling doesn't happen, it should just use min_size
        bboxes = [
            {"bbox": [0, 0, 600, 600], "class": 0},  # valid (>584)
            {"bbox": [0, 0, 500, 500], "class": 1},  # invalid (<584)
        ]
        photo = self._create_mock_photo(altitude=0.0, bboxes=bboxes)
        
        res = DecisionSystem.input_format(photo, min_size=584.0, ref_alt=50.0)
        coords = res[4]
        classes = res[5]
        
        assert coords.shape[0] == 1
        assert classes[0] == 0
