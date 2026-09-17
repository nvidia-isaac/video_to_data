from unittest.mock import patch

import numpy as np
import pytest
import torch

from learning.datasets.mhr_window_sampling import minimum_true_frame_fraction_window_mask
from learning.training.mhr_input_grid import _contact_prediction_differs
from learning.training.mhr_opt_refineout import _initial_contact_activation
from lib_mhr.hand_surface_contact import MHR_CONTACT_THRESHOLD_M, MHR_HAND_ORDER, MHR_HAND_SAMPLE_COUNT, MHR_HAND_SURFACE_CONTACT_REVISION, MHR_HAND_VERTEX_COUNT
from prep.mhr_geometry_crop import MHR_GEOMETRY_CROP_REVISION, filter_crop_masks_by_geometry
from prep.mhr_packed_h5 import validate_mhr_hand_surface_contact_metadata


def test_canonical_object_visibility_requires_ten_percent_in_every_camera():
    mask = np.zeros((2, 20), dtype=bool)
    mask[0, :2] = True
    mask[1, :1] = True
    assert not minimum_true_frame_fraction_window_mask(mask, [0], [1], 20, 0.10)[0]
    mask[1, 1] = True
    assert minimum_true_frame_fraction_window_mask(mask, [0], [1], 20, 0.10)[0]


def test_canonical_object_crop_preserves_pixels_overlapping_human():
    shape = (32, 32)
    human = np.zeros(shape, dtype=np.uint8)
    object_mask = np.zeros(shape, dtype=np.uint8)
    human[5:25, 5:20] = 255
    object_mask[12:28, 12:28] = 255
    crop_h, crop_o, diagnostics = filter_crop_masks_by_geometry(human, object_mask, human, object_mask, object_pose_valid=True, min_dilation_px=0, relative_dilation=0.0)
    assert np.any(crop_h)
    assert np.any(crop_o & (human > 127))
    assert diagnostics["object_crop_revision"] == MHR_GEOMETRY_CROP_REVISION


def test_geometry_rejection_requires_depth_tested_object_fallback():
    shape = (32, 32)
    human = np.zeros(shape, dtype=np.uint8)
    object_mask = np.zeros(shape, dtype=np.uint8)
    human[4:20, 4:16] = 255
    object_mask[1:3, 1:3] = 255
    object_silhouette = np.zeros(shape, dtype=np.uint8)
    object_silhouette[20:28, 20:28] = 255
    with pytest.raises(ValueError, match="requires a depth-tested visible object silhouette"):
        filter_crop_masks_by_geometry(human, object_mask, human, object_silhouette, object_pose_valid=True, min_dilation_px=0, relative_dilation=0.0)
    visible = object_silhouette.copy()
    _crop_h, crop_o, diagnostics = filter_crop_masks_by_geometry(human, object_mask, human, object_silhouette, object_pose_valid=True, object_visible_silhouette=visible, min_dilation_px=0, relative_dilation=0.0)
    np.testing.assert_array_equal(crop_o, visible.astype(bool))
    assert diagnostics["object_crop_support_source"] == "gt_depth_tested_object_silhouette_fallback"


def test_initial_contact_gate_requires_prediction_and_strictly_less_than_five_centimeters():
    contact_weights = torch.tensor([[1.0, 1.0], [0.0, 1.0]])
    hand_vertices = torch.zeros((2, 2, 2, 3))
    rotation = torch.eye(3)[None].expand(2, -1, -1)
    translation = torch.zeros((2, 3))
    object_vertices = torch.zeros((3, 3))
    object_faces = torch.tensor([[0, 1, 2]], dtype=torch.long)
    point_distances = torch.tensor([[[0.049, 0.060], [0.070, 0.080]], [[0.010, 0.020], [0.050, 0.060]]])
    with patch("learning.training.mhr_opt_refineout._point_triangle_surface_distances", return_value=point_distances) as distance_fn:
        effective, distances, proximity = _initial_contact_activation(contact_weights, hand_vertices, rotation, translation, object_vertices, object_faces, 0.05)
    assert distance_fn.call_count == 1
    torch.testing.assert_close(distances, torch.tensor([[0.049, 0.070], [0.010, 0.050]]))
    torch.testing.assert_close(proximity, torch.tensor([[True, False], [True, False]]))
    torch.testing.assert_close(effective, torch.tensor([[1.0, 0.0], [0.0, 0.0]]))


def test_packed_contact_schema_rejects_legacy_wrist_metadata():
    metadata = {"mhr_contact_revision": MHR_HAND_SURFACE_CONTACT_REVISION, "mhr_contact_distance_units": "meters", "mhr_contact_hand_order": list(MHR_HAND_ORDER), "mhr_contact_hand_vertex_count": [MHR_HAND_VERTEX_COUNT, MHR_HAND_VERTEX_COUNT], "mhr_contact_sample_count": MHR_HAND_SAMPLE_COUNT, "mhr_contact_threshold_m": MHR_CONTACT_THRESHOLD_M, "mhr_contact_closest_points_space": "world"}
    assert validate_mhr_hand_surface_contact_metadata(metadata, "test")["mhr_contact_revision"] == MHR_HAND_SURFACE_CONTACT_REVISION
    with pytest.raises(ValueError, match="legacy wrist-contact metadata"):
        validate_mhr_hand_surface_contact_metadata({**metadata, "mhr_contact_wrist_order": list(MHR_HAND_ORDER)}, "test")


def test_contact_header_disagreement_uses_probability_threshold():
    predictions = np.array([[0.50, 0.49], [0.49, 0.51]], dtype=np.float32)
    ground_truth = np.array([[True, False], [True, False]], dtype=bool)
    assert not _contact_prediction_differs(predictions, "probability", ground_truth, 0, True)
    assert _contact_prediction_differs(predictions, "probability", ground_truth, 1, True)
