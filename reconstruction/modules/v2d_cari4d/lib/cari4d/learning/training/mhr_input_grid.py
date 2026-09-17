from __future__ import annotations

import cv2
import numpy as np

from lib_mhr.contact import compose_object_mesh_poses
from lib_mhr.human_texture import batched_vertex_normals


GRID_TITLE_HEIGHT = 48
GRID_COLUMN_HEADER_HEIGHT = 58
GRID_LABEL_WIDTH = 224
MHR_INPUT_GRID_ROWS = (
    "Observed RGB branch\nwith source background",
    "Initialized RGB network input\nzero RGB shown black",
    "CoCoNet prediction overlay\nwith source background",
    "Ground-truth mesh overlay\nwith source background",
    "Observed CoCoNet Z\nzero XYZ shown black",
    "Initialized CoCoNet Z\nzero XYZ shown black",
    "Observed mask channels\nwith source background",
    "Initialized mask channels\nwith source background",
)
HUMAN_COLOR = np.array([52, 142, 235], dtype=np.float32)
OBJECT_COLOR = np.array([170, 86, 230], dtype=np.float32)
FULL_OBJECT_COLOR = np.array([250, 205, 70], dtype=np.float32)


def evenly_spaced_frame_indices(clip_len: int, num_frames: int) -> list[int]:
    clip_len, num_frames = int(clip_len), int(num_frames)
    if clip_len <= 0:
        raise ValueError(f"clip_len must be positive, got {clip_len}")
    if num_frames <= 0:
        raise ValueError(f"num_frames must be positive, got {num_frames}")
    count = min(clip_len, num_frames)
    return np.rint(np.linspace(0, clip_len - 1, count)).astype(np.int64).tolist()


def _validate_streams(render_rgbs: np.ndarray, input_rgbs: np.ndarray, render_xyz: np.ndarray, input_xyz: np.ndarray, background_rgbs: np.ndarray, prediction_rgbs: np.ndarray, prediction_masks: np.ndarray, ground_truth_rgbs: np.ndarray, ground_truth_masks: np.ndarray) -> tuple[int, int, int]:
    arrays = {"render_rgbs": render_rgbs, "input_rgbs": input_rgbs, "render_xyz": render_xyz, "input_xyz": input_xyz, "background_rgbs": background_rgbs, "prediction_rgbs": prediction_rgbs, "ground_truth_rgbs": ground_truth_rgbs}
    for name, value in arrays.items():
        if value.ndim != 4:
            raise ValueError(f"{name} must have shape [T,C,H,W], got {value.shape}")
    frame_count, _, height, width = input_rgbs.shape
    if input_rgbs.shape[1] != 3 or render_rgbs.shape != input_rgbs.shape or background_rgbs.shape != input_rgbs.shape or prediction_rgbs.shape != input_rgbs.shape or ground_truth_rgbs.shape != input_rgbs.shape:
        raise ValueError(f"RGB streams must share shape [T,3,H,W], got {render_rgbs.shape}, {input_rgbs.shape}, {background_rgbs.shape}, {prediction_rgbs.shape}, and {ground_truth_rgbs.shape}")
    if render_xyz.shape[0] != frame_count or input_xyz.shape[0] != frame_count or render_xyz.shape[2:] != (height, width) or input_xyz.shape[2:] != (height, width):
        raise ValueError(f"XYZ streams do not match RGB frame/spatial dimensions: {render_xyz.shape}, {input_xyz.shape}, {input_rgbs.shape}")
    if render_xyz.shape[1] < 3 or input_xyz.shape[1] < 3:
        raise ValueError(f"XYZ streams require at least three coordinate channels, got {render_xyz.shape[1]} and {input_xyz.shape[1]}")
    for name, masks in (("prediction_masks", prediction_masks), ("ground_truth_masks", ground_truth_masks)):
        if masks.shape != (frame_count, height, width) or masks.dtype != np.bool_:
            raise ValueError(f"{name} must be bool [T,H,W], got {masks.shape} {masks.dtype}")
    return frame_count, height, width


def _rgb_tiles(value: np.ndarray) -> np.ndarray:
    if value.dtype == np.uint8:
        return value.transpose(0, 2, 3, 1).copy()
    return (np.clip(value.transpose(0, 2, 3, 1), 0.0, 1.0) * 255.0).astype(np.uint8)


def _masks(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shape = (xyz.shape[0], xyz.shape[2], xyz.shape[3])
    if xyz.shape[1] == 3:
        foreground = np.linalg.norm(xyz[:, :3], axis=1) > 1e-8
        return foreground, np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    if xyz.shape[1] == 4:
        foreground = xyz[:, 3] > 0.5
        return foreground, np.zeros(shape, dtype=bool), np.zeros(shape, dtype=bool)
    human = xyz[:, 3] > 0.5
    object_mask = xyz[:, 4] > 0.5
    full_object = xyz[:, 5] > 0.5 if xyz.shape[1] > 5 else object_mask.copy()
    return human, object_mask, full_object


def _foreground_on_background(foreground_rgb: np.ndarray, background_rgb: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    output = background_rgb.copy()
    output[foreground] = foreground_rgb[foreground]
    return output


def _depth_tiles(xyz: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    z = np.asarray(xyz[:, 2], dtype=np.float32)
    valid = foreground & np.isfinite(z)
    output = np.zeros((*z.shape, 3), dtype=np.uint8)
    for index, (frame_z, frame_valid) in enumerate(zip(z, valid)):
        values = frame_z[frame_valid]
        if values.size == 0:
            continue
        lower, upper = np.percentile(values, [2.0, 98.0])
        if upper <= lower:
            lower, upper = float(values.min()) - 0.5, float(values.max()) + 0.5
        normalized = np.zeros_like(frame_z)
        normalized[frame_valid] = np.clip((frame_z[frame_valid] - lower) / (upper - lower), 0.0, 1.0)
        colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)[:, :, ::-1]
        output[index][frame_valid] = colored[frame_valid]
    return output


def _mask_overlay(rgb: np.ndarray, human: np.ndarray, object_mask: np.ndarray, full_object: np.ndarray) -> np.ndarray:
    output = (rgb.astype(np.float32) * 0.30).astype(np.uint8)
    for index in range(len(output)):
        for mask, color in ((human[index], HUMAN_COLOR), (object_mask[index], OBJECT_COLOR)):
            output[index][mask] = np.clip(rgb[index][mask].astype(np.float32) * 0.20 + color * 0.80, 0, 255).astype(np.uint8)
        if full_object[index].any():
            eroded = cv2.erode(full_object[index].astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1).astype(bool)
            output[index][full_object[index] & ~eroded] = FULL_OBJECT_COLOR.astype(np.uint8)
    return output


def overlay_mesh_tiles_on_rgb(input_rgb: np.ndarray, render_rgb: np.ndarray, render_foreground: np.ndarray) -> np.ndarray:
    input_rgb = np.asarray(input_rgb, dtype=np.uint8)
    render_rgb = np.asarray(render_rgb, dtype=np.uint8)
    render_foreground = np.asarray(render_foreground)
    if input_rgb.ndim != 4 or input_rgb.shape[-1] != 3 or render_rgb.shape != input_rgb.shape:
        raise ValueError(f"input_rgb and render_rgb must share [T,H,W,3] shape, got {input_rgb.shape} and {render_rgb.shape}")
    if render_foreground.shape != input_rgb.shape[:3] or render_foreground.dtype != np.bool_:
        raise ValueError(f"render_foreground must be bool [T,H,W], got {render_foreground.shape} {render_foreground.dtype}")
    output = (input_rgb.astype(np.float32) * 0.30).astype(np.uint8)
    for index, mask in enumerate(render_foreground):
        output[index][mask] = np.clip(input_rgb[index][mask].astype(np.float32) * 0.35 + render_rgb[index][mask].astype(np.float32) * 0.65, 0, 255).astype(np.uint8)
    return output


def _render_overlay(input_rgb: np.ndarray, render_rgb: np.ndarray, render_foreground: np.ndarray) -> np.ndarray:
    return overlay_mesh_tiles_on_rgb(input_rgb, render_rgb, render_foreground)


def _batched_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return batched_vertex_normals(vertices, faces)


def render_human_object_mesh_tiles(renderer, human_vertices: np.ndarray, human_faces: np.ndarray, object_mesh_source: str, object_mesh_to_pose_transform: np.ndarray, object_poses: np.ndarray, intrinsics: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    human_vertices = np.asarray(human_vertices, dtype=np.float32)
    human_faces = np.asarray(human_faces, dtype=np.int32)
    object_poses = np.asarray(object_poses, dtype=np.float32)
    intrinsics = np.asarray(intrinsics, dtype=np.float32)
    frame_count = human_vertices.shape[0]
    if human_vertices.ndim != 3 or human_vertices.shape[2] != 3 or not np.isfinite(human_vertices).all():
        raise ValueError(f"human_vertices must be finite [T,V,3], got {human_vertices.shape}")
    if human_faces.ndim != 2 or human_faces.shape[1] != 3 or human_faces.size == 0 or human_faces.min() < 0 or human_faces.max() >= human_vertices.shape[1]:
        raise ValueError(f"human_faces are incompatible with {human_vertices.shape[1]} vertices: {human_faces.shape}")
    if not isinstance(object_mesh_source, str) or not object_mesh_source:
        raise ValueError(f"object_mesh_source must be a nonempty path, got {object_mesh_source!r}")
    if object_poses.shape != (frame_count, 4, 4) or intrinsics.shape != (frame_count, 3, 3) or not np.isfinite(object_poses).all() or not np.isfinite(intrinsics).all():
        raise ValueError(f"object_poses and intrinsics must be finite [T,4,4]/[T,3,3], got {object_poses.shape} and {intrinsics.shape}")
    if len(image_shape) != 2 or min(map(int, image_shape)) <= 0:
        raise ValueError(f"image_shape must contain positive height and width, got {image_shape}")
    human_vertex_normals = _batched_vertex_normals(human_vertices, human_faces)
    rendered, foreground = renderer.render_front_batch_textured_human_textured_object(human_vertices, human_faces, human_vertex_normals, object_mesh_source, compose_object_mesh_poses(object_poses, object_mesh_to_pose_transform), intrinsics, image_shape)
    rendered = np.asarray(rendered, dtype=np.float32)
    foreground = np.asarray(foreground)
    expected_shape = (frame_count, int(image_shape[0]), int(image_shape[1]), 3)
    if rendered.shape != expected_shape or not np.isfinite(rendered).all():
        raise ValueError(f"mesh renderer returned invalid output {rendered.shape}, expected {expected_shape}")
    if foreground.shape != expected_shape[:3] or foreground.dtype != np.bool_:
        raise ValueError(f"mesh renderer returned invalid foreground {foreground.shape} {foreground.dtype}, expected bool {expected_shape[:3]}")
    rgb = np.rint(np.clip(rendered, 0.0, 1.0) * 255.0).astype(np.uint8).transpose(0, 3, 1, 2)
    return rgb, foreground


def render_ground_truth_mesh_tiles(renderer, human_vertices: np.ndarray, human_faces: np.ndarray, object_mesh_source: str, object_mesh_to_pose_transform: np.ndarray, object_poses: np.ndarray, intrinsics: np.ndarray, image_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    return render_human_object_mesh_tiles(renderer, human_vertices, human_faces, object_mesh_source, object_mesh_to_pose_transform, object_poses, intrinsics, image_shape)


def _put_lines(image: np.ndarray, lines: list[str], origin: tuple[int, int], color: tuple[int, int, int], scale: float = 0.42, thickness: int = 1) -> None:
    x, y = origin
    for line in lines:
        cv2.putText(image, str(line), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += max(15, int(30 * scale))


def _put_fitted_line(image: np.ndarray, text: str, origin: tuple[int, int], color: tuple[int, int, int], max_width: int, scale: float = 0.28, thickness: int = 1) -> None:
    text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]
    fitted_scale = scale if text_width <= max_width else scale * max_width / text_width
    if fitted_scale < 0.18:
        raise ValueError(f"Contact header does not fit in {max_width} pixels: {text!r}")
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, fitted_scale, color, thickness, cv2.LINE_AA)


def _validate_contact_values(prediction_contact_values: np.ndarray | None, prediction_contact_value_type: str | None, ground_truth_contacts: np.ndarray | None, frame_count: int) -> tuple[np.ndarray | None, str | None, np.ndarray | None]:
    if ground_truth_contacts is None:
        if prediction_contact_values is not None or prediction_contact_value_type is not None:
            raise ValueError("Predicted contact values require ground-truth contacts")
        return None, None, None
    ground_truth_contacts = np.asarray(ground_truth_contacts)
    if ground_truth_contacts.shape != (frame_count, 2) or ground_truth_contacts.dtype != np.bool_:
        raise ValueError(f"ground_truth_contacts must be bool [T,2] in left/right order, got {ground_truth_contacts.shape} {ground_truth_contacts.dtype}")
    if prediction_contact_values is None:
        if prediction_contact_value_type is not None:
            raise ValueError("prediction_contact_value_type requires predicted contact values")
        return None, None, ground_truth_contacts
    prediction_contact_values = np.asarray(prediction_contact_values)
    if prediction_contact_values.shape != (frame_count, 2):
        raise ValueError(f"prediction_contact_values must have shape [T,2] in left/right order, got {prediction_contact_values.shape}")
    if prediction_contact_value_type == "probability":
        if not np.issubdtype(prediction_contact_values.dtype, np.floating) or not np.isfinite(prediction_contact_values).all() or np.any((prediction_contact_values < 0.0) | (prediction_contact_values > 1.0)):
            raise ValueError("Probability contact predictions must be finite floating-point values in [0,1]")
    elif prediction_contact_value_type == "binary":
        if prediction_contact_values.dtype != np.bool_:
            raise ValueError(f"Binary contact predictions must be bool [T,2], got {prediction_contact_values.dtype}")
    else:
        raise ValueError(f"prediction_contact_value_type must be probability or binary, got {prediction_contact_value_type!r}")
    return prediction_contact_values, prediction_contact_value_type, ground_truth_contacts


def _contact_header_line(prediction_contact_values: np.ndarray | None, prediction_contact_value_type: str | None, ground_truth_contacts: np.ndarray, frame_index: int, ground_truth_valid: bool) -> str:
    ground_truth_text = f"{int(ground_truth_contacts[frame_index, 0])}/{int(ground_truth_contacts[frame_index, 1])}" if ground_truth_valid else "--/--"
    if prediction_contact_values is None:
        return f"Contact GT L/R={ground_truth_text}"
    if prediction_contact_value_type == "probability":
        left, right = prediction_contact_values[frame_index]
        return f"Contact P L/R={left:.2f}/{right:.2f} GT={ground_truth_text}"
    left, right = prediction_contact_values[frame_index]
    return f"Contact pred L/R={int(left)}/{int(right)} GT={ground_truth_text}"


def _contact_prediction_differs(prediction_contact_values: np.ndarray | None, prediction_contact_value_type: str | None, ground_truth_contacts: np.ndarray, frame_index: int, ground_truth_valid: bool) -> bool:
    if prediction_contact_values is None or not ground_truth_valid:
        return False
    if prediction_contact_value_type == "probability":
        predicted_contacts = prediction_contact_values[frame_index] >= 0.5
    elif prediction_contact_value_type == "binary":
        predicted_contacts = prediction_contact_values[frame_index]
    else:
        raise ValueError(f"prediction_contact_value_type must be probability or binary, got {prediction_contact_value_type!r}")
    return bool(np.any(predicted_contacts != ground_truth_contacts[frame_index]))


def _selected_frame_names(metadata: dict | None, frame_indices: list[int], frame_count: int) -> list[str]:
    if not metadata or metadata.get("frame_names") is None:
        return [f"frame {index}" for index in frame_indices]
    names = [str(value) for value in metadata["frame_names"]]
    if len(names) == frame_count:
        return names
    if max(frame_indices, default=-1) >= len(names):
        raise ValueError(f"frame indices {frame_indices} exceed metadata frame count {len(names)}")
    return [names[index] for index in frame_indices]


def _window_length(metadata: dict | None, frame_indices: list[int]) -> int:
    if metadata:
        for key in ("frame_names", "frame_indices"):
            values = metadata.get(key)
            if values is not None:
                return len(values)
    return max(frame_indices, default=-1) + 1


def build_mhr_input_grid(render_rgbs: np.ndarray, input_rgbs: np.ndarray, render_xyz: np.ndarray, input_xyz: np.ndarray, frame_indices: list[int], *, background_rgbs: np.ndarray, prediction_rgbs: np.ndarray, prediction_masks: np.ndarray, ground_truth_rgbs: np.ndarray, ground_truth_masks: np.ndarray, prediction_contact_values: np.ndarray | None = None, prediction_contact_value_type: str | None = None, ground_truth_contacts: np.ndarray | None = None, contact_threshold_m: float = 0.015, metadata: dict | None = None, frame_valid_mask: np.ndarray | None = None, global_step: int | None = None, split: str = "train", include_prediction: bool = True) -> tuple[np.ndarray, str]:
    frame_count, height, width = _validate_streams(render_rgbs, input_rgbs, render_xyz, input_xyz, background_rgbs, prediction_rgbs, prediction_masks, ground_truth_rgbs, ground_truth_masks)
    if len(frame_indices) != frame_count:
        raise ValueError(f"frame_indices has length {len(frame_indices)}, expected {frame_count}")
    split_label = {"train": "training", "val": "validation"}.get(str(split).lower())
    if split_label is None:
        raise ValueError(f"split must be train or val, got {split!r}")
    valid = np.ones(frame_count, dtype=bool) if frame_valid_mask is None else np.asarray(frame_valid_mask).reshape(-1) > 0.5
    if len(valid) != frame_count:
        raise ValueError(f"frame_valid_mask has length {len(valid)}, expected {frame_count}")
    prediction_contact_values, prediction_contact_value_type, ground_truth_contacts = _validate_contact_values(prediction_contact_values, prediction_contact_value_type, ground_truth_contacts, frame_count)
    if ground_truth_contacts is not None and (not np.isfinite(contact_threshold_m) or contact_threshold_m <= 0):
        raise ValueError(f"contact_threshold_m must be positive and finite, got {contact_threshold_m}")

    render_rgb, input_rgb, background_rgb, prediction_rgb, ground_truth_rgb = _rgb_tiles(render_rgbs), _rgb_tiles(input_rgbs), _rgb_tiles(background_rgbs), _rgb_tiles(prediction_rgbs), _rgb_tiles(ground_truth_rgbs)
    render_human, render_object, render_full_object = _masks(render_xyz)
    input_human, input_object, input_full_object = _masks(input_xyz)
    observed_network_xyz = np.asarray(input_xyz[:, :3], dtype=np.float32)
    initialized_network_xyz = np.asarray(render_xyz[:, :3], dtype=np.float32)
    observed_network_xyz_valid = np.isfinite(observed_network_xyz).all(axis=1) & np.any(np.abs(observed_network_xyz) > 1e-8, axis=1)
    initialized_network_xyz_valid = np.isfinite(initialized_network_xyz).all(axis=1) & np.any(np.abs(initialized_network_xyz) > 1e-8, axis=1)
    rows = [
        _foreground_on_background(input_rgb, background_rgb, input_human | input_object),
        render_rgb,
        _render_overlay(background_rgb, prediction_rgb, prediction_masks),
        _render_overlay(background_rgb, ground_truth_rgb, ground_truth_masks),
        _depth_tiles(observed_network_xyz, observed_network_xyz_valid),
        _depth_tiles(initialized_network_xyz, initialized_network_xyz_valid),
        _mask_overlay(background_rgb, input_human, input_object, input_full_object),
        _mask_overlay(background_rgb, render_human, render_object, render_full_object),
    ]
    row_labels = list(MHR_INPUT_GRID_ROWS)
    if not include_prediction:
        del rows[2]
        del row_labels[2]
    grid = np.full((GRID_TITLE_HEIGHT + GRID_COLUMN_HEADER_HEIGHT + len(rows) * height, GRID_LABEL_WIDTH + frame_count * width, 3), 22, dtype=np.uint8)
    sequence = "unknown sequence" if metadata is None else str(metadata.get("sequence_name", "unknown sequence"))
    camera = "?" if metadata is None else str(metadata.get("camera_id", "?"))
    step_text = "?" if global_step is None else f"{int(global_step):,}"
    _put_lines(grid, [f"MHR {split_label} input grid | step {step_text} | {sequence} | camera {camera}"], (10, 29), (235, 235, 235), scale=0.58, thickness=1)
    frame_names = _selected_frame_names(metadata, frame_indices, frame_count)
    for column, (frame_index, frame_name, is_valid) in enumerate(zip(frame_indices, frame_names, valid)):
        x = GRID_LABEL_WIDTH + column * width
        header_color = (78, 206, 120) if is_valid else (245, 96, 96)
        _put_lines(grid, [f"t={frame_index} valid={int(is_valid)}", frame_name], (x + 4, GRID_TITLE_HEIGHT + 20), header_color, scale=0.34, thickness=1)
        if ground_truth_contacts is not None:
            contact_color = (245, 96, 96) if _contact_prediction_differs(prediction_contact_values, prediction_contact_value_type, ground_truth_contacts, column, bool(is_valid)) else header_color
            _put_fitted_line(grid, _contact_header_line(prediction_contact_values, prediction_contact_value_type, ground_truth_contacts, column, bool(is_valid)), (x + 4, GRID_TITLE_HEIGHT + GRID_COLUMN_HEADER_HEIGHT - 6), contact_color, max_width=width - 8)
    for row_index, (label, panels) in enumerate(zip(row_labels, rows)):
        y = GRID_TITLE_HEIGHT + GRID_COLUMN_HEADER_HEIGHT + row_index * height
        _put_lines(grid, label.split("\n"), (8, y + max(16, height // 2 - 5)), (220, 220, 220), scale=0.42, thickness=1)
        for column, panel in enumerate(panels):
            x = GRID_LABEL_WIDTH + column * width
            grid[y:y + height, x:x + width] = panel
    window_length = _window_length(metadata, frame_indices)
    row_summary = "Eight rows" if include_prediction else "Seven rows without a CoCoNet prediction overlay"
    prediction_summary = "the CoCoNet-predicted human/object mesh overlay, " if include_prediction else ""
    prediction_material = "Predicted and ground-truth humans" if include_prediction else "Ground-truth humans"
    object_material = "Initialized, predicted, and ground-truth objects" if include_prediction else "Initialized and ground-truth objects"
    diagnostic_overlays = "prediction overlay, and ground-truth overlay" if include_prediction else "ground-truth overlay"
    if ground_truth_contacts is None:
        contact_summary = ""
    else:
        prediction_contact_summary = "predicted left/right contact probabilities and " if prediction_contact_value_type == "probability" else "predicted thresholded left/right contact states and " if prediction_contact_value_type == "binary" else ""
        contact_summary = f" Each unchanged-height frame header shows {prediction_contact_summary}ground-truth left/right contact states using the {contact_threshold_m * 100.0:.1f} cm hand-surface threshold; probability predictions use a 0.5 decision threshold, a red contact line marks at least one prediction/ground-truth disagreement, and GT=--/-- marks invalid ground truth."
    caption = f"MHR {split_label} input grid. {sequence}, camera {camera}: {frame_count} evenly spaced frames from the {window_length}-frame window. {row_summary} show paired observed and initialized-pose RGB branches, {prediction_summary}the ground-truth human/object mesh overlay, their exact normalized and MHR-root-joint-centered CoCoNet Z inputs, and their human/object/full-object mask channels.{contact_summary} The initialized RGB row is the exact stored render-H5 RGB tensor passed to CoCoNet, converted from [0,1] float to 8-bit RGB for display without source-background compositing; zero RGB remains black. {prediction_material} use the MHR-native skinning-part palette with normal shading; newly generated initialized RGB branches use the same material. {object_material} retain the object's original texture. The per-frame 2nd/98th-percentile Turbo color mapping, source background added to other diagnostic rows, {diagnostic_overlays} are diagnostic only and are not passed to CoCoNet."
    return grid, caption
