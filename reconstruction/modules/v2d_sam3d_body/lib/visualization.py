import numpy as np
import cv2
import trimesh

from .renderer import Renderer
from sam_3d_body.visualization.skeleton_visualizer import SkeletonVisualizer
from sam_3d_body.metadata.mhr70 import pose_info as mhr70_pose_info


LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
SIDE_VIEW_POSE = trimesh.transformations.rotation_matrix(np.radians(-90), [0, 1, 0])

visualizer = SkeletonVisualizer(line_width=2, radius=5)
visualizer.set_pose_meta(mhr70_pose_info)


def _cam_t_to_pose(cam_t: np.ndarray) -> np.ndarray:
    """Build a camera pose from a SMPL-style translation vector.

    The SMPL pipeline outputs pred_cam_t with X negated relative to the
    standard CV camera convention, so we flip it here.
    """
    pose = np.eye(4, dtype=np.float64)
    pose[0, 3] = -cam_t[0]
    pose[1, 3] = cam_t[1]
    pose[2, 3] = cam_t[2]
    return pose


def visualize_mesh(img_cv2, outputs, renderer: Renderer):
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        camera_pose = _cam_t_to_pose(person_output["pred_cam_t"])
        img = (
            renderer(
                person_output["pred_vertices"],
                camera_pose,
                img_mesh.copy(),
                mesh_base_color_rgb=LIGHT_BLUE,
                scene_bg_color_rgb=(1, 1, 1),
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img



def visualize_sample(img_cv2, outputs, renderer: Renderer):
    img_keypoints = img_cv2.copy()
    img_mesh = img_cv2.copy()

    rend_img = []
    for pid, person_output in enumerate(outputs):
        keypoints_2d = person_output["pred_keypoints_2d"]
        keypoints_2d = np.concatenate(
            [keypoints_2d, np.ones((keypoints_2d.shape[0], 1))], axis=-1
        )
        img1 = visualizer.draw_skeleton(img_keypoints.copy(), keypoints_2d)

        img1 = cv2.rectangle(
            img1,
            (int(person_output["bbox"][0]), int(person_output["bbox"][1])),
            (int(person_output["bbox"][2]), int(person_output["bbox"][3])),
            (0, 255, 0),
            2,
        )

        if "lhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["lhand_bbox"][0]),
                    int(person_output["lhand_bbox"][1]),
                ),
                (
                    int(person_output["lhand_bbox"][2]),
                    int(person_output["lhand_bbox"][3]),
                ),
                (255, 0, 0),
                2,
            )

        if "rhand_bbox" in person_output:
            img1 = cv2.rectangle(
                img1,
                (
                    int(person_output["rhand_bbox"][0]),
                    int(person_output["rhand_bbox"][1]),
                ),
                (
                    int(person_output["rhand_bbox"][2]),
                    int(person_output["rhand_bbox"][3]),
                ),
                (0, 0, 255),
                2,
            )

        camera_pose = _cam_t_to_pose(person_output["pred_cam_t"])
        img2 = (
            renderer(
                person_output["pred_vertices"],
                camera_pose,
                img_mesh.copy(),
                mesh_base_color_rgb=LIGHT_BLUE,
                scene_bg_color_rgb=(1, 1, 1),
            )
            * 255
        )

        white_img = np.ones_like(img_cv2) * 255
        side_pose = SIDE_VIEW_POSE @ camera_pose
        img3 = (
            renderer(
                person_output["pred_vertices"],
                side_pose,
                white_img,
                mesh_base_color_rgb=LIGHT_BLUE,
                scene_bg_color_rgb=(1, 1, 1),
            )
            * 255
        )

        cur_img = np.concatenate([img_cv2, img1, img2, img3], axis=1)
        rend_img.append(cur_img)

    return rend_img
