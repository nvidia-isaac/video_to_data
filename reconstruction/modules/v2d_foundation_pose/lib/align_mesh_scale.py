import cv2
import trimesh
import json
import numpy as np
import argparse
import os
from v2d.datatypes import DepthImage, Mask

def wxyz_quat_to_rotation_matrix(w, x, y, z):
    rotation_matrix = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])
    return rotation_matrix

def align_mesh_scale(
    mesh_path: str,
    depth_path: str,
    mask_path: str,
    intrinsics_path: str,
    transform_path: str,
    output_transform_path: str = None
):
    with open(transform_path, "r") as f:
        transform = json.load(f)
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)

    depth_data = DepthImage.load(depth_path).depth
    mask_data = Mask.load(mask_path).mask > 0

    scene = trimesh.load(mesh_path, process=False)
    mesh = next(value for name, value in scene.geometry.items())

    object_translation = np.array(transform['translation'])
    object_rotation = np.array(transform['rotation'])
    object_scale = np.array(transform['scale'])

    flip_matrix = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T
    rotation_transform_matrix = np.eye(4)
    rotation_transform_matrix[:3, :3] = wxyz_quat_to_rotation_matrix(object_rotation[0], object_rotation[1], object_rotation[2], object_rotation[3])
    rotation_transform_matrix = flip_matrix @ rotation_transform_matrix

    vertices = np.array(mesh.vertices)

    vertices_scaled = vertices * object_scale

    vertices_homo = np.concatenate([vertices_scaled, np.ones((vertices_scaled.shape[0], 1))], axis=1)
    vertices_rotated = (vertices_homo @ rotation_transform_matrix)[:, :3]

    vertices_final = vertices_rotated + object_translation
    vertices_final[:, 0] = -vertices_final[:, 0]
    vertices_final[:, 1] = -vertices_final[:, 1]

    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    u = vertices_final[:, 0]
    v = vertices_final[:, 1]
    w = vertices_final[:, 2]

    valid_w = w > 0.1
    u, v, w = u[valid_w], v[valid_w], w[valid_w]

    x_img = (u / w) * fx + cx
    y_img = (v / w) * fy + cy

    h, w_img = depth_data.shape
    valid_img = (x_img >= 0) & (x_img < w_img) & (y_img >= 0) & (y_img < h)

    x_idx = x_img[valid_img].astype(int)
    y_idx = y_img[valid_img].astype(int)

    mask_hits = mask_data[y_idx, x_idx]

    if not np.any(mask_hits):
        print("Warning: No vertices project into the mask. Scale refinement might be inaccurate.")
        depth_samples = depth_data[y_idx, x_idx]
        mesh_z_samples = w[valid_img]
    else:
        depth_samples = depth_data[y_idx[mask_hits], x_idx[mask_hits]]
        mesh_z_samples = w[valid_img][mask_hits]

    valid_depth = depth_samples > 0.1
    if not np.any(valid_depth):
        print("Error: No valid depth values found in mask area.")
        return object_scale

    depth_samples = depth_samples[valid_depth]
    mesh_z_samples = mesh_z_samples[valid_depth]

    scale_factor = np.median(depth_samples / mesh_z_samples)

    refined_scale = object_scale * scale_factor
    refined_translation = object_translation * scale_factor

    print(f"Original scale: {object_scale}")
    print(f"Original translation: {object_translation}")
    print(f"Scale factor: {scale_factor:.4f}")
    print(f"Refined scale: {refined_scale}")
    print(f"Refined translation: {refined_translation}")

    if output_transform_path:
        refined_transform = transform.copy()
        refined_transform['scale'] = refined_scale.tolist()
        refined_transform['translation'] = refined_translation.tolist()
        with open(output_transform_path, "w") as f:
            json.dump(refined_transform, f, indent=4)
        print(f"Saved refined transform to {output_transform_path}")

        base_path = os.path.splitext(output_transform_path)[0]

        img_path = depth_path.replace("depth", "color").replace(".png", ".jpg")
        if os.path.exists(img_path):
            base_vis_img = cv2.imread(img_path)
        else:
            base_vis_img = np.zeros((h, w_img, 3), dtype=np.uint8)

        def create_debug_render(scale_to_use, translation_to_use, label):
            vis_img = base_vis_img.copy()
            mask_vis = np.zeros_like(vis_img)
            mask_vis[mask_data] = [0, 255, 0]
            vis_img = cv2.addWeighted(vis_img, 0.7, mask_vis, 0.3, 0)

            v_scaled = vertices * scale_to_use
            v_homo = np.concatenate([v_scaled, np.ones((v_scaled.shape[0], 1))], axis=1)
            v_final = (v_homo @ rotation_transform_matrix)[:, :3] + translation_to_use
            v_final[:, 0] = -v_final[:, 0]
            v_final[:, 1] = -v_final[:, 1]

            u_p = v_final[:, 0]
            v_p = v_final[:, 1]
            w_p = v_final[:, 2]
            valid_w_p = w_p > 0.1

            x_i = (u_p[valid_w_p] / w_p[valid_w_p]) * fx + cx
            y_i = (v_p[valid_w_p] / w_p[valid_w_p]) * fy + cy

            for i in range(0, len(x_i), max(1, len(x_i) // 2000)):
                px, py = int(x_i[i]), int(y_i[i])
                if 0 <= px < w_img and 0 <= py < h:
                    cv2.circle(vis_img, (px, py), 1, (255, 0, 0), -1)

            cv2.putText(vis_img, f"Scale: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            return vis_img

        orig_vis = create_debug_render(object_scale, object_translation, "Original")
        orig_vis_path = f"{base_path}_original.jpg"
        cv2.imwrite(orig_vis_path, orig_vis)
        print(f"Saved original scale debug visualization to {orig_vis_path}")

        refined_vis = create_debug_render(refined_scale, refined_translation, "Refined")
        refined_vis_path = f"{base_path}_refined.jpg"
        cv2.imwrite(refined_vis_path, refined_vis)
        print(f"Saved refined scale debug visualization to {refined_vis_path}")

    return refined_scale.tolist()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align mesh scale to a specific depth image")
    parser.add_argument("--mesh", required=True, help="Path to mesh GLB")
    parser.add_argument("--depth", required=True, help="Path to depth image")
    parser.add_argument("--mask", required=True, help="Path to object mask")
    parser.add_argument("--intrinsics", required=True, help="Path to camera intrinsics JSON")
    parser.add_argument("--transform", required=True, help="Path to original transform JSON")
    parser.add_argument("--output-transform", required=True, help="Path to save refined transform JSON")

    args = parser.parse_args()

    align_mesh_scale(
        args.mesh,
        args.depth,
        args.mask,
        args.intrinsics,
        args.transform,
        args.output_transform
    )
