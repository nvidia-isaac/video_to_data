
import cv2
import trimesh
import json
import numpy as np

def wxyz_quat_to_rotation_matrix(w, x, y, z):
    rotation_matrix = np.array([[1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
                                [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
                                [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]])
    return rotation_matrix

def render_debug_image(
        image_path: str,
        mesh_path: str,
        transform_path: str,
        intrinsics_path: str,
        output_image_path: str,
        num_vertices_to_use: int = 5000
    ):

    with open(transform_path, "r") as f:
        transform = json.load(f)
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)

    # load glb
    scene = trimesh.load(mesh_path, process=False)

    mesh = next(value for name, value in scene.geometry.items())

    # load camera intrinsics
    camera_intrinsics = intrinsics

    # load object pose and scale
    object_translation = np.array(transform['translation'])  
    object_rotation = np.array(transform['rotation'])
    object_scale = np.array(transform['scale'])

    flip_matrix = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T

    # creat 90 degree rotation matrix around x axis
    rotation_transform_matrix = np.eye(4)
    rotation_transform_matrix[:3, :3] = wxyz_quat_to_rotation_matrix(object_rotation[0], object_rotation[1], object_rotation[2], object_rotation[3])
    rotation_transform_matrix = flip_matrix @ rotation_transform_matrix

    vertices_3d = np.array(mesh.vertices)
    vertices_3d = np.concatenate([vertices_3d, np.ones((vertices_3d.shape[0], 1))], axis=1)
    vertices_3d = vertices_3d * np.concatenate([object_scale, [1]])
    vertices_3d = vertices_3d @ rotation_transform_matrix
    
    vertices_3d = vertices_3d + np.concatenate([object_translation, [0]])
    vertices_3d[:, 0] = -vertices_3d[:, 0]
    vertices_3d[:, 1] = -vertices_3d[:, 1]
    
    vertices_2d = vertices_3d

    u = vertices_3d[:, 0]
    v = vertices_3d[:, 1]
    w = vertices_3d[:, 2]
    x = u / w
    y = v / w

    x_image = x * camera_intrinsics['fx'] + camera_intrinsics['cx']
    y_image = y * camera_intrinsics['fy'] + camera_intrinsics['cy']
    z_image = w

    vertices_2d = np.stack([x_image, y_image, z_image], axis=1)
    vertices_2d = vertices_2d[:, :2]

    # # Render the vertices to the image
    image = cv2.imread(image_path)
    for vertex in vertices_2d[np.random.permutation(len(vertices_2d))[:num_vertices_to_use]]:
        cv2.circle(image, (int(vertex[0]), int(vertex[1])), 1, (255, 0, 0), -1)

    cv2.imwrite(output_image_path, image)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Render a debug image of a mesh")
    parser.add_argument("image_path", type=str, help="Path to the image")
    parser.add_argument("mesh_path", type=str, help="Path to the mesh")
    parser.add_argument("transform_path", type=str, help="Path to the transform")
    parser.add_argument("intrinsics_path", type=str, help="Path to the intrinsics")
    parser.add_argument("output_image_path", type=str, help="Path to the output image")
    parser.add_argument("--num_vertices_to_use", type=int, default=5000, help="Number of vertices to use")
    args = parser.parse_args()
    render_debug_image(args.image_path, args.mesh_path, args.transform_path, args.intrinsics_path, args.output_image_path, args.num_vertices_to_use)