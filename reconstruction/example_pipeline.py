from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d.moge.docker.run_video_to_depth import run_video_to_depth
from v2d.foundation_pose.docker.run_simplify_mesh import run_simplify_mesh
from v2d.foundation_pose.docker.run_transform_mesh import run_transform_mesh
from v2d.foundation_pose.docker.run_align_mesh_scale import run_align_mesh_scale
from v2d.foundation_pose.docker.run_video_to_poses import run_video_to_poses
from v2d.foundation_pose.docker.run_render_overlay import run_render_overlay


run_image_to_mesh(
    image_path="modules/v2d_sam3d/assets/test_image.jpg",
    mask_path="modules/v2d_sam3d/assets/mask_1.png",
    mesh_path="data/outputs/sam3d/mesh_1.glb",
    transform_path="data/outputs/sam3d/transform_1.json",
    intrinsics_path="data/outputs/sam3d/intrinsics_1.json",
    weights_dir="data/weights/sam3d",
    dev=False,
)

run_video_to_depth(
    video_path="modules/v2d_moge/assets/test_video.mp4",
    depth_folder="data/outputs/moge/depth",
    intrinsics_folder="data/outputs/moge/intrinsics",
    weights_path="data/weights/moge",
    dev=False,
)


run_simplify_mesh(
    input_mesh="data/outputs/sam3d/mesh_1.glb",
    output_mesh="data/outputs/sam3d/mesh_1_simplified.glb",
    factor=0.1,
    dev=False,
)

run_align_mesh_scale(
    mesh_path="data/outputs/sam3d/mesh_1_simplified.glb",
    depth_path="data/outputs/moge/depth/000000.png",
    mask_path="data/outputs/sam2/masks/1/000000.png",
    intrinsics_path="data/outputs/sam3d/intrinsics_1.json",
    transform_path="data/outputs/sam3d/transform_1.json",
    output_transform_path="data/outputs/sam3d/transform_1_aligned.json",
    dev=False,
)

run_transform_mesh(
    input_mesh="data/outputs/sam3d/mesh_1_simplified.glb",
    output_mesh="data/outputs/sam3d/mesh_1_transformed.glb",
    transform_path="data/outputs/sam3d/transform_1_aligned.json",
    dev=False,
)

run_video_to_poses(
    video_path="modules/v2d_sam2/assets/test_video.mp4",
    depth_folder="data/outputs/moge/depth",
    masks_folder="data/outputs/sam2/masks/1",
    camera_intrinsics_path="data/outputs/moge/intrinsics/000000.json",
    mesh_path="data/outputs/sam3d/mesh_1_transformed.glb",
    poses_dir="data/outputs/foundation_pose/poses",
    weights_dir="data/weights/foundation_pose",
    debug_dir="data/outputs/foundation_pose/debug",
    dev=False,
)

run_render_overlay(
    video_path="modules/v2d_sam2/assets/test_video.mp4",
    poses_dir="data/outputs/foundation_pose/poses",
    mesh_path="data/outputs/sam3d/mesh_1_transformed.glb",
    camera_intrinsics_path="data/outputs/moge/intrinsics/000000.json",
    output_dir="data/outputs/foundation_pose/overlay",
    dev=False,
)