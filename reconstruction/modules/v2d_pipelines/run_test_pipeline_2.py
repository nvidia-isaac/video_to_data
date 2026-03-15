from v2d.mesh.lib.run_mesh_render_mask import run_mesh_render_mask
frok v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh


run_image_to_mesh(
    image_path="modules/v2d_sam3d/assets/test_image.jpg",
    mask_path="modules/v2d_sam3d/assets/mask_1.png",
    mesh_path="data/outputs/sam3d/mesh_1.glb",
    transform_path="data/outputs/sam3d/transform_1.json",
    intrinsics_path="data/outputs/sam3d/intrinsics_1.json",
    weights_dir="data/weights/sam3d",
    dev=False,
)