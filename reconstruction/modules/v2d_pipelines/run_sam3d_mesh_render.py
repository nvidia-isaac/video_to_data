# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
"""
Pipeline: SAM3D image-to-mesh, then render depth, mask, and image.
Run from reconstruction/: python -m v2d.pipelines.run_sam3d_mesh_render

Steps:
  1. SAM3D  : image + mask → mesh + transform + intrinsics
  2. Transform mesh into camera space
  3. Render depth image of the transformed mesh
  4. Render silhouette mask of the transformed mesh
  5. Render RGB image of the transformed mesh
"""

from v2d.sam3d.docker.run_image_to_mesh import run_image_to_mesh
from v2d.mesh.docker.run_mesh_transform import run_mesh_transform
from v2d.mesh.docker.run_mesh_render_depth import run_mesh_render_depth
from v2d.mesh.docker.run_mesh_render_mask import run_mesh_render_mask
from v2d.mesh.docker.run_mesh_render_image import run_mesh_render_image


def main():
    # -------------------------------------------------------------------------
    # Step 1: SAM3D — reconstruct 3D mesh from image + object mask
    # -------------------------------------------------------------------------
    run_image_to_mesh(
        image_path="modules/v2d_sam3d/assets/test_image.jpg",
        mask_path="modules/v2d_sam3d/assets/mask_1.png",
        mesh_path="data/outputs/sam3d_mesh_render/mesh.glb",
        transform_path="data/outputs/sam3d_mesh_render/transform.json",
        intrinsics_path="data/outputs/sam3d_mesh_render/intrinsics.json",
        weights_dir="data/weights/sam3d",
    )

    # -------------------------------------------------------------------------
    # Step 2: Transform mesh — apply SAM3D transform to bring mesh into
    #         OpenCV camera space (Z forward, Y down)
    # -------------------------------------------------------------------------
    run_mesh_transform(
        input_mesh_path="data/outputs/sam3d_mesh_render/mesh.glb",
        transform_path="data/outputs/sam3d_mesh_render/transform.json",
        output_mesh_path="data/outputs/sam3d_mesh_render/mesh_camera.glb",
    )

    # -------------------------------------------------------------------------
    # Step 3: Render depth — uint16 PNG, inverse-depth encoded
    # -------------------------------------------------------------------------
    run_mesh_render_depth(
        mesh_path="data/outputs/sam3d_mesh_render/mesh_camera.glb",
        intrinsics_path="data/outputs/sam3d_mesh_render/intrinsics.json",
        output_depth_path="data/outputs/sam3d_mesh_render/depth.png",
    )

    # -------------------------------------------------------------------------
    # Step 4: Render mask — grayscale PNG, white where mesh is visible
    # -------------------------------------------------------------------------
    run_mesh_render_mask(
        mesh_path="data/outputs/sam3d_mesh_render/mesh_camera.glb",
        intrinsics_path="data/outputs/sam3d_mesh_render/intrinsics.json",
        output_mask_path="data/outputs/sam3d_mesh_render/mask.png",
    )

    # -------------------------------------------------------------------------
    # Step 5: Render image — RGB PNG with vertex colors
    # -------------------------------------------------------------------------
    run_mesh_render_image(
        mesh_path="data/outputs/sam3d_mesh_render/mesh_camera.glb",
        intrinsics_path="data/outputs/sam3d_mesh_render/intrinsics.json",
        output_image_path="data/outputs/sam3d_mesh_render/image.png",
    )


if __name__ == "__main__":
    main()
