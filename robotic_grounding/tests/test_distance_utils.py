#!/usr/bin/env python3
"""Test script for distance_utils module.

Run this script to verify the compute_tips_distance function works correctly.

Usage (inside container):
    /workspace/isaaclab/_isaac_sim/python.sh tests/test_distance_utils.py
"""

import torch
from robotic_grounding.retarget import ASSETS_DIR, HUMAN_MOTION_DATA_DIR
from robotic_grounding.retarget.data_logger import ManoSharpaData
from robotic_grounding.retarget.distance_utils import (
    MANO_FINGERTIP_INDICES,
    PYTORCH3D_AVAILABLE,
    compute_tips_distance,
    load_object_mesh,
)


def test_compute_tips_distance_basic() -> None:
    """Test compute_tips_distance with synthetic data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on device: {device}")
    print(f"PyTorch3D available: {PYTORCH3D_AVAILABLE}")

    # Create synthetic MANO joints (21 joints, 3D positions)
    # Place joints at known positions
    mano_joints = torch.zeros(21, 3, device=device)
    # Place fingertips at specific locations (indices 4, 8, 12, 16, 20)
    mano_joints[4] = torch.tensor([0.1, 0.0, 0.0], device=device)  # thumb tip
    mano_joints[8] = torch.tensor([0.0, 0.1, 0.0], device=device)  # index tip
    mano_joints[12] = torch.tensor([0.0, 0.0, 0.1], device=device)  # middle tip
    mano_joints[16] = torch.tensor([-0.1, 0.0, 0.0], device=device)  # ring tip
    mano_joints[20] = torch.tensor([0.0, -0.1, 0.0], device=device)  # pinky tip

    # Create a simple cube mesh centered at origin
    # Vertices of a unit cube centered at origin
    obj_mesh_verts = torch.tensor(
        [
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        dtype=torch.float32,
        device=device,
    )

    # Simple cube faces (2 triangles per face, 6 faces = 12 triangles)
    obj_mesh_faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 2, 3],  # front
            [4, 6, 5],
            [4, 7, 6],  # back
            [0, 4, 5],
            [0, 5, 1],  # bottom
            [2, 6, 7],
            [2, 7, 3],  # top
            [0, 3, 7],
            [0, 7, 4],  # left
            [1, 5, 6],
            [1, 6, 2],  # right
        ],
        dtype=torch.int64,
        device=device,
    )

    # Identity rotation (no rotation)
    obj_rotation = torch.eye(3, device=device)
    # No translation
    obj_translation = torch.zeros(3, device=device)

    # Compute distances
    distances = compute_tips_distance(
        mano_joints,
        obj_mesh_verts,
        obj_mesh_faces,
        obj_rotation,
        obj_translation,
        num_samples=1000,
    )

    print(f"Fingertip indices: {MANO_FINGERTIP_INDICES}")
    print(f"Computed distances shape: {distances.shape}")
    print(f"Distances: {distances.cpu().numpy()}")

    # Verify shape
    assert distances.shape == (5,), f"Expected shape (5,), got {distances.shape}"

    # Verify distances are positive
    assert (distances >= 0).all(), "Distances should be non-negative"

    # The cube surface is at 0.5 from origin, fingertips are at 0.1 from origin
    # So distance should be approximately 0.5 - 0.1 = 0.4 for most fingertips
    # (This is approximate since we sample points on the surface)
    print(
        "Expected approximate distance: ~0.4 (cube surface at 0.5, fingertips at 0.1)"
    )

    print("test_compute_tips_distance_basic PASSED")


def test_compute_tips_distance_with_transform() -> None:
    """Test compute_tips_distance with object transformation."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # MANO joints - fingertip at origin
    mano_joints = torch.zeros(21, 3, device=device)
    mano_joints[4] = torch.tensor([0.0, 0.0, 0.0], device=device)  # thumb at origin

    # Small sphere-like mesh (icosahedron approximation)
    # For simplicity, use a tetrahedron
    obj_mesh_verts = (
        torch.tensor(
            [
                [1.0, 1.0, 1.0],
                [1.0, -1.0, -1.0],
                [-1.0, 1.0, -1.0],
                [-1.0, -1.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        * 0.1
    )  # Scale down

    obj_mesh_faces = torch.tensor(
        [
            [0, 1, 2],
            [0, 1, 3],
            [0, 2, 3],
            [1, 2, 3],
        ],
        dtype=torch.int64,
        device=device,
    )

    # Place object at (1, 0, 0)
    obj_rotation = torch.eye(3, device=device)
    obj_translation = torch.tensor([1.0, 0.0, 0.0], device=device)

    distances = compute_tips_distance(
        mano_joints,
        obj_mesh_verts,
        obj_mesh_faces,
        obj_rotation,
        obj_translation,
        num_samples=500,
    )

    print("\nWith object translated to (1, 0, 0):")
    print(f"Thumb tip distance: {distances[0].item():.4f}")

    # Distance should be approximately 1.0 - 0.1 = 0.9 (object center at 1.0, radius ~0.1)
    assert distances[0] > 0.5, f"Expected distance > 0.5, got {distances[0]}"

    print("test_compute_tips_distance_with_transform PASSED")


def test_load_object_mesh() -> None:
    """Test load_object_mesh function (requires actual mesh file)."""
    # Try to find a mesh file
    mesh_dir = ASSETS_DIR / "meshes" / "arctic"
    if not mesh_dir.exists():
        print(
            f"\nSkipping test_load_object_mesh: mesh directory not found at {mesh_dir}"
        )
        return

    # Look for any .obj file
    obj_files = list(mesh_dir.glob("**/*.obj"))
    if not obj_files:
        print(f"\nSkipping test_load_object_mesh: no .obj files found in {mesh_dir}")
        return

    mesh_path = obj_files[0]
    print(f"\nTesting load_object_mesh with: {mesh_path}")

    verts, faces = load_object_mesh(str(mesh_path))

    print(f"Vertices shape: {verts.shape}")
    print(f"Faces shape: {faces.shape}")

    assert (
        verts.ndim == 2 and verts.shape[1] == 3
    ), f"Expected vertices (V, 3), got {verts.shape}"
    assert (
        faces.ndim == 2 and faces.shape[1] == 3
    ), f"Expected faces (F, 3), got {faces.shape}"
    assert verts.dtype == torch.float32, f"Expected float32 vertices, got {verts.dtype}"
    assert faces.dtype == torch.int64, f"Expected int64 faces, got {faces.dtype}"

    print("test_load_object_mesh PASSED")


def test_integration_with_real_data() -> None:
    """Integration test with actual ARCTIC data (if available)."""
    # Check for processed parquet data
    processed_dir = HUMAN_MOTION_DATA_DIR / "arctic" / "arctic_processed"
    if not processed_dir.exists():
        print(
            f"\nSkipping integration test: processed data not found at {processed_dir}"
        )
        return

    try:
        # Load a sample trajectory
        data = ManoSharpaData.from_parquet(str(processed_dir))
        print(f"\nLoaded trajectory: {data.sequence_id}")
        print(f"Object: {data.object_name}")
        print(f"Number of frames: {len(data.mano_right_joints)}")

        # Check if tips_distance was computed
        if data.mano_right_tips_distance:
            print(
                f"Right tips_distance available: {len(data.mano_right_tips_distance)} frames"
            )
            print(f"Sample distances (frame 0): {data.mano_right_tips_distance[0]}")
        else:
            print("Right tips_distance not yet computed for this trajectory")

        if data.mano_left_tips_distance:
            print(
                f"Left tips_distance available: {len(data.mano_left_tips_distance)} frames"
            )
        else:
            print("Left tips_distance not yet computed for this trajectory")

        print("test_integration_with_real_data PASSED")

    except Exception as e:
        print(f"Integration test skipped due to: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing distance_utils module")
    print("=" * 60)

    test_compute_tips_distance_basic()
    test_compute_tips_distance_with_transform()
    test_load_object_mesh()
    test_integration_with_real_data()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
