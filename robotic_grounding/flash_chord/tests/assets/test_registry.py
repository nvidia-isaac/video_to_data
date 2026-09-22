# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for asset path resolution."""

from flash_chord.assets import ASSETS_DIR
from flash_chord.assets.registry import resolve_asset_path, resolve_object_asset_specs, support_usda_for_parquet


def test_resolve_reroots_on_assets_boundary():
    out = resolve_asset_path("/any/source/prefix/assets/meshes/arctic/box/bottom.obj")
    assert out == str(ASSETS_DIR / "meshes/arctic/box/bottom.obj")


def test_resolve_is_idempotent_for_local_paths():
    local = str(ASSETS_DIR / "meshes/arctic/box/bottom.obj")
    assert resolve_asset_path(local) == local


def test_resolve_uses_last_assets_segment():
    out = resolve_asset_path("/a/assets/b/assets/meshes/m.obj")
    assert out == str(ASSETS_DIR / "meshes/m.obj")


def test_resolve_dataset_container_object_assets_root():
    out = resolve_asset_path("/data/object_assets/meshes/example.obj")
    assert out == str(ASSETS_DIR / "meshes/example.obj")


def test_resolve_no_assets_segment_returns_unchanged():
    assert resolve_asset_path("/other/x.obj") == "/other/x.obj"


def test_resolve_prefers_materialized_reference_asset_tree(tmp_path):
    assets = tmp_path / "repo/robotic_grounding/assets"
    relative = "human_motion_data/ego_recon/processed/object.urdf"
    actual = assets / relative
    actual.parent.mkdir(parents=True)
    actual.write_text("<robot/>")
    reference = (
        assets / "human_motion_data/ego_recon/processed" / "sequence_id=clip/robot_name=vega_sharpa/data.parquet"
    )

    resolved = resolve_asset_path(
        f"/producer/video_to_data/robotic_grounding/assets/{relative}",
        reference_path=reference,
    )

    assert resolved == str(actual)


def test_support_resolution_prefers_materialized_robot_specific_surface(tmp_path):
    dataset = tmp_path / "assets/human_motion_data/ego_recon"
    reference = dataset / "processed/sequence_id=clip/robot_name=vega_sharpa/data.parquet"
    stage = dataset / "reconstructed_stage"
    stage.mkdir(parents=True)
    generic = stage / "clip_support.usda"
    specific = stage / "clip_vega_sharpa_support.usda"
    generic.write_text("#usda 1.0")
    specific.write_text("#usda 1.0")

    assert support_usda_for_parquet(str(reference)) == specific
    assert support_usda_for_parquet(str(reference), robot_name="sharpa_wave") == generic


def test_resolve_v2d_articulated_object_layout_beside_reference(tmp_path):
    dataset = tmp_path / "assets/human_motion_data/synthbox"
    urdf = dataset / "object_assets/urdfs/synthbox/box.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text("<robot/>")
    reference = dataset / "synthbox_processed/sequence_id=clip/robot_name=sharpa_wave/data.parquet"

    assets = resolve_object_asset_specs(
        source_dataset="synthbox",
        object_name="box",
        body_names=["bottom", "top"],
        mesh_paths=[
            "object_assets/meshes/synthbox/box/bottom.obj",
            "object_assets/meshes/synthbox/box/top.obj",
        ],
        urdf_paths=[],
        num_articulations=1,
        reference_path=reference,
    )

    assert len(assets) == 1
    assert assets[0].urdf_path == str(urdf)
    assert [body.reference_name for body in assets[0].bodies] == ["bottom", "top"]
    assert assets[0].root_reference_name == "bottom"
    assert assets[0].articulations[0].simulation_joint_name == "rotation"
