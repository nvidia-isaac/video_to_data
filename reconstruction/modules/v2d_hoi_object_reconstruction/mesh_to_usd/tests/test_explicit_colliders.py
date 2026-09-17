import sys
from types import SimpleNamespace

import pytest
import trimesh

import explicit_colliders
from explicit_colliders import (
    MIN_COLLIDER_VOLUME_M3,
    build_explicit_colliders,
    collider_set_to_report,
)


def _install_fake_coacd(monkeypatch, decomposition):
    fake_coacd = SimpleNamespace(
        Mesh=lambda vertices, faces: (vertices, faces),
        run_coacd=lambda *_args, **_kwargs: decomposition,
    )
    monkeypatch.setitem(sys.modules, "coacd", fake_coacd)
    monkeypatch.setattr(explicit_colliders, "package_version", lambda _name: "test")


def test_build_explicit_colliders_filters_numerical_volume_sliver(monkeypatch):
    source = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    retained = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    sliver = trimesh.creation.box(extents=(9e-6, 9e-6, 9e-6))
    assert 0.0 < sliver.volume <= MIN_COLLIDER_VOLUME_M3
    _install_fake_coacd(
        monkeypatch,
        (
            (retained.vertices, retained.faces),
            (sliver.vertices, sliver.faces),
        ),
    )

    colliders = build_explicit_colliders(
        source.vertices,
        source.faces,
        "convexDecomposition",
    )
    report = collider_set_to_report(colliders)

    assert len(colliders.parts) == 1
    assert report["part_filter"] == {
        "minimum_retained_volume_m3": MIN_COLLIDER_VOLUME_M3,
        "input_part_count": 2,
        "retained_part_count": 1,
        "discarded_part_count": 1,
        "discarded_parts": [
            {
                "source_index": 1,
                "reason": "volume_at_or_below_minimum",
                "vertex_count": 8,
                "face_count": 12,
                "volume_m3": pytest.approx(sliver.volume),
                "extents_m": pytest.approx(tuple(sliver.extents)),
            }
        ],
    }


def test_build_explicit_colliders_fails_if_every_part_is_negligible(monkeypatch):
    source = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    sliver = trimesh.creation.box(extents=(9e-6, 9e-6, 9e-6))
    _install_fake_coacd(monkeypatch, ((sliver.vertices, sliver.faces),))

    with pytest.raises(RuntimeError, match="no collider parts above"):
        build_explicit_colliders(
            source.vertices,
            source.faces,
            "convexDecomposition",
        )
