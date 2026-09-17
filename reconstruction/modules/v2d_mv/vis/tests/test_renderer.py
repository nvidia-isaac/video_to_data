import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pyrender")

VIS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIS_DIR))

import renderer


class FakeNode:
    def __init__(self, payload=None, **kwargs):
        self.payload = payload
        self.kwargs = kwargs


class FakeMesh:
    @staticmethod
    def from_trimesh(mesh):
        return ("mesh", mesh)


class FakeScene:
    def __init__(self, *args, **kwargs):
        self.nodes = []
        self.removed_nodes = []
        self.poses = {}

    def add(self, payload, pose=None):
        node = FakeNode(payload)
        self.nodes.append(node)
        self.poses[node] = pose
        return node

    def add_node(self, node, parent_node=None):
        self.nodes.append(node)
        return node

    def remove_node(self, node):
        self.removed_nodes.append(node)
        if node in self.nodes:
            self.nodes.remove(node)

    def set_pose(self, node, pose):
        self.poses[node] = pose


class FakeOffscreenRenderer:
    def __init__(self, viewport_width, viewport_height):
        self.viewport_width = viewport_width
        self.viewport_height = viewport_height
        self.deleted = False

    def render(self, scene, flags=None):
        color = np.zeros((self.viewport_height, self.viewport_width, 4), dtype=np.uint8)
        return color, None

    def delete(self):
        self.deleted = True


@pytest.fixture
def fake_pyrender(monkeypatch):
    monkeypatch.setattr(renderer.pyrender, "OffscreenRenderer", FakeOffscreenRenderer)
    monkeypatch.setattr(renderer.pyrender, "Scene", FakeScene)
    monkeypatch.setattr(renderer.pyrender, "Mesh", FakeMesh)
    monkeypatch.setattr(renderer.pyrender, "IntrinsicsCamera", lambda **kwargs: ("camera", kwargs))
    monkeypatch.setattr(renderer.pyrender, "DirectionalLight", lambda **kwargs: ("light", kwargs))
    monkeypatch.setattr(renderer.pyrender, "Node", FakeNode)


def test_persistent_mesh_lifecycle(fake_pyrender):
    r = renderer.Renderer(image_size=(4, 3))
    pose = np.eye(4)

    handle = r.add_persistent_mesh(object(), pose=pose)
    node = r._persistent_mesh_nodes[handle]
    assert np.array_equal(r._scene.poses[node], pose)

    updated_pose = np.eye(4)
    updated_pose[0, 3] = 2.0
    r.set_persistent_mesh_pose(handle, updated_pose)
    assert np.array_equal(r._scene.poses[node], updated_pose)

    r.remove_persistent_mesh(handle)
    assert handle not in r._persistent_mesh_nodes
    assert node in r._scene.removed_nodes

    with pytest.raises(KeyError):
        r.set_persistent_mesh_pose(handle, pose)

    handle_a = r.add_persistent_mesh(object())
    handle_b = r.add_persistent_mesh(object())
    node_a = r._persistent_mesh_nodes[handle_a]
    node_b = r._persistent_mesh_nodes[handle_b]
    r.clear_persistent_meshes()
    assert r._persistent_mesh_nodes == {}
    assert node_a in r._scene.removed_nodes
    assert node_b in r._scene.removed_nodes


def test_render_overlay_keeps_persistent_meshes_and_replaces_dynamic_meshes(fake_pyrender):
    r = renderer.Renderer(image_size=(4, 3))
    persistent_handle = r.add_persistent_mesh(object())
    persistent_node = r._persistent_mesh_nodes[persistent_handle]
    K = np.eye(3)
    T = np.eye(4)
    image = np.zeros((3, 4, 3), dtype=np.uint8)

    r.render_overlay(meshes=[object()], K=K, T=T, image=image)
    first_dynamic_node = r._mesh_nodes[0]
    assert persistent_node in r._scene.nodes

    r.render_overlay(meshes=[object()], K=K, T=T, image=image)
    assert first_dynamic_node in r._scene.removed_nodes
    assert persistent_node in r._scene.nodes
    assert r._persistent_mesh_nodes[persistent_handle] is persistent_node
