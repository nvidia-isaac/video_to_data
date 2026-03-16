import shutil

from v2d.mesh.docker.run_mesh_simplify import run_mesh_simplify

from .conftest import is_glb


def test_simplify_by_factor(output_dir, mesh):
    out = str(output_dir / "simplified.glb")
    run_mesh_simplify(mesh, out, factor=0.5)
    assert is_glb(out)


def test_simplify_by_face_count(output_dir, mesh):
    out = str(output_dir / "simplified.glb")
    run_mesh_simplify(mesh, out, face_count=500)
    assert is_glb(out)


def test_simplify_broadcast(output_dir, tmp_path, mesh):
    meshes_dir = tmp_path / "meshes"
    meshes_dir.mkdir()
    shutil.copy(mesh, meshes_dir / "a.glb")
    shutil.copy(mesh, meshes_dir / "b.glb")

    run_mesh_simplify(str(meshes_dir / "*.glb"), str(output_dir / "*.glb"), factor=0.5)

    assert is_glb(str(output_dir / "a.glb"))
    assert is_glb(str(output_dir / "b.glb"))
