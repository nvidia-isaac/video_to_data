from pathlib import Path

import pytest

ASSETS = Path(__file__).parent.parent.parent / "assets"


def pytest_addoption(parser):
    parser.addoption(
        "--output-dir",
        default=None,
        help="Save test output artifacts here instead of a temp dir (useful for local inspection)",
    )


@pytest.fixture
def output_dir(request, tmp_path):
    """
    Output directory for a single test.

    With --output-dir /some/path: writes to /some/path/<test_name>/
    Without:                      writes to pytest's tmp_path/output/
    """
    custom = request.config.getoption("--output-dir")
    if custom:
        d = Path(custom) / request.node.name
        d.mkdir(parents=True, exist_ok=True)
        return d
    return tmp_path / "output"


@pytest.fixture
def mesh():
    return str(ASSETS / "mesh.glb")


@pytest.fixture
def intrinsics():
    return str(ASSETS / "intrinsics.json")


@pytest.fixture
def transform():
    return str(ASSETS / "transform.json")


@pytest.fixture
def transforms_glob():
    return str(ASSETS / "transforms/*.json")


@pytest.fixture
def background_image():
    return str(ASSETS / "test_image.jpg")


def is_glb(path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"glTF"


def is_png(path) -> bool:
    with open(path, "rb") as f:
        return f.read(4) == b"\x89PNG"
