"""Access the shared, dependency-free storage helpers from a source checkout."""
from pathlib import Path
import sys

# Host orchestration does not require installing the GPU/module packages.
_COMMON_DIR = Path(__file__).resolve().parents[3] / "modules" / "v2d_common"
if str(_COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(_COMMON_DIR))
from object_storage import parse_storage_url, s3_client_kwargs

__all__ = ["parse_storage_url", "s3_client_kwargs"]
