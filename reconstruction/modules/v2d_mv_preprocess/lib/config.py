"""Configuration helpers for multi-view preprocessing."""

from pathlib import Path
import warnings


def resolve_calibration_camera_params_path(
    calibration_camera_params_path: str | Path | None = None,
    extrinsics_camera_params_path: str | Path | None = None,
) -> str | Path | None:
    """Resolve the canonical calibration path and its deprecated alias.

    The deprecated name is retained for Python, YAML, CLI, and Docker callers.
    Supplying both names is allowed only when they identify the same path.
    """
    if extrinsics_camera_params_path is not None:
        warnings.warn(
            "extrinsics_camera_params_path is deprecated; use "
            "calibration_camera_params_path instead",
            FutureWarning,
            stacklevel=2,
        )

    if calibration_camera_params_path is None:
        return extrinsics_camera_params_path
    if extrinsics_camera_params_path is None:
        return calibration_camera_params_path

    canonical = Path(calibration_camera_params_path).expanduser().resolve(strict=False)
    deprecated = Path(extrinsics_camera_params_path).expanduser().resolve(strict=False)
    if canonical != deprecated:
        raise ValueError(
            "Conflicting calibration camera params paths: "
            f"calibration_camera_params_path={calibration_camera_params_path!s}, "
            f"extrinsics_camera_params_path={extrinsics_camera_params_path!s}"
        )
    return calibration_camera_params_path
