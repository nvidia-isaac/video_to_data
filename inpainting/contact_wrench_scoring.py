"""Deterministic, dependency-light CHORD contact-wrench scoring.

This module is a NumPy port of the contact-wrench geometry used by
Video2Data's CHORD reward:

* ``v2d.mdp.utils.compute_tangent_basis``
* ``v2d.mdp.utils.compute_friction_cone_edges``
* ``v2d.mdp.utils.compute_wrench_space``
* ``v2d.mdp.utils.compute_wrench_space_support_function``
* ``v2d.mdp.utils_jit.contact_wrench_support_reward_jit``

The low-level functions use the same conventions as those implementations:
contact points are relative to the object center of mass and contact normals
point *into* the object.  The high-level candidate scorer deliberately accepts
object-frame points and outward mesh normals, subtracts the supplied object
center of mass, and negates the normals.  Keeping that conversion explicit
prevents the common error of rewarding forces that point out of the object.

The CHORD runtime samples a 6D basis with Torch.  Candidate ranking needs the
same directions for every grasp and reference envelope, so this module instead
provides one fixed PCG64-generated unit basis.  Its exact float64 bytes are
SHA-256 identified and its generation recipe is included in every score result.
No Torch, Isaac Lab, simulator, or mesh package is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
from typing import Any, Mapping

import numpy as np


BASIS_SCHEMA_VERSION = "v2d.inpainting.contact-wrench-basis/v1"
REFERENCE_SCHEMA_VERSION = "v2d.inpainting.contact-wrench-reference/v1"
SCORER_VERSION = "1.0.0"
NORMAL_CONVERSION_CONVENTION = (
    "outward_mesh_normals_negated_to_inward_contact_force"
)
TORQUE_NORMALIZATION_CONVENTION = (
    "cross(point_minus_object_com,unit_force)/object_radius"
)
FRICTION_CONE_PHASE_CONVENTION = (
    "uniform_[0,2pi)_edges_plus_center_normal"
)
DEFAULT_NUM_WRENCH_BASIS_SAMPLES = 512
DEFAULT_WRENCH_BASIS_SEED = 0
DEFAULT_NUM_FRICTION_CONE_EDGES = 8
DEFAULT_FRICTION_COEFFICIENT = 0.1
DEFAULT_LOW_QUANTILE = 0.1
DEFAULT_SUPPORT_THRESHOLD = 1.0e-3
DEFAULT_CHORD_TOLERANCE = 0.1
DEFAULT_CHORD_VARIANCE = 0.1
_EPS = 1.0e-6


class ContactWrenchScoringError(ValueError):
    """Raised when contact-wrench inputs have ambiguous or invalid geometry."""


@dataclass(frozen=True)
class WrenchBasis:
    """A reproducible shared set of unit directions in 6D wrench space."""

    directions: np.ndarray
    sha256: str
    seed: int
    generator: str

    def provenance(self) -> dict[str, Any]:
        """Return a JSON-serializable record of the exact basis contract."""

        return {
            "schema_version": BASIS_SCHEMA_VERSION,
            "scorer_version": SCORER_VERSION,
            "generator": self.generator,
            "seed": self.seed,
            "shape": list(self.directions.shape),
            "dtype": "float64-little-endian",
            "normalization": "row L2 normalization after standard normal sampling",
            "torque_basis_radius": 1.0,
            "sha256": self.sha256,
            "hash_payload": "C-contiguous little-endian float64 direction bytes",
            "v2d_parity_functions": [
                "v2d.mdp.utils.sample_wrench_space_basis_scaled(rc=1.0)",
                "v2d.mdp.utils.compute_wrench_space",
                "v2d.mdp.utils.compute_wrench_space_support_function",
            ],
        }


@dataclass(frozen=True)
class CandidateWrenchScores:
    """Support envelope and summary metrics for one or more candidates.

    For input contact arrays shaped ``(N, C, 3)``, the support envelope is
    ``(N, B)`` and each metric is ``(N,)``.  For a single ``(C, 3)`` contact
    array, the leading candidate dimension is removed: supports are ``(B,)``
    and metrics are NumPy scalar values.
    """

    supports: np.ndarray
    low_quantile_support: np.ndarray | np.floating[Any]
    mean_support: np.ndarray | np.floating[Any]
    support_coverage: np.ndarray | np.floating[Any]
    chord_reference_match: np.ndarray | np.floating[Any] | None
    basis_sha256: str
    basis_provenance: Mapping[str, Any]
    object_radius_m: float
    low_quantile: float
    support_threshold: float


def _finite_array(value: Any, *, name: str) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ContactWrenchScoringError(f"{name} must be numeric") from exc
    if not np.all(np.isfinite(array)):
        raise ContactWrenchScoringError(f"{name} contains non-finite values")
    return array


def _positive_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContactWrenchScoringError(f"{name} must be numeric") from exc
    if not np.isfinite(result) or result <= 0.0:
        raise ContactWrenchScoringError(f"{name} must be positive and finite")
    return result


def _canonical_basis_bytes(directions: np.ndarray) -> bytes:
    little_endian = np.asarray(directions, dtype="<f8", order="C")
    return little_endian.tobytes(order="C")


def generate_deterministic_wrench_basis(
    *,
    num_samples: int = DEFAULT_NUM_WRENCH_BASIS_SAMPLES,
    seed: int = DEFAULT_WRENCH_BASIS_SEED,
) -> WrenchBasis:
    """Generate a deterministic analogue of CHORD's sampled 6D unit basis.

    CHORD samples standard-normal 6-vectors, applies the optional torque-basis
    radius scaling, and L2-normalizes each row.  The production CHORD path uses
    ``rc=1.0`` for its shared basis; object radius is applied separately when
    constructing torque components.  This function mirrors that convention.
    """

    if isinstance(num_samples, bool) or not isinstance(
        num_samples, (int, np.integer)
    ):
        raise ContactWrenchScoringError("num_samples must be an integer")
    if num_samples <= 0:
        raise ContactWrenchScoringError("num_samples must be positive")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise ContactWrenchScoringError("seed must be an integer")

    generator = np.random.Generator(np.random.PCG64(int(seed)))
    directions = generator.standard_normal((int(num_samples), 6))
    norms = np.linalg.norm(directions, axis=-1, keepdims=True)
    directions = directions / np.maximum(norms, np.finfo(np.float64).eps)
    directions = np.ascontiguousarray(directions, dtype=np.float64)
    digest = hashlib.sha256(_canonical_basis_bytes(directions)).hexdigest()
    directions.setflags(write=False)
    return WrenchBasis(
        directions=directions,
        sha256=digest,
        seed=int(seed),
        generator="numpy.random.Generator(PCG64).standard_normal",
    )


@lru_cache(maxsize=None)
def shared_wrench_basis(
    num_samples: int = DEFAULT_NUM_WRENCH_BASIS_SAMPLES,
    seed: int = DEFAULT_WRENCH_BASIS_SEED,
) -> WrenchBasis:
    """Return the process-wide shared basis for a deterministic recipe.

    Calls with the same ``num_samples`` and ``seed`` return the same immutable
    object.  This lets a command-line experiment expose both parameters without
    regenerating or accidentally varying directions between candidates.
    """

    return generate_deterministic_wrench_basis(
        num_samples=num_samples,
        seed=seed,
    )


def compute_tangent_basis(
    normals: Any,
    *,
    eps: float = _EPS,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror CHORD's Frisvad-2012 orthonormal tangent construction."""

    normal_array = _finite_array(normals, name="normals")
    if normal_array.ndim < 1 or normal_array.shape[-1] != 3:
        raise ContactWrenchScoringError(
            f"normals must have shape (...,3), got {normal_array.shape}"
        )
    epsilon = _positive_float(eps, name="eps")

    nx, ny, nz = np.moveaxis(normal_array, -1, 0)
    sign = np.where(nz >= 0.0, 1.0, -1.0)
    denominator = sign + nz
    denominator = np.where(
        np.abs(denominator) < epsilon,
        sign * epsilon,
        denominator,
    )
    a = -1.0 / denominator
    b = nx * ny * a
    tangent_1 = np.stack(
        (
            1.0 + sign * nx * nx * a,
            sign * b,
            -sign * nx,
        ),
        axis=-1,
    )
    tangent_2 = np.stack(
        (
            b,
            sign + ny * ny * a,
            -ny,
        ),
        axis=-1,
    )
    tangent_1 /= np.maximum(
        np.linalg.norm(tangent_1, axis=-1, keepdims=True),
        epsilon,
    )
    tangent_2 /= np.maximum(
        np.linalg.norm(tangent_2, axis=-1, keepdims=True),
        epsilon,
    )
    return tangent_1, tangent_2


def friction_cone_phases(
    num_edges: int = DEFAULT_NUM_FRICTION_CONE_EDGES,
) -> tuple[np.ndarray, np.ndarray]:
    """Return CHORD-compatible evenly spaced friction-cone phases."""

    if isinstance(num_edges, bool) or not isinstance(num_edges, (int, np.integer)):
        raise ContactWrenchScoringError("num_edges must be an integer")
    if num_edges <= 0:
        raise ContactWrenchScoringError("num_edges must be positive")
    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        num=int(num_edges),
        endpoint=False,
        dtype=np.float64,
    )
    return np.cos(theta).reshape(1, -1, 1), np.sin(theta).reshape(1, -1, 1)


def _phase_array(value: Any, *, name: str) -> np.ndarray:
    array = _finite_array(value, name=name)
    if array.ndim == 1:
        array = array.reshape(1, -1, 1)
    if array.ndim != 3 or array.shape[0] != 1 or array.shape[2] != 1:
        raise ContactWrenchScoringError(
            f"{name} must have shape (K,) or (1,K,1), got {array.shape}"
        )
    if array.shape[1] == 0:
        raise ContactWrenchScoringError(f"{name} must contain at least one edge")
    return array


def compute_friction_cone_edges(
    normals: Any,
    cos_t: Any,
    sin_t: Any,
    *,
    friction_coefficients: float = DEFAULT_FRICTION_COEFFICIENT,
    eps: float = _EPS,
    append_normal: bool = True,
) -> np.ndarray:
    """Build CHORD's normalized polyhedral friction-cone rays."""

    normal_array = _finite_array(normals, name="normals")
    if normal_array.ndim != 3 or normal_array.shape[-1] != 3:
        raise ContactWrenchScoringError(
            f"normals must have shape (B,C,3), got {normal_array.shape}"
        )
    if normal_array.shape[1] == 0:
        raise ContactWrenchScoringError("normals must contain at least one contact")
    cosines = _phase_array(cos_t, name="cos_t")
    sines = _phase_array(sin_t, name="sin_t")
    if cosines.shape != sines.shape:
        raise ContactWrenchScoringError(
            f"cos_t and sin_t shapes differ: {cosines.shape} vs {sines.shape}"
        )
    coefficient = float(friction_coefficients)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ContactWrenchScoringError(
            "friction_coefficients must be finite and nonnegative"
        )
    epsilon = _positive_float(eps, name="eps")

    batch_size, num_contacts, _ = normal_array.shape
    normals_flat = normal_array.reshape(-1, 3)
    tangent_1, tangent_2 = compute_tangent_basis(normals_flat, eps=epsilon)
    edges = normals_flat[:, None, :] + coefficient * (
        cosines * tangent_1[:, None, :] + sines * tangent_2[:, None, :]
    )
    edges /= np.maximum(
        np.linalg.norm(edges, axis=-1, keepdims=True),
        epsilon,
    )
    if append_normal:
        edges = np.concatenate((edges, normals_flat[:, None, :]), axis=1)
    return edges.reshape(batch_size, num_contacts, edges.shape[1], 3)


def compute_wrench_space(
    contact_points_com: Any,
    contact_normals_inward: Any,
    cos_t: Any,
    sin_t: Any,
    *,
    object_radius_m: float,
    friction_coefficients: float = DEFAULT_FRICTION_COEFFICIENT,
    eps: float = _EPS,
) -> np.ndarray:
    """Construct primitive wrenches from COM-relative contacts.

    Args:
        contact_points_com: ``(B,C,3)`` points relative to the object COM.
        contact_normals_inward: ``(B,C,3)`` normals pointing into the object.
        cos_t: Friction-cone edge cosines, shaped ``(K,)`` or ``(1,K,1)``.
        sin_t: Friction-cone edge sines with the same shape as ``cos_t``.
        object_radius_m: Bounding-ball radius used to nondimensionalize torque.

    Returns:
        Primitive wrench columns shaped ``(B,6,C*(K+1))``.
    """

    points = _finite_array(contact_points_com, name="contact_points_com")
    normals = _finite_array(contact_normals_inward, name="contact_normals_inward")
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ContactWrenchScoringError(
            f"contact_points_com must have shape (B,C,3), got {points.shape}"
        )
    if normals.shape != points.shape:
        raise ContactWrenchScoringError(
            "contact_normals_inward must have the same shape as "
            f"contact_points_com, got {normals.shape} vs {points.shape}"
        )
    if points.shape[1] == 0:
        raise ContactWrenchScoringError(
            "contact_points_com must contain at least one contact"
        )
    radius = _positive_float(object_radius_m, name="object_radius_m")
    epsilon = _positive_float(eps, name="eps")

    normal_norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normalized_normals = normals / np.maximum(normal_norms, epsilon)
    contact_is_active = (
        np.linalg.norm(normalized_normals, axis=-1) > 1.0e-3
    )
    forces = compute_friction_cone_edges(
        normalized_normals,
        cos_t,
        sin_t,
        friction_coefficients=friction_coefficients,
        eps=epsilon,
    )
    torques = np.cross(points[:, :, None, :], forces, axis=-1)
    wrench_space = np.concatenate((forces, torques / radius), axis=-1)
    wrench_space *= contact_is_active[:, :, None, None]
    wrench_space = wrench_space.reshape(points.shape[0], -1, 6)
    return np.ascontiguousarray(np.swapaxes(wrench_space, 1, 2))


def compute_wrench_space_support_function(
    wrench_space: Any,
    basis: Any,
) -> np.ndarray:
    """Evaluate CHORD's nonnegative support envelope on shared directions."""

    wrenches = _finite_array(wrench_space, name="wrench_space")
    directions = _finite_array(basis, name="basis")
    if wrenches.ndim != 3 or wrenches.shape[1] != 6:
        raise ContactWrenchScoringError(
            f"wrench_space must have shape (B,6,W), got {wrenches.shape}"
        )
    if wrenches.shape[2] == 0:
        raise ContactWrenchScoringError(
            "wrench_space must contain at least one primitive wrench"
        )
    if directions.ndim != 2 or directions.shape[1] != 6:
        raise ContactWrenchScoringError(
            f"basis must have shape (K,6), got {directions.shape}"
        )
    if directions.shape[0] == 0:
        raise ContactWrenchScoringError("basis must contain at least one direction")
    support = np.matmul(directions[None, :, :], wrenches).max(axis=-1)
    return np.maximum(support, 0.0)


def outward_to_inward_contact_normals(
    contact_normals_outward_object: Any,
) -> np.ndarray:
    """Convert outward object-mesh normals to CHORD's inward-force convention."""

    outward = _finite_array(
        contact_normals_outward_object,
        name="contact_normals_outward_object",
    )
    if outward.ndim not in (2, 3) or outward.shape[-1] != 3:
        raise ContactWrenchScoringError(
            "contact_normals_outward_object must have shape (C,3) or (N,C,3), "
            f"got {outward.shape}"
        )
    return -outward


def compute_chord_reference_match(
    candidate_supports: Any,
    reference_supports: Any,
    *,
    tolerance: float = DEFAULT_CHORD_TOLERANCE,
    variance: float = DEFAULT_CHORD_VARIANCE,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
) -> np.ndarray | np.floating[Any]:
    """Compute the exact one-hand/one-body CHORD support-match reward.

    This mirrors ``contact_wrench_support_reward_jit`` after its left/right
    hand and object-body aggregation.  Reference-active directions are those
    above ``support_threshold``.  A candidate earns inclusion reward only when
    it is also active, and support outside the ``(1 ± tolerance)`` reference
    band receives the same squared exponential penalty as CHORD.
    """

    candidate = _finite_array(candidate_supports, name="candidate_supports")
    reference = _finite_array(reference_supports, name="reference_supports")
    if candidate.ndim not in (1, 2):
        raise ContactWrenchScoringError(
            f"candidate_supports must have shape (K,) or (N,K), got {candidate.shape}"
        )
    if reference.ndim not in (1, 2):
        raise ContactWrenchScoringError(
            f"reference_supports must have shape (K,) or (N,K), got {reference.shape}"
        )
    try:
        reference = np.broadcast_to(reference, candidate.shape)
    except ValueError as exc:
        raise ContactWrenchScoringError(
            "reference_supports are not broadcast-compatible with "
            f"candidate_supports: {reference.shape} vs {candidate.shape}"
        ) from exc
    tolerance_value = float(tolerance)
    if (
        not np.isfinite(tolerance_value)
        or tolerance_value < 0.0
        or tolerance_value > 1.0
    ):
        raise ContactWrenchScoringError(
            "tolerance must be finite and lie in [0,1]"
        )
    variance_value = _positive_float(variance, name="variance")
    threshold = float(support_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ContactWrenchScoringError(
            "support_threshold must be finite and nonnegative"
        )

    command_active = reference > threshold
    current_active = candidate > threshold
    command_count = np.maximum(command_active.sum(axis=-1), 1.0e-6)
    better = np.maximum(
        (1.0 - tolerance_value) * reference - candidate,
        0.0,
    )
    too_large = np.maximum(
        candidate - (1.0 + tolerance_value) * reference,
        0.0,
    )
    loss = np.square(better) + np.square(too_large)
    reward = np.sum(
        (command_active & current_active) * np.exp(-loss / variance_value),
        axis=-1,
    )
    return reward / command_count


def score_contact_wrench_candidates(
    contact_points_object: Any,
    contact_normals_outward_object: Any,
    *,
    object_com_object: Any,
    object_radius_m: float,
    basis: WrenchBasis | None = None,
    reference_supports: Any | None = None,
    friction_coefficient: float = DEFAULT_FRICTION_COEFFICIENT,
    num_friction_cone_edges: int = DEFAULT_NUM_FRICTION_CONE_EDGES,
    low_quantile: float = DEFAULT_LOW_QUANTILE,
    support_threshold: float = DEFAULT_SUPPORT_THRESHOLD,
    chord_tolerance: float = DEFAULT_CHORD_TOLERANCE,
    chord_variance: float = DEFAULT_CHORD_VARIANCE,
) -> CandidateWrenchScores:
    """Score parallel-jaw contact candidates without launching a simulator.

    ``contact_points_object`` and ``contact_normals_outward_object`` may
    describe one candidate as ``(C,3)`` or a batch as ``(N,C,3)``.  Points are
    converted to object-COM-relative coordinates and outward mesh normals are
    explicitly negated before the low-level CHORD computation.
    """

    points = _finite_array(contact_points_object, name="contact_points_object")
    outward_normals = _finite_array(
        contact_normals_outward_object,
        name="contact_normals_outward_object",
    )
    if points.ndim not in (2, 3) or points.shape[-1] != 3:
        raise ContactWrenchScoringError(
            "contact_points_object must have shape (C,3) or (N,C,3), "
            f"got {points.shape}"
        )
    if outward_normals.shape != points.shape:
        raise ContactWrenchScoringError(
            "contact_normals_outward_object must have the same shape as "
            f"contact_points_object, got {outward_normals.shape} vs {points.shape}"
        )
    if points.shape[-2] == 0:
        raise ContactWrenchScoringError(
            "contact_points_object must contain at least one contact"
        )
    object_com = _finite_array(object_com_object, name="object_com_object")
    if object_com.shape != (3,):
        raise ContactWrenchScoringError(
            f"object_com_object must have shape (3,), got {object_com.shape}"
        )
    radius = _positive_float(object_radius_m, name="object_radius_m")
    quantile = float(low_quantile)
    if not np.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
        raise ContactWrenchScoringError(
            "low_quantile must be finite and lie in [0,1]"
        )
    threshold = float(support_threshold)
    if not np.isfinite(threshold) or threshold < 0.0:
        raise ContactWrenchScoringError(
            "support_threshold must be finite and nonnegative"
        )
    coefficient = float(friction_coefficient)
    if not np.isfinite(coefficient) or coefficient < 0.0:
        raise ContactWrenchScoringError(
            "friction_coefficient must be finite and nonnegative"
        )

    selected_basis = shared_wrench_basis() if basis is None else basis
    if not isinstance(selected_basis, WrenchBasis):
        raise ContactWrenchScoringError("basis must be a WrenchBasis")

    single_candidate = points.ndim == 2
    if single_candidate:
        points = points[None, ...]
        outward_normals = outward_normals[None, ...]
    points_com = points - object_com.reshape(1, 1, 3)
    inward_normals = outward_to_inward_contact_normals(outward_normals)
    cos_t, sin_t = friction_cone_phases(num_friction_cone_edges)
    wrench_space = compute_wrench_space(
        points_com,
        inward_normals,
        cos_t,
        sin_t,
        object_radius_m=radius,
        friction_coefficients=coefficient,
    )
    supports = compute_wrench_space_support_function(
        wrench_space,
        selected_basis.directions,
    )
    low_support = np.quantile(supports, quantile, axis=-1)
    mean_support = supports.mean(axis=-1)
    coverage = (supports > threshold).mean(axis=-1)
    reference_match = None
    if reference_supports is not None:
        reference_array = _finite_array(
            reference_supports,
            name="reference_supports",
        )
        if single_candidate and reference_array.ndim == 2:
            if reference_array.shape[0] != 1:
                raise ContactWrenchScoringError(
                    "single-candidate reference_supports may only have one row"
                )
            reference_array = reference_array[0]
        reference_match = compute_chord_reference_match(
            supports,
            reference_array,
            tolerance=chord_tolerance,
            variance=chord_variance,
            support_threshold=threshold,
        )

    if single_candidate:
        supports = supports[0]
        low_support = low_support[0]
        mean_support = mean_support[0]
        coverage = coverage[0]
        if reference_match is not None:
            reference_match = np.asarray(reference_match)[0]

    supports = np.asarray(supports)
    supports.setflags(write=False)
    return CandidateWrenchScores(
        supports=supports,
        low_quantile_support=low_support,
        mean_support=mean_support,
        support_coverage=coverage,
        chord_reference_match=reference_match,
        basis_sha256=selected_basis.sha256,
        basis_provenance=selected_basis.provenance(),
        object_radius_m=radius,
        low_quantile=quantile,
        support_threshold=threshold,
    )


DEFAULT_WRENCH_BASIS_SHA256 = shared_wrench_basis().sha256
