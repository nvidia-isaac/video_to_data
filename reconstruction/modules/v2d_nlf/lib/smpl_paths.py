"""
Resolve SMPL model root paths from weights_dir.

Always uses weights_dir/smpl as the SMPL root. Supports:
- smplh: weights_dir/smpl/smplh/ (or weights_dir/smpl if flat)
- smpl: weights_dir/smpl/ with SMPL_*.pkl at top level, or weights_dir/smpl/smpl/

Also ensures smplfitter assets (e.g. kid_template.npy) are present in the model root.
"""
import os
import shutil


def ensure_smplfitter_assets(model_root: str, model_type: str, weights_dir: str) -> None:
    """Copy smplfitter assets from parent smpl/ dir into model_root when missing.
    Manual extracts often have kid_template.npy at smpl/ level; smplfitter expects it in smplh/.
    """
    if model_type != "smplh":
        return
    kid_dst = os.path.join(model_root, "kid_template.npy")
    if os.path.isfile(kid_dst):
        return
    smpl_root = os.path.join(weights_dir, "smpl")
    kid_src = os.path.join(smpl_root, "kid_template.npy")
    if os.path.isfile(kid_src):
        shutil.copy2(kid_src, kid_dst)


def get_smpl_model_root(model_type: str, weights_dir: str) -> str:
    """Return the directory containing model files for the given model type.
    weights_dir: directory containing NLF weights and smpl/ subdir (e.g. from download or manual extract).
    """
    smpl_root = os.path.join(weights_dir, "smpl")

    if model_type == "smplh":
        subpath = os.path.join(smpl_root, "smplh")
        if os.path.isdir(subpath):
            ensure_smplfitter_assets(subpath, model_type, weights_dir)
            return subpath
        return smpl_root

    if model_type == "smpl":
        standard = os.path.join(smpl_root, "smpl")
        if os.path.isdir(standard) and _has_smpl_files(standard):
            return standard
        if _has_smpl_files(smpl_root):
            return smpl_root
        return standard

    return os.path.join(smpl_root, model_type)


def _has_smpl_files(directory: str) -> bool:
    """Check if directory contains SMPL model files (SMPL_*.pkl or SMPL_*.npz)."""
    patterns = [
        os.path.join(directory, "SMPL_MALE.pkl"),
        os.path.join(directory, "SMPL_FEMALE.pkl"),
        os.path.join(directory, "SMPL_neutral.pkl"),
        os.path.join(directory, "SMPL_MALE.npz"),
        os.path.join(directory, "SMPL_FEMALE.npz"),
    ]
    return any(os.path.isfile(p) for p in patterns)
