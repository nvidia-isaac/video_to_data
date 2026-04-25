# Task Library OSMO Bundle Storage

Consumable task-library data is published as one OSMO dataset bundle per source
dataset. A bundle contains the derived motion data and every asset needed to
train, validate, replay, or visualize it without checking out git-lfs assets
from the `robotic_grounding` repository.

Raw provider datasets are not part of the published OSMO bundle. They may still
be used as private/bootstrap inputs when generating a new bundle.

## Bundle Layout

Example:

```text
v2d_{dataset}_bundle:<version>
  manifest.json              # Bundle schema, provenance, and asset checksums
  assets/
    meshes/{dataset}/        # Object meshes used by parquets and URDFs
    urdfs/{dataset}/         # Generated or curated object URDFs
  {dataset}_loaded/          # Parquet: MANO + object poses
  {dataset}_processed/       # Parquet: IK-retargeted robot trajectories
  reconstructed_stage/       # Support surface USDs, when present
  {dataset}_html/            # Optional Viser recordings / pyrender MP4s
  {dataset}_videos/          # Optional Isaac Sim MP4s via dummy_agent
  {dataset}_quality.csv      # Optional sequence quality report
```

Parquet `object_mesh_paths` and `object_urdf_paths` must be bundle-relative, for
example:

```text
assets/meshes/taco/023_cm.obj
assets/urdfs/taco/023_rigid.urdf
```

## Materializing A Bundle

Download a pinned bundle version into the local layout expected by training:

```bash
python scripts/materialize_osmo_bundle.py \
  --dataset taco \
  --version <version>
```

By default this downloads to:

```text
${HUMAN_MOTION_DATA_DIR:-source/robotic_grounding/robotic_grounding/assets/human_motion_data}/{dataset}/
```

You can also call OSMO directly:

```bash
osmo dataset download v2d_taco_bundle:<version> \
  source/robotic_grounding/robotic_grounding/assets/human_motion_data/taco/
```

After materialization, training motion paths resolve as:

```text
taco/taco_processed/<sequence_id>/sharpa_wave
```

which maps to:

```text
.../human_motion_data/taco/taco_processed/sequence_id=<sequence_id>/robot_name=sharpa_wave
```

## ARCTIC Assets

ARCTIC is a curated/manual asset case. There is no script that regenerates its
URDFs; `generate_rigid_urdfs.py` intentionally skips articulated datasets.
The ARCTIC bundle must copy the curated asset set:

```text
assets/meshes/arctic/<object>/
assets/urdfs/arctic/<object>.urdf
assets/urdfs/arctic/<object>_art.urdf
assets/urdfs/arctic/<object>_rigid.urdf
```

The manifest should mark these entries as curated/manual assets and include
checksums. Runtime code resolves ARCTIC registry paths relative to the
materialized bundle root.

## Validation Gate

Before publishing a bundle, validate that:

- No newly generated parquet stores absolute `object_mesh_paths` or `object_urdf_paths`.
- Every parquet asset reference resolves inside the materialized bundle.
- Every URDF mesh reference resolves inside the bundle asset tree.
- `manifest.json` lists the expected assets and checksums.

## Legacy Outputs

Older retarget outputs may contain repo-absolute asset paths. They are treated
as obsolete and are not part of the portable bundle contract.

## Raw Input Discovery

`list_css_sequences.py` remains useful for private/bootstrap raw data discovery
while generating new bundles, but CSS raw storage is not the source of truth for
published consumable data.
