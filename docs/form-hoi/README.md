# FORM-HOI

**FORM-HOI: Foundry for Multi-view Reconstruction of Human-Object
Interaction** is a dataset of human demonstrations involving everyday objects,
captured from synchronized camera views and reconstructed into temporally
aligned 3D representations.

FORM-HOI is produced using this repository's MV-HOI reconstruction pipeline.
See the [local MV-HOI pipeline guide](../../reconstruction/docs/mv_hoi_local_pipeline.md)
for an overview of the processing stages.

## Dataset contents

Each released sequence includes:

- synchronized multi-view RGB videos and calibrated camera parameters;
- stereo depth and human and object segmentation masks;
- an aligned object mesh, symmetry metadata, and framewise 6-DoF object poses;
- reconstructed human body parameters and meshes, SOMA-X data, and an estimated
  ground plane;
- tiled visualization overlays; and
- interaction-trim metadata, quality metrics, and human and automatic failure
  segments.

The sequences are trimmed around the recorded interaction and quality checked
before release.

## Availability

The FORM-HOI dataset is coming soon to Hugging Face.
