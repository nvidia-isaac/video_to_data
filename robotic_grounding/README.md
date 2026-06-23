# Robotic Grounding

This is a stage of the **Video to Data (V2D)** pipeline. It turns
reconstructed human demonstrations into deployable robot policies:

- **Motion retargeting** — maps human hand and whole-body motion 
  onto a target robot embodiment (e.g. Sharpa, G1), across the supported
  hand-object and whole-body schemas.
- **RL training** — drives [NVIDIA Isaac Lab](https://isaac-sim.github.io/IsaacLab/)
  environments with the retargeted motion and the reconstructed scene, training
  control policies with RL.

The source for this stage is not included in this release and will be published
in a later release.

The technical report and project website are available on the project page: [Webpage](https://nvidia-isaac.github.io/video_to_data/chord/).