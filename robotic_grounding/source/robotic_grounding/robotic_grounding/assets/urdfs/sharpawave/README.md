## Sharpa Wave Hand Assets

The Sharpa Wave hand-related meshes, URDFs, and XML files are derived from the original files provided in the [Sharpa Wave hand repository](https://github.com/sharpa-robotics/sharpa-urdf-usd-xml).

For our use case, we adapt the URDF and XML assets as follows:

1. Add sites to the XML files for retargeting.
2. Replace collision meshes with primitive capsule geometries to reduce simulation time.
3. Modify the thumb and palm meshes to avoid self-collisions.
