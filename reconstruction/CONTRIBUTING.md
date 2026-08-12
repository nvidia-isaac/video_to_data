# Contributing to Video-to-Data Reconstruction

Thanks for your interest. This file is a quick map; the authoritative
developer reference is **`reconstruction/CLAUDE.md`**, which documents the
host/container architecture and the typed-contract conventions every module
follows.

## Reporting issues

File a GitHub issue with:

- Repro steps (a minimal video / config / command).
- Expected vs actual behavior.
- The module(s) involved (e.g. `v2d_moge`, `v2d_foundation_pose`) and
  whether you were running the host wrapper or inside a container.
- Your host Python version, Docker / NVIDIA Container Toolkit versions,
  GPU model, and driver/CUDA versions.

## Pull-request workflow

1. Fork or branch from `main`.
2. Set up the host-side environment and build the images you need:

   ```bash
   cd reconstruction
   ./scripts/install_packages.sh      # host-side orchestration wrappers
   ./scripts/build_containers.sh      # or build a single module's image
   ```

3. Make focused commits. Each commit should be independently reviewable;
   prefer many small commits over one mega-commit.
4. When you change `lib/` code, run the affected module's wrapper with
   `dev=True` (or `--dev`) so the container mounts your local `/workspace`
   and picks up the change without a rebuild.
5. Open the PR; CI (`workflow_dispatch` on the self-hosted GPU runner) runs
   the module build + smoke inference. Aim for a green CI before requesting
   review.

## Code style

- **Typed contracts:** packages communicate through dataclasses from
  `v2d_common` / the shared packages, not raw `np.ndarray`, `trimesh.Trimesh`,
  etc. See "Typed Contracts Between Packages" in `CLAUDE.md`.
- **Function naming:** name functions by the primary data type they operate
  on (`depth_to_pointcloud`, `mesh_render_image`), so the data flow reads
  left-to-right.
- **Host / container split:** host installs (`modules/v2d_*/docker/`) carry
  zero ML dependencies; heavy ML code lives in `lib/` and runs only inside
  containers.
- **Wrapper completeness:** every `lib/` function parameter must be reachable
  from both its `lib/run_*.py` and `docker/run_*.py` wrappers.

## Commit messages

Follow the convention already in `git log`:

```
[<area>] <imperative summary>

Optional body explaining the *why*.
```

Example: `[reconstruction] v2d_moge: stabilize intrinsics across frames`.

## Signing Your Work

* We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

  * Any contribution which contains commits that are not Signed-Off will not be accepted.

* To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
  ```bash
  $ git commit -s -m "Add cool feature."
  ```
  This will append the following to your commit message:
  ```
  Signed-off-by: Your Name <your@email.com>
  ```

* Full text of the DCO (https://developercertificate.org/):

  ```
    Developer Certificate of Origin
    Version 1.1

    Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

    Everyone is permitted to copy and distribute verbatim copies of this
    license document, but changing it is not allowed.


    Developer's Certificate of Origin 1.1

    By making a contribution to this project, I certify that:

    (a) The contribution was created in whole or in part by me and I
        have the right to submit it under the open source license
        indicated in the file; or

    (b) The contribution is based upon previous work that, to the best
        of my knowledge, is covered under an appropriate open source
        license and I have the right under that license to submit that
        work with modifications, whether created in whole or in part
        by me, under the same open source license (unless I am
        permitted to submit under a different license), as indicated
        in the file; or

    (c) The contribution was provided directly to me by some other
        person who certified (a), (b) or (c) and I have not modified
        it.

    (d) I understand and agree that this project and the contribution
        are public and that a record of the contribution (including all
        personal information I submit with it, including my sign-off) is
        maintained indefinitely and may be redistributed consistent with
        this project or the open source license(s) involved.
  ```

## Architecture conventions

Documented in **`reconstruction/CLAUDE.md`**. Highlights:

- Each package does one thing; shared logic is pulled up into
  `v2d_common` / `v2d_depth` / `v2d_pointcloud` / `v2d_mesh` / `v2d_smpl`
  rather than duplicated.
- Modules communicate via files on disk (depth PNGs, intrinsics JSON,
  `Transform3d` JSON, SMPL `.npz`, etc.), not in-process objects — though a
  module may import another module's `lib` for in-memory utilities.
- Multi-view programs live in `lib/` as `mv_*.py` with a same-named `.yaml`
  config and a `*_from_config(cfg)` entry point.
