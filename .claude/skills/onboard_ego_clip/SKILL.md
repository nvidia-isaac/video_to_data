---
name: onboard_ego_clip
description: Interactive walkthrough for turning a NEW ego-video reconstruction (a result.npz bundle) into a trainable robotic_grounding sequence — validate the bundle, level it, then retarget to the floating Sharpa hand and/or the Dexmate Vega whole body, build object assets and support surfaces, visually check the replay, and smoke-train. Use when the user says "onboard this reconstruction", "run retargeting on <new clip>", "add this new clip/result.npz for training", "retarget my ego video", "retarget this clip to Vega", or points at a new outputs_*/result folder. For an entirely new DATASET plus loader, follow "Adding a New Dataset" in robotic_grounding/workflow/data_pipeline.md; for generating a single command use robotic_grounding_run; for failures use robotic_grounding_doctor.
---

# onboard_ego_clip — new ego reconstruction → validated training sequence

Takes a raw ego reconstruction bundle (`.../outputs_<name>/result/{result.npz, mesh.obj,
manifest.json}`) to a committed, trainable sequence, with **two interactive user gates**: a
visual check of the retarget, and a review of the smoke training. All work runs from
`robotic_grounding/`; heavy steps run inside the container.

One clip can feed **two embodiments**. Steps 0–2 are shared: they validate and level the raw
bundle, which is the expensive, judgement-heavy part. The tracks diverge at retargeting.

| | Floating hand | Vega whole body |
|---|---|---|
| `robot_name` | `sharpa_wave` | `vega_sharpa` |
| Track guide | [references/sharpa_wave.md](references/sharpa_wave.md) | [references/vega_sharpa.md](references/vega_sharpa.md) |
| Consumes | the **`loaded` Parquet** (Step 2b) | **`result.npz` directly** — skips the loader |
| Object mesh | installs `<seq>.obj` | **reuses** the same `<seq>.obj` |
| Isaac task | `Sharpa-V2D-v0` | `VegaSharpa-WholeBody-Manip-v0` |

Pick `<seq>` = a distinct sequence/object name per reconstruction (e.g. `tissue_box`).
**Never reuse an existing object name** — each clip's mesh has its own local frame, so a reused
name silently pairs a motion with the wrong mesh. Avoid generic names like `box`, which collide
in the object registry and resolve to an ARCTIC URDF. Reusing a name across *clips* has already
caused a silent regression: a `tissue_box_simple` data swap left a committed Vega motion pointing
at a mesh from a different recording.

## Step 0 — Validate the raw bundle (host, cheap, do FIRST)

Load the npz and report a small table to the user:

- **NaN scan** over every float array. The refinement stage has shipped all-NaN pose tracks
  before; any NaN means stop and rebuild the bundle upstream.
- Validity masks (`camera/object/hand_*_is_valid`) ≈ 100%.
- **MANO hand scale (`hand_right_scale` / `hand_left_scale`) ≈ 1.0.** This is the metric-ness
  check, and the most consequential number in the bundle. The hand is the only absolute metric
  ruler in an ego clip; betas absorb real anatomy at ±10–15%, so a fitted scale ≳ 1.3 means the
  whole **world** is oversized by that factor. The object mesh inherits the error (deep
  reference penetration) and the retarget's `--mano_to_robot_scale` (default 1.2, which assumes
  ≈1.0 hands) is violated. Preferred fix is a uniform whole-scene rescale of the bundle by
  ~1/hand_scale (mesh plus all translations), not a retarget-side workaround.
- Object-size sanity against the physical object, using **bbox-volume-minimizing**
  canonical dims. Raw bbox extents lie when the mesh is internally tilted.
- `gravity_alignment_applied` and `gravity_in_world_before_alignment` — note them, but do not
  trust them yet (Step 2 decides).
- Frame-0 rest contract: object drift over the first ~30 frames should be mm–cm.
- Rotation blocks of `*_to_world_transform` may carry a uniform **scale**. Polar-decompose
  (SVD) before deriving any correction rotation; a naive transpose product silently yields a
  non-rotation (check `det == 1`).

**Then ask which embodiment(s) to onboard** — floating hand, Vega, or both. Both tracks read
the same leveled bundle, so onboarding both later costs only the track-specific steps.

## Step 1 — Container setup (one-time)

Use `robotic-grounding:latest`; the training images lack `manotorch`.

```bash
docker run -d --gpus all --entrypoint /bin/sleep -e ACCEPT_EULA=Y -e HEADLESS=1 \
  -v <repo-root>/video_to_data:/workspace/video_to_data \
  -v <repo-root>/video_to_data/reconstruction/data/weights:/workspace/mano_weights \
  --name rg-retarget robotic-grounding:latest infinity
```

- **`--entrypoint /bin/sleep` is mandatory.** The default entrypoint spawns a never-exiting
  Isaac app that squats on the GPU.
- **Mount the tree you actually intend to build.** The bind mount is whatever path you pass, so
  running this from a different worktree silently retargets against that worktree's assets.
- The MANO weights mount is only needed for the Vega track, which runs its own MANO FK and takes
  `--mano_model_dir /workspace/mano_weights/hand`. Harmless to always include.
- pinocchio needs `liburdfdom_*.so.4.0`, which the image ships as `.so.6`; and
  `libtinyxml2.so.11`, which it ships as `.so.9` under the ros2 bridge. Symlink both into one
  shim dir and prepend it to `LD_LIBRARY_PATH`, or link in place next to the `.so.6` files.
- `pip` alone is not on PATH — always `python -m pip`.
- Container writes land **root-owned**. `chown -R $(id -u):$(id -g)` the output tree before any
  host-side `git rm` / `rm`, or they fail with `Permission denied`.

## Step 2 — Level the bundle (shared, and the judgement-heavy step)

**Do not blindly trust `gravity_alignment_applied`.** The decisive test is the **contact
footprint at rest**: take mesh vertices within 5 mm of the lowest world-z at a rest frame. A
face-down box gives a broad planar patch (e.g. 23 × 8 cm); an edge-resting one gives a thin
strip (e.g. 16 × 1 cm), meaning the upstream alignment is wrong.

**If you see a strip, do not expect either loader flag to fix it.** `--no_ground_align` keeps
the tilt, and the loader's OBB path picks "up" by *smallest PCA extent*, which is arbitrary
when the two smaller extents are close — on a 22.2 × 12.7 × 11.2 cm tissue box whose PCA
extents came out [16.8, 16.7, 22.8] cm it chose an axis 40° off and turned a 15.6 × 3.4 cm
strip into a 22.1 × 1.1 cm knife edge. Level the bundle once instead, with
`v2d.task_library_loader.lib.level_result_bundle`, then load with `--no_ground_align`.
Prefer the `--up_rank` giving the **smallest** correction angle: a small angle is residual
GeoCalib error, while a large one re-orients the object onto a different face and needs the
video to confirm. Cross-check by measuring the tilt across the whole clip — a tilt that stays
constant while the object is carried is a world-frame error, not object motion.

The leveled bundle directory is the **shared artifact both tracks consume**. Record its path;
the Vega track reads it directly and the floating-hand track loads from it.

## Step 2b — Load to the `loaded` Parquet (floating-hand track only)

Skip this entirely if you are only onboarding Vega — that driver does its own MANO FK from
`result.npz` and never reads the `loaded` Parquet.

The loader is `reconstruction/modules/v2d_task_library_loader/lib/ego_recon_loader.py`
(registered as `ego_recon` in `loader_registry.py`). It imports `robotic_grounding.retarget`,
like every sibling loader, so run it with both packages installed — `python -m pip install
--no-deps -e modules/v2d_common -e modules/v2d_task_library_loader/lib`.

- **`--ground_z_offset 1.0`** — the table-height convention (object rest ≈ 1.07 m). With `0.0`
  the object sits on the floor and support reconstruction yields **zero disks**.
- Pass `--no_ground_align` when Step 2 already leveled the bundle, to avoid a double rotation.

Success markers: a world-repose gate of ~1e-7 m max joint error for both hands, and the object
mesh auto-installed alongside the motion data.

> **Layout.** Unlike the other datasets, `ego_recon` keeps every per-sequence artifact in one
> directory — `human_motion_data/ego_recon/processed/` holds the motion Parquet, the object
> mesh and material, the generated collision STL and the rigid URDF. A clip's assets are
> self-contained and move as a unit; there is no separate `meshes/` or `urdfs/` tree for it.
> If the loader installs the mesh elsewhere, move it into `processed/` before Step 3.
>
> The **loaded Parquet is the exception**: it is a regenerable intermediate, and ego_recon's
> assets are committed, so it is written to `robotic_grounding/.cache/ego_recon/loaded/`
> (gitignored) rather than into the asset tree — see `DatasetConfig.loaded_in_intermediate`.
> Never commit it. Support surfaces are *not* intermediates: `SceneConfig` discovers them
> relative to the processed motion path, so they belong in the asset tree.

## Steps 3–7 — follow your track

Both tracks cover: retarget → object assets → support surface → **USER GATE 1** (visual replay)
→ smoke train → **USER GATE 2** (review) → commit.

**Both render a verification MP4 by default**, into the gitignored `robotic_grounding/out/`.
Always report its absolute path to the user — they cannot see your tool output, and the video is
the one artefact that lets them judge the retarget themselves. It needs no X11 and no Isaac, so
there is no reason to skip it.

- **Floating hand** → [references/sharpa_wave.md](references/sharpa_wave.md)
- **Vega whole body** → [references/vega_sharpa.md](references/vega_sharpa.md)

Track guides are named for the embodiment's **`robot_name`** — the same key used by the
`robot_name=` parquet partition, `ROBOT_REGISTRY`, and `--robot`. A new embodiment gets
`references/<robot_name>.md` and a row in the table above; nothing else about the shared spine
changes, since Steps 0–2 are embodiment-independent.

Onboarding more than one? Run them in any order off the same leveled bundle, then commit once.
Read the shared-asset warning below first — it is how two tracks corrupt each other.

## One object mesh, two support surfaces

Both tracks describe the **same physical object**, so they share one `<seq>.obj` and one
`<seq>_rigid.urdf`. The Vega driver takes `--reuse_object_assets` to point at the pair the
floating-hand track already installed rather than writing its own copy.

This only works because **the bundle is metric** — which is what the Step-0 hand-scale check is
for. The Vega driver's `--scene_scale` rescales world translations *and* object mesh vertices;
at anything but 1.0 its object no longer matches the shared mesh. It defaults to 1.0. The
`auto` setting (`2/(mean(sL)+mean(sR))`) is an escape hatch for a genuinely non-metric bundle,
not a default: MANO betas absorb real anatomy at ±10–15%, so a hand scale near 1.1 is usually a
large hand, and rescaling on that evidence silently shrinks the whole scene and desyncs the two
tracks' geometry. If a bundle really is non-metric, rescale the **bundle** once (Step 0) so both
tracks inherit the fix — do not correct it inside one retargeter.

**Support surfaces still split per robot**, because the two tracks place the scene differently
even at identical scale. `_discover_support_surface` prefers `<seq>_<robot>_support.usda` and
falls back to the shared `<seq>_support.usda`, so give each track its own file. That fallback is
why a missing one stays invisible: the Vega scene silently loads the floating-hand surface and
the object rests at the wrong height.

## Gotcha index (each cost real debugging time)

| Symptom | Cause / fix |
|---|---|
| all-NaN pose arrays in the npz | refinement NaN bug — rebuild the bundle from non-refined tracks |
| object tilted at rest in replay | wrong alignment choice (Step 2), or a reused object name pairing the motion with the wrong mesh |
| support reconstruction keeps 0 disks | loaded with `ground_z_offset 0` — reload at 1.0 |
| `ModuleNotFoundError: v2d` | loader editable installs missing (Step 1) |
| `liburdfdom_*.so.4.0` missing | urdfdom shim not on `LD_LIBRARY_PATH` (Step 1) |
| `libtinyxml2.so.11` missing (pinocchio) | same shim treatment as urdfdom — link the image's `.so.9` into the shim dir (Step 1) |
| zombie Isaac app hogging the GPU | container started without `--entrypoint /bin/sleep` |
| `No data found` at train time | point `--motion_file` at the partition path `<ds>/<ds>_processed/<seq>/<robot>` |
| PCA/OBB tilt checks contradict each other | degenerate extents — use the contact-footprint test instead |
| box tilted even with loader OBB alignment | neither loader flag fixes a residual world tilt; level the bundle with `level_result_bundle`, then load with `--no_ground_align` (Step 2) |
| support USDA written next to the loaded Parquet | `--input_dir` was passed explicitly, bypassing the registry's `reconstructed_stage_dir`; support surfaces belong in the asset tree |
| `hand_*_scale` ≈ 1.4 | the world is oversized by that factor — rescale the bundle upstream; the mesh and reference penetration inherit the error |
| penetration at training resets | uniform start-frame sampling over penetrating reference frames — fix the data, not the reset |
| offset/projection makes penetration worse | the bundle is metric-consistent; the treatment is for non-metric hands only |
| correction rotation has `det != 1` | `*_to_world` rotation blocks carry a uniform scale — polar-decompose first |
| `<seq>.obj` modified after a Vega retarget | ran without `--reuse_object_assets`, so it wrote its own copy over the shared one |
| Vega object floats or sinks at reset | no `<seq>_vega_sharpa_support.usda`, so discovery fell back to the unscaled floating-hand surface |
| `Permission denied` deleting retarget output | container wrote as root — `chown` the tree first (Step 1) |
| retarget ran against the wrong assets | the container bind-mounted a different worktree than the one you are editing (Step 1) |
