# Track: Dexmate Vega whole body (`vega_sharpa`)

Steps 3–7 of `onboard_ego_clip` for the Vega whole-body embodiment. Assumes Steps 0–2 are done:
the bundle is validated and leveled.

**Skip Step 2b.** This driver reads `result.npz` directly and runs its own MANO forward
kinematics; it never touches the `loaded` Parquet. The leveled bundle directory is its input.

## What makes this track different

The driver places the human scene into the robot's frame: it yaws by `--yaw_deg` (180° by
default) and translates the wrist midpoint onto the IK reference posture's wrist midpoint. It
does **not** resize the scene — `--scene_scale` defaults to 1.0, because the pipeline contract
is that bundles are metric (Step 0).

Things to keep straight:

- **The object mesh is shared with the floating-hand track.** Pass `--reuse_object_assets` so
  the parquet points at the `<seq>.obj` / `<seq>_rigid.urdf` already installed, instead of
  writing a private copy. Same physical object, one asset.
- All world quantities in the parquet (`ee_pose_w`, object poses) are in the **reduced-model
  world** of `vega_sharpa_reduced.urdf`, root at the origin. `T_reduced_to_fixed` ships in the
  diagnostics sidecar only and is deliberately **not** applied to world poses — it is not a pure
  yaw, so folding it in tilts gravity relative to the replayed robot.
- Contacts are generated here, not inherited: FK probe frames against the object mesh at
  `--contact_threshold` (0.018 m), de-flickered by `--min_consecutive`, and gated to the
  manipulation window so a resting object does not register contact.

### On `--scene_scale auto`

`auto` sets `2 / (mean(hand_scale_left) + mean(hand_scale_right))` — i.e. `1/mean_hand_scale` —
and applies it to every world translation **and** to the object mesh vertices. It is an escape
hatch for a bundle you have established is non-metric, not a default.

Reaching for it on a mild hand scale is a trap. Betas absorb real anatomy at ±10–15%, so ~1.1
is far more likely a large hand than an oversized world; `auto` then shrinks the whole scene by
that factor and the object mesh silently stops matching the floating-hand track's copy of the
same object. If a bundle genuinely is non-metric, rescale the **bundle** once at Step 0 so both
tracks inherit the correction. The driver warns if you combine `--reuse_object_assets` with a
vertex scale away from 1.0, because that pairs a shared mesh with a rescaled trajectory.

## Step 3 — Retarget and support surface

With `--reuse_object_assets` the driver writes no object assets, so the shared
`<seq>.obj` / `<seq>_rigid.urdf` must already exist.

```bash
python scripts/retarget/ego_recon_to_dexmate_sharpa.py \
  --bundle_dir <leveled-bundle-dir> \
  --mano_model_dir /workspace/mano_weights/hand \
  --sequence_id <seq> --object_name <seq> --reuse_object_assets --save --video
```

- **`--reuse_object_assets` needs the object assets installed first**, so run the floating-hand
  track's `generate_rigid_urdfs.py` (or the loader) before this. Without the flag the driver
  writes its own `<object_name>.obj` / `.urdf` into `--output_root` and overwrites the shared
  mesh. A missing asset is a hard error, not a silent objectless parquet.
- `--video` is on in the command above and should stay on: it is the fastest way to catch a
  bad placement before Isaac. It lands in `robotic_grounding/out/` (gitignored) along with the IK sidecar
  and report, never in the asset tree. The default `--video_azimuth_deg 225` is a
  three-quarter view of the robot's front; 45 is the same view of its back, which hides the
  grasp behind the torso.
- `--source_fps` must match the bundle (30 for most ego clips); `--target_fps` defaults to 50.

Then the support surface, which **must** get an explicit robot-specific `--output`:

```bash
python scripts/reconstruct_support_surfaces.py --dataset motion_v1 \
  --input_dir source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/processed \
  --output source/robotic_grounding/robotic_grounding/assets/human_motion_data/ego_recon/reconstructed_stage/<seq>_vega_sharpa_support.usda
```

Without the explicit `--output` this writes the shared `<seq>_support.usda` and clobbers the
floating-hand surface. Both tracks now share the object name, so confirm the generator read
the *Vega* parquet: its frame count should match this track's `--target_fps` output, not the
floating-hand one's.

Checks before moving on:

- The driver's JSON report: `warnings` empty, `converged_pct` high, and position errors around
  **1 cm mean fingertips / 1.5 cm wrists**, well under 4 cm max.
- `scene_scale` is 1.0 unless you deliberately overrode it. If you passed `auto`, check the
  printed value: anything away from 1 desyncs the shared object mesh from this motion.
- Support reconstruction keeps **at least one disk above ground**.
- **Seating check** — the real test that the surface matches the motion. Transform the object
  mesh by every frame's pose and compare the minimum vertex z against the support-disk top
  (`translate.z + height/2`). Expect the minimum across all frames to land *on* the top with
  zero penetration; more than ~1 cm either way means the surface and the motion came from
  different parquets.
- Contact coverage: the driver prints active frames per side. A grasp-and-move clip should show
  contact on a large minority of frames (≈ 50% is typical). Near-zero contact means
  `force_closure` will be un-gated and the policy will never learn to grasp.

## Step 4 — USER GATE 1: visual replay check

Step 3 already rendered the verification video — that is the artefact to show. **Tell the user
its absolute path**, because they cannot see your tool output:

```
robotic_grounding/out/<seq>_vega_sharpa.mp4
```

`out/` is gitignored, so nothing lands in the asset tree. It needs no X11 and no Isaac. Note the
video can only be produced during a `--save` run: there is no "render this parquet" path, so
re-rendering means re-running the retarget.

For a live look, write the command to `tmp.sh` and have the user run `bash tmp.sh`.

```bash
python scripts/replay_motion.py --motion_file <.../processed/sequence_id=<seq>/robot_name=vega_sharpa>
```

**Ask the user to confirm**: the object rests on the support with no float or sink, both arms
reach plausibly without the torso lunging, fingers close on the object rather than through it,
and the motion is smooth. **Do not proceed until confirmed.**

## Step 5 — Smoke train

```bash
python scripts/rsl_rl/train.py \
  --headless --task VegaSharpa-WholeBody-Manip-v0 \
  --motion_file ego_recon/processed/<seq>/vega_sharpa \
  --num_envs 64 --max_iterations 10 --logger tensorboard --run_name <seq>_vega_smoke
```

**No reward overrides.** Unlike the floating-hand track, `VegaSharpaManipEnvCfg` already carries
the force-closure recipe in the config — the contact-wrench, missed-contact and unintended-contact
terms ship at weight 0.0 and `force_closure` at 5.0 — and its curriculum schedules only the
virtual-object-control scale, with no `rewards_*` params. Passing the floating-hand overrides
here fails on unknown config keys.

Still confirm the **Active Reward Terms** table at startup reads as above before trusting a run.

Pass criteria: exit 0, all iterations complete, no NaN or traceback, nonzero object and hand
tracking rewards.

**Read the metrics, not just the exit code.** A metric pinned at exactly `0.0000` on every
iteration is a wiring failure, not good tracking -- Vega hit this on both wrist terms at once.
Equally, an error stuck near the robot's own scale (~1 m for a wrist) usually means the
reference is zeros rather than that the policy is bad. Iteration 0 reads all-zero legitimately,
because no episode has finished yet; judge from the last iteration. For reference, a healthy
10-iteration Vega smoke lands around 7 cm wrist position error, 0.19 rad wrist orientation
error, and 0.06 m object position error.

## Step 6 — USER GATE 2: review the smoke run

Show the metrics table (iterations, episode-length ratio, tracking rewards, wrist errors,
terminations) and ask whether to proceed to a full run. For a real run, judge on **drop rate,
not reward** — a ~1.5 rad finger tracking error is inherent to this retarget and is not the
signal to optimize.

## Step 7 — Commit (on approval)

One `feat(ego_recon):` commit with:

- `processed/sequence_id=<seq>/robot_name=vega_sharpa/data.parquet`
- `reconstructed_stage/<seq>_vega_sharpa_support.usda`

The driver's verification video, IK sidecar and report go to `robotic_grounding/out/`
(gitignored) via `--artifact_dir`, so there is nothing regenerable to exclude from the commit.

`chown` the tree first if the container wrote it, then confirm every binary is an LFS pointer:

```bash
git diff --cached --name-only | while read f; do
  b=$(git rev-parse ":$f"); s=$(git cat-file -s "$b")
  [ "$s" -gt 1000000 ] && echo "RAW BLOB $((s/1024/1024))MB $f"
done
```

Expect no output.

## Track-specific gotchas

| Symptom | Cause / fix |
|---|---|
| `find_bodies` fails, `root: []`, bodies start at `base` | the task loaded `vega_sharpa_reduced_fixed.urdf`; the motion is authored in `vega_sharpa_reduced.urdf`, whose root link is `root` |
| all contact rewards silently 0 | same reduced-vs-reduced_fixed mismatch, or `hand_contact_active` all-zero because no object mesh was available at retarget time |
| object contact sensor fails at env build | the object URDF's link must be `<link name="object">`; this driver writes that, hand-written URDFs often do not |
| `<seq>.obj` shows as modified in `git status` | ran without `--reuse_object_assets`, so the driver wrote its own copy over the shared mesh |
| object rests at the wrong height in Isaac | no `<seq>_vega_sharpa_support.usda`; discovery fell back to the unscaled shared surface |
| support surface built from the wrong embodiment | `reconstruct_support_surfaces.py` picked the `sharpa_wave` parquet — check the `Object bodies:` log line |
| wrist tracking error reads exactly 0.0000 every iteration | the wrist body never resolved: side inferred by `"left" in name.lower()`, which misses Vega's `L_arm_l7`. The keypoint reward then sees zero error -- max reward, no signal. Set `left/right_wrist_body_name` on the command cfg |
| wrist tracking error sits near 1.0 m and never improves | the wrist *reference* is missing, so the command falls back to zeros and the "error" is just the wrist's distance from the env origin. Same side-token miss, in the motion reader deriving `<side>_wrist_position` from `ee_link_names` |
| PhysX `Filter pattern ... did not match ... found 0` | a contact body name that `.*`->side substitution cannot produce (`left_arm_l7` vs `L_arm_l7`). PhysX only logs it, so that body silently stops sensing contact -- declare `hand_contact_bodies_by_side` on the robot spec |
| `ModuleNotFoundError: contact_labels` | the driver needs its siblings `contact_labels.py` and `mano_to_dexmate_sharpa.py` in `scripts/retarget/` |
| joints rejected by Isaac `_validate_cfg` | float32 cast landed an epsilon outside the URDF limits; the driver clamps with an interior margin, so suspect a hand-edited parquet |
