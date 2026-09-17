# Track: floating Sharpa hand (`sharpa_wave`)

Steps 3–7 of `onboard_ego_clip` for the floating-hand embodiment. Assumes Steps 0–2b are done:
the bundle is validated, leveled, and loaded to the `loaded` Parquet.

Consumes the **`loaded` Parquet**. Object mesh stays at reconstruction scale.

## Step 3 — URDF, retarget, support surface (all fast)

```bash
python scripts/retarget/ego_recon_to_sharpa.py --sequence_id <seq> --save --device cuda:0
python scripts/generate_rigid_urdfs.py --dataset ego_recon
python scripts/reconstruct_support_surfaces.py --dataset ego_recon --sequence_id <seq>
```

This writes `processed/sequence_id=<seq>/robot_name=sharpa_wave/`, the object mesh `<seq>.obj`
plus its material sidecars, the collision `<seq>_visual.stl`, `<seq>_rigid.urdf`, and
`reconstructed_stage/<seq>_support.usda`.

If you are also onboarding Vega, run `reconstruct_support_surfaces.py` with an explicit
`--output .../<seq>_support.usda` when the processed dir already holds a Vega parquet — the
generator picks one parquet per sequence, and you want this surface built from the
`sharpa_wave` one.

Checks before moving on:

- Support reconstruction must report **at least one disk kept above ground**. Zero disks means
  the clip was loaded at floor level — redo Step 2b with `--ground_z_offset 1.0`.
- Verify the processed Parquet: frame count and fps match the bundle, contact columns present,
  and `robot_{right,left}_frame_task_errors` around **2.2 cm mean / under 6 cm max**. That is
  the reference-quality bar.
- **Penetration statistics.** Training resets sample start frames uniformly with no validity
  mask, so a penetrating reference frame becomes a penetrating reset. Transform the Parquet's
  `robot_*_frames` positions into the object's canonical box per frame and report the fraction
  of frames with any point inside, deeper than 1 cm, and deeper than 2 cm. Bar: **no frames
  deeper than 2 cm**. Worse clips have trained successfully, so treat this as achievable rather
  than blocking — but a high `object_away_from_trajectory` rate in the smoke run is the
  penetration-ejection signature.
- **Frame-0 seating.** Object mesh bottom under the frame-0 pose versus support-disk top.
  Ideal is within 5 mm; up to ~2 cm of float settles at reset; 4 cm or more means z-drift or a
  wrong-orientation rest.

If deep penetration remains, the fix depends on the bundle:

| Bundle | Fix |
|---|---|
| metric (hand scale ≈ 1.0) | suspect the **object** scale; do **not** apply surface projection |
| non-metric | re-retarget with `--hand_object_offset 0.01 --surface_project --surface_margin 0.005 --surface_granularity hand` |

Never use `--surface_granularity finger` (per-finger pushes produce IK-unreachable targets),
and never apply the projection treatment to a metric-consistent bundle — it projects in MANO
space *before* robot scaling and displaces an already-clean grasp.

## Step 4 — USER GATE 1: visual replay check

**Always render the verification video first** — it needs no X11, no Isaac, and no live session,
so it is the cheapest way to see the retarget and the only artefact you can hand the user
directly:

```bash
python scripts/retarget/vis_retargeted.py --dataset ego_recon --sequence_id <seq> \
  --robot sharpa_wave --save_mp4
```

That writes a single file — no viser client tree — and **you must tell the user its absolute
path**, because they cannot see your tool output:

```
robotic_grounding/out/<seq>.mp4
```

`out/` is gitignored; nothing lands in the asset tree. Add `--save_html` only when the user wants
the interactive scrub page (that is what builds the viser client bundle), or `--mp4_dir` to put
the video elsewhere.

For a live look instead, write the replay command to `tmp.sh` (project convention — never paste
multi-line commands inline) and tell the user to run `bash tmp.sh`. Use a one-shot container with
X11 forwarding; long-lived containers lack `DISPLAY`. The support USDA is auto-discovered from
the motion path.

```bash
python scripts/replay_motion.py --motion_file <.../processed/sequence_id=<seq>/robot_name=sharpa_wave>
```

`python scripts/retarget/vis_retargeted.py --dataset ego_recon --sequence_id <seq>
--start_paused` opens the viser viewer on port 8080 with the Frame slider paused at 0, so you
can scrub frame by frame. Those indices are what `motion_start_frame` / `motion_end_frame` take
(both index raw parquet frames before FPS interpolation; `motion_end_frame` is **exclusive**, so
pass the last frame you want plus one).

**Ask the user to confirm**: the object rests flat on the support disk with no tilt or float,
the hands approach and grasp plausibly, and the motion is smooth. **Do not proceed until
confirmed.** A tilted object sends you back to the Step-2 alignment decision; hands inside the
object send you to the penetration table in Step 3.

## Step 5 — Smoke train

Ego clips use the **force-closure recipe**, because monocular contact geometry is too noisy for
`contact_wrench_support_reward`. The full recipe is documented under **Ego-video (monocular)
sequences** in the package [`README.md`](../../../../robotic_grounding/README.md); for a local
smoke, run the same overrides with a short iteration budget:

```bash
python scripts/rsl_rl/train.py \
  --headless --task Sharpa-V2D-v0 \
  --motion_file ego_recon/processed/<seq>/sharpa_wave \
  --num_envs 64 --max_iterations 10 --logger tensorboard --run_name <seq>_smoke \
  env.rewards.force_closure.weight=5.0 \
  env.rewards.contact_wrench_support_reward.weight=0.0 \
  env.rewards.unintended_contact_penalty.weight=0.0 \
  env.rewards.missed_contact_penalty.weight=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_contact_wrench_support_reward=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_unintended_contact_penalty=0.0 \
  env.curriculum.fixed_timestep_curriculum.params.rewards_missed_contact_penalty=0.0
```

**Each contact term must be zeroed in BOTH places** — `env.rewards.<name>.weight` is the value
in force from step 0, and `...fixed_timestep_curriculum.params.rewards_<name>` is what the
curriculum writes on schedule. The first curriculum step lands at 2000 × `num_steps_per_env`
(24) = **48,000 sim steps**, so zeroing only the curriculum params leaves the contact terms at
full strength through the whole early phase, and a short smoke never even reaches the point
where the curriculum would correct them.

**Verify it took**: Isaac prints an **Active Reward Terms** table at startup showing the weights
in force at step 0. Confirm the three contact terms read `0.0` and `force_closure` reads `5.0`
before trusting a run — this is the failure mode that is otherwise invisible until the policy
will not grasp.

Pass criteria: exit 0, all iterations complete, no NaN or traceback, nonzero object and hand
tracking rewards. **Contact rewards near zero at iteration 10 is expected** — virtual object
control starts at 1.0 and carries the object initially.

## Step 6 — USER GATE 2: review the smoke run

Show the user the metrics table (iterations, episode-length ratio, tracking rewards, wrist
errors, terminations) and ask whether to proceed to a full run. Offer a longer local run with
`--logger wandb` if they want learning curves first.

## Step 7 — Commit (on approval)

One `feat(ego_recon):` commit with the mesh `<seq>.{obj,mtl}`, `<seq>_material_0.png`,
`<seq>_visual.stl`, `<seq>_rigid.urdf`, `processed/sequence_id=<seq>/robot_name=sharpa_wave`,
and `reconstructed_stage/<seq>_support.usda`. **Do not commit the loaded Parquet** — it is a
regenerable intermediate under the gitignored `.cache/` (see the Step-2b layout note).
Confirm every binary is an **LFS pointer** before pushing:

```bash
git diff --cached --name-only | while read f; do
  b=$(git rev-parse ":$f"); s=$(git cat-file -s "$b")
  [ "$s" -gt 1000000 ] && echo "RAW BLOB $((s/1024/1024))MB $f"
done
```

Expect no output. Anything listed is raw binary content and must be LFS-tracked instead.
