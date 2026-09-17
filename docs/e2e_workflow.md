# Reconstruction-to-policy E2E workflow

An end-to-end workflow serves as a reference for connecting the repository's components into a
complete reconstruction-to-policy pipeline. The included example starts with a raw egocentric
human demonstration captured on an iPhone and continues through reconstruction, retargeting,
source-expert training, demonstration collection, GR00T N1.7 fine-tuning, and task-level
closed-loop evaluation. It has been verified with both floating Sharpa and Vega Sharpa
embodiments.

`run_e2e.sh` provides a reproducible launcher that orchestrates the stages and records their
commands, inputs, outputs, runtime provenance, status, and logs beneath one run root for inspection
and resumption. A companion agentic skill uses the launcher under the hood to guide setup and
execution. The main path trains its own source expert; the bundled tissue-box ONNX policy is an
optional shortcut for users who want to try GR00T fine-tuning without first training the source
expert.

During expert demonstration collection, *source eligibility* means that an expert rollout reached
the configured episode timeout. Source eligibility, dataset/model plumbing, and closed-loop task
success are separate gates; passing one does not imply that either of the others passed.

## Workflow at a glance

| Main stage | Command | Primary result |
|---|---|---|
| [Initialize](#initialize-one-run) | `init` | Durable run configuration |
| [Set up and reconstruct](#choose-one-reconstruction-route) | `setup`, `doctor`, `reconstruct` or `import-reconstruction`, then `inspect` | Validated runtimes and reviewed reconstruction bundle |
| [Retarget](#retarget-and-validate-the-simulator-task) | `retarget`, `simulate` | Vega motion and simulator validation |
| [Train the source expert](#train-and-validate-the-source-expert) | `train-expert` | Validated RSL-RL expert checkpoint |
| [Collect expert demonstrations](#collect-timeout-eligible-expert-demonstrations) | `collect` | Audited joint-action demonstrations |
| [Fine-tune GR00T](#fine-tune-and-run-the-open-loop-gate) | `finetune` | Audited dataset, checkpoint, and open-loop gate |
| [Evaluate task success](#evaluate-task-success-in-closed-loop) | `evaluate` | Closed-loop task-success metrics and videos |
| [Inspect or resume](#status-resume-and-intentional-reruns) | `status` | Per-stage status and provenance |

This table is the main pipeline. The bundled tissue-box ONNX expert described later is an optional
shortcut that replaces only the source-expert training output; collection, GR00T fine-tuning, and
evaluation still follow the same stages.

## Requirements

- Complete the [Ego Reconstruction setup](../reconstruction/docs/ego_e2e_setup.md) when running
  reconstruction locally.
- Complete the [Robotic Grounding setup](../robotic_grounding/docs/SETUP.md).
- Clone and initialize [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T) in a separate
  Python 3.10 environment.
- Provide Docker, Git LFS objects, and enough disk for uncompressed semantic recordings, encoded
  videos, datasets, and model checkpoints.
- Use one 48 GB NVIDIA GPU. The complete workflow can run on an RTX A6000 and has also been
  verified on L40 and L40S GPUs.

After completing the Ego Reconstruction setup, the MANO models should be ready in the configured
MANO directory as `models/MANO_LEFT.pkl` and `models/MANO_RIGHT.pkl`. The Isaac-GR00T checkout
must support `uv run python` from its repository root. The reconstruction models and their
dependencies remain in Docker; GR00T statistics, fine-tuning, and evaluation run in this separate
Isaac-GR00T Python environment.

Throughout this guide, `vega` is the launcher shorthand for the `vega_sharpa_joint` embodiment
contract. Commands that support `--embodiment floating` select the floating Sharpa comparison;
`--embodiment both` runs both variants.

## Initialize one run

Create one durable run configuration. Choose a task profile that supplies the object prompt,
language instruction, and closed-loop evaluator independently of the embodiment contract.

```bash
./run_e2e.sh init \
  --run-root /absolute/path/to/e2e_run \
  --sequence-id example_sequence \
  --embodiment-contract vega_sharpa_joint \
  --task-profile /absolute/path/to/task_profile.json
```

The release supports the `lift_hold` evaluator. Example profiles are under
`robotic_grounding/groot_finetune/task_profiles/`; a minimal profile has this form:

```json
{
  "schema_version": 1,
  "task_id": "example_lift_hold",
  "object_prompt": "the object to reconstruct",
  "instruction": "lift the object and hold it",
  "target_object": {
    "selector": "primary"
  },
  "evaluator": {
    "id": "lift_hold",
    "lift_threshold_m": 0.1,
    "hold_threshold_m": 0.05,
    "min_hold_steps": 20
  }
}
```

Set the evaluator thresholds for the task being measured. They remain in this separate profile,
not in the environment configuration. Inputs supplied to later stages are saved in
`e2e_config.json` and reused when a command resumes.

## Choose one reconstruction route

### Route A: reconstruct a raw video

Full setup installs the host orchestration packages, builds the reconstruction and loader images,
downloads the reconstruction weights, and prepares the Robotic Grounding container.
The repository includes `reconstruction/assets/tissue_box.mp4` through Git LFS. The example below
uses its matching task profile, which supplies the object prompt `"a tissue box"`.

```bash
./run_e2e.sh init \
  --run-root "$PWD/.e2e/tissue_box" \
  --sequence-id tissue_box \
  --embodiment-contract vega_sharpa_joint \
  --task-profile "$PWD/robotic_grounding/groot_finetune/task_profiles/tissue_box_lift_hold.json"
./run_e2e.sh setup \
  --mano-dir /absolute/path/to/mano \
  --isaac-groot-dir /absolute/path/to/Isaac-GR00T \
  --accept-nvidia-model-eula
./run_e2e.sh doctor
./run_e2e.sh reconstruct \
  --video "$PWD/reconstruction/assets/tissue_box.mp4" \
  --hand-tracking hamer \
  --run-gsplat-refinement \
  --dev
./run_e2e.sh inspect
```

Pass `--accept-nvidia-model-eula` only after reviewing and accepting the NVIDIA Open Model
License described in the [reconstruction README](../reconstruction/README.md#v2d_foundation_pose).

The launcher writes the pipeline output beneath the run root and always enables reference frame
0, undistortion, DROID-SLAM, gravity alignment, and Three.js export. The final portable bundle is
`$PWD/.e2e/tissue_box/reconstruction/result_slam_gravity_aligned/`. `--dev` mounts the local
`reconstruction/modules/` sources into the worker containers; omit it when validating only the
built images rather than source-checkout changes.

![Monocular egocentric reconstruction showing the source-camera overlay and three reconstructed world views](figures/e2e_workflow/ego_reconstruction.webp)

*Monocular egocentric reconstruction: tracked hands and tissue box in the source-camera view,
alongside three reconstructed world views.*

### Route B: import a prepared reconstruction

`--skip-reconstruction` skips the heavyweight reconstruction pipeline and weight download. It
still installs and builds the task-library loader required by retargeting.

```bash
./run_e2e.sh setup \
  --mano-dir /absolute/path/to/mano \
  --isaac-groot-dir /absolute/path/to/Isaac-GR00T \
  --skip-reconstruction
./run_e2e.sh doctor
./run_e2e.sh import-reconstruction \
  --bundle /absolute/path/to/result_slam_gravity_aligned
./run_e2e.sh inspect
```

The prepared bundle must contain nonempty `result.npz`, `mesh.obj`, `manifest.json`, and
`threejs_scene/index.html` files. `inspect` reports each required file and the Three.js scene path;
review that scene before retargeting.

`doctor` must finish successfully. It checks Git LFS, the GPU, Docker, the loader image, the
external Isaac-GR00T environment, the mounted worktree, and the Isaac Lab messaging dependencies.

## Retarget and validate the simulator task

```bash
./run_e2e.sh retarget --embodiment vega
./run_e2e.sh simulate --embodiment vega
```

Retargeting must produce all of the following beneath the run root:

- `human_motion_data/ego_recon/processed/sequence_id=<id>/robot_name=vega_sharpa/`;
- `human_motion_data/ego_recon/reconstructed_stage/<id>_vega_sharpa_support.usda`;
- `vega_retarget_qa/<id>_vega_sharpa_report.json`;
- `vega_retarget_qa/<id>_vega_sharpa_ik.npz`; and
- `vega_retarget_qa/<id>_vega_sharpa.mp4`.

The report, Parquet motion, and video must have consistent robot/object frame counts, finite
values, the released 58-joint order, and no unresolved QA warnings. Inspect the beginning,
interaction, and final frames of the video.

`simulate` is a finite two-step registration and environment-advance smoke. It does not replay
the complete motion; the retarget QA artifacts are the full-motion visual gate.

![Side-by-side reconstructed human and Vega Sharpa motion retargeting](figures/e2e_workflow/vega_retargeting.webp)

*Robot motion retargeting: reconstructed human hand motion (left) and the corresponding Vega
Sharpa motion (right).*

## Train and validate the source expert

The main pipeline trains a Vega Sharpa RSL-RL source expert on the retargeted motion. The released
Vega task recipe trains for 20,000 iterations; the example below uses a validated 128-environment
configuration:

```bash
./run_e2e.sh train-expert \
  --embodiment vega \
  --max-iterations 20000 \
  --num-envs 128
```

The launcher supplies the retargeted motion, output root, environment count, and iteration budget
to the registered task. Rewards, observations, actions, commands, curricula, randomization,
terminations, runner settings, and checkpoint contents remain owned by the expert task. Training
artifacts are written beneath
`<run-root>/train_expert/vega/logs/rsl_rl/vega_sharpa_whole_body/`.

Review the task-owned training metrics and validate candidate checkpoints before collection. The
launcher does not automatically choose a checkpoint; set `RL_CHECKPOINT` to the validated RSL-RL
`.pt` file and preserve its SHA-256 in the run report:

```bash
RL_CHECKPOINT="/absolute/path/to/e2e_run/train_expert/vega/logs/rsl_rl/vega_sharpa_whole_body/<run>/model_<validated_iteration>.pt"
test -f "${RL_CHECKPOINT}"
sha256sum "${RL_CHECKPOINT}"
```

### Optional tissue-box shortcut: use the bundled ONNX expert

The repository includes a validated Vega expert ONNX policy solely as a convenience for users who
want to try the GR00T portion of the released tissue-box example without first waiting for source
expert training. It replaces the trained `.pt` checkpoint above for collection, letting the user
proceed to demonstration collection, dataset conversion, and GR00T fine-tuning. It does not skip
collection or the dataset audits required by fine-tuning.

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
RL_CHECKPOINT="${REPO_ROOT}/robotic_grounding/source/robotic_grounding/robotic_grounding/assets/policies/e2e_example/tissue_box/vega_sharpa_policy.onnx"
test -f "${RL_CHECKPOINT}"
sha256sum "${RL_CHECKPOINT}"
```

Collection runs this convenience policy through ONNX Runtime. Do not treat it as the canonical
output of the main pipeline or reuse it for another task or retargeted motion.

Whichever expert checkpoint is selected, compatibility includes its observation/action contract,
robot and task registration, reference-motion semantics, and reset distribution. A checkpoint
that loads successfully can still be incompatible with a newly retargeted motion. Inspect pilot
behavior before a large collection; if it differs materially from the expert's validation
behavior, resolve the checkpoint and motion pairing first.

The Vega source expert uses `motion_speed = 0.5`. Collection uses the corresponding playback rate
without changing the reference motion or expert action. Its pilot measures the resulting episode
horizon instead of requiring a task-specific frame count in this guide.

## Collect timeout-eligible expert demonstrations

Start with conservative rendered parallelism. Increase `--render-envs` only after measuring the
active renderer/GPU; camera-free collection and three-camera replay have different capacity.

```bash
# Example only; recalculate this value for the retargeted motion as described below.
COLLECTION_MAX_STEPS=600
./run_e2e.sh collect \
  --rl-checkpoint "${RL_CHECKPOINT}" \
  --target-successes 10 \
  --pilot-attempts 32 \
  --collection-max-steps "${COLLECTION_MAX_STEPS}" \
  --render-envs 1
```

`--collection-max-steps` is required when collection is configured for the first time. It is a
safety ceiling and must be long enough for the expert episode timeout. Estimate the timeout from
the retarget report before adding a safety margin:

```text
ceil(source_frames / source_fps / motion_speed * control_fps)
```

For this workflow, `motion_speed` is 0.5 and `control_fps` is 20. Use `frames` and `source_fps`
from the retarget report, then add a 10–20% margin and round up. The pilot records the actual
uniform episode horizon; the ceiling is not used as the recorded frame count.

Collection uses this reproducible expert rollout profile:

- resets start from the first reference frame;
- reset freeze, virtual object control, and curriculum are disabled;
- non-timeout terminations are disabled on the collection environment instance;
- arm, finger, object-XY, and object-yaw noise default to zero; and
- optional noise perturbs simulator state through reset events, never the reference motion or
  expert action.

The pilot is a separate set of attempts. If its measured timeout-eligible rate is `r`, the full
camera-free collection wave contains

```text
ceil(target_successes / r * collection_safety_factor)
```

attempts. The default safety factor is 1.125. Therefore, `--target-successes 10` means exactly ten
selected and recorded demonstrations, not ten total simulator attempts.

An equivalent prior measurement may replace the pilot only when both
`--measured-success-rate` and `--expected-frames` come from the same checkpoint, task, reset
profile, evaluator, and episode length.

Collection passes only when `semantic_audit.json` reports `ok: true`, no errors, the exact target
episode count, one uniform horizon for every state/action/camera stream, and matching embodiment
contract and task-profile hashes. This timeout-only result is source eligibility, not evidence of
closed-loop manipulation success.

## Fine-tune and run the open-loop gate

This workflow fine-tunes GR00T for the Vega/Dexmate Sharpa task using joint positions. Each state
contains 58 joint positions for both arms and hands, and each action contains 58 absolute
joint-position targets. Demonstrations include front and wrist camera streams recorded at 20 FPS.

```bash
./run_e2e.sh finetune \
  --epochs 5 \
  --global-batch-size 32
```

Conversion runs in the Robotic Grounding container. Statistics, GR00T training, and open-loop
evaluation run in the configured external Isaac-GR00T Python 3.10 environment. The default base
model is `nvidia/GR00T-N1.7-3B`.

For `E` epochs, `N` episodes, `F` measured frames, action horizon `H`, and global batch size `B`,
the launcher trains for

```text
ceil(E * N * (F - H + 1) / B)
```

optimizer steps. Raw video frames are not optimizer steps.

The stage runs these gates in order:

1. Convert semantic HDF5 to the released LeRobot modality contract.
2. Generate finite statistics with the same modality config used by training and evaluation.
3. Audit exact HDF5, Parquet, video, state, action, episode, and frame equality.
4. Fine-tune and save a numbered, serveable checkpoint plus processor and trainer state.
5. Run Isaac-GR00T open-loop evaluation over every action key and save
   `open_loop/trajectory_0.png`.

The command prints the resolved checkpoint. `--checkpoint` does not skip or resume training; when
provided, it selects the checkpoint used by the open-loop gate after training.

Fine-tuning passes only when `dataset_audit.json` reports `ok: true`, the checkpoint is complete,
the open-loop log reports finite MSE and MAE for every action key, and the trajectory plot is
viewable. Open-loop evaluation validates data/model plumbing, not task success.

## Evaluate task success in closed loop

Use the checkpoint printed by fine-tuning. Keep one rendered environment for the first smoke,
then increase parallelism only after measuring capacity.

```bash
FINETUNED_CHECKPOINT=/absolute/path/to/checkpoint-N

# Required one-episode lifecycle, camera, and control smoke.
./run_e2e.sh evaluate \
  --checkpoint "${FINETUNED_CHECKPOINT}" \
  --episodes 1 \
  --num-envs 1 \
  --execution-length 4

# Final task-success measurement; use separately measured renderer-safe parallelism.
EVAL_EPISODES=100
RENDERER_SAFE_ENVS=1
./run_e2e.sh evaluate \
  --checkpoint "${FINETUNED_CHECKPOINT}" \
  --episodes "${EVAL_EPISODES}" \
  --num-envs "${RENDERER_SAFE_ENVS}" \
  --execution-length 4
```

Both commands use the same evaluator and policy settings. A smaller multi-episode run, such as 20
episodes, is an optional preliminary estimate before the final measurement; it is not a separate
evaluation mode or acceptance gate.

The evaluation horizon defaults to the measured source episode length. Override it with
`--evaluation-horizon` only when the evaluation protocol explicitly requires a different bounded
horizon.

Progress messages use an execution safety budget, not the episode frame count. The wrapper allows
`ceil(episodes / num_envs) * episode_horizon + 100` environment steps and exits as soon as all
requested episodes finish. The `episode_lengths` and `metric_sample_counts` fields in the result
JSON are the authoritative frame counts.

The one-episode gate must prove that post-warmup camera observations are nonblack, reset and
randomization behavior match the inference contract, only intended safety/timeout terminations are
active, and the episode reaches its expected horizon. Structured JSON must report task metrics
independently of termination reason.

`[DONE]` means the evaluation process completed and saved its result; it does not mean the policy
succeeded at the task. Check `completed_episodes`, `successful_episodes`, `success_rate`,
`episode_lengths`, and `termination_reasons` in `closed_loop/evaluation_<N>.json`. A completed run
with zero successful episodes is a valid evaluation execution but a failed task-success result.

The selected task profile—not timeout survival—defines closed-loop success. For lift-and-hold
profiles, the evaluator samples object height before every policy action, excludes reset-cleared
terminal observations, reports lift/hold metrics, and retains only successful videos. Probe the
retained MP4 files for codec, resolution, FPS, duration, and complete episode length, then inspect
their beginning, interaction, success moment, and end.

![Closed-loop GR00T observations and simulated Vega Sharpa policy rollout](figures/e2e_workflow/groot_closed_loop_evaluation.webp)

*Closed-loop GR00T evaluation: the policy completed 87 of 100 full cycles in simulation trained with 100 rollouts.*

## Sim-to-real validation with limited real data

The Vega/Dexmate Sharpa tissue-box task also validated that V2D simulation data can support
real-robot deployment with a small amount of real adaptation data. The successful recipe used the
same 58-D joint-position policy contract as the workflow above:

- **Simulation pretraining:** fine-tune GR00T N1.7 on 200 successful simulation rollouts from a V2D
  RL expert, with dynamics and visual randomization.
- **Limited real adaptation:** continue training for five epochs on 10 open-loop real-robot replays
  of one trajectory generated by the V2D RL expert. No human teleoperation data was used.

The real-robot results show the value of the simulation-pretrained initialization:

| Initialization and adaptation | Real-robot result |
|---|---:|
| V2D Sim200 pretraining, then Real10 for 5 epochs | **4/5** |
| Raw GR00T N1.7, then Real10 for 5 epochs | **0/5** |
| Raw GR00T N1.7, then Real30 for 5 epochs | **0/5** |

| Successful recipe: V2D Sim200 → Real10 | Comparison failure: Raw GR00T N1.7 → Real10 |
|:---:|:---:|
| <img src="figures/e2e_workflow/v2d_sim200_real10_success.webp" alt="Successful V2D Sim200 to Real10 real-robot trial" width="320"> | <img src="figures/e2e_workflow/raw_groot_real10_failure.webp" alt="Raw GR00T N1.7 to Real10 real-robot failure" width="320"> |

A trial counted as successful when the robot lifted the box above the predefined height and placed
it stably back on the support within the rollout timeout. On this task, simulation pretraining
enabled repeated end-to-end completion with one-third as many real episodes as the largest
real-only run, which did not complete the task. This is a task-specific validation rather than a
general sim-to-real benchmark.

A separate co-training check found that simply mixing simulation and real episodes was not a
reliable recipe: Sim100 + Real10 and Sim100 + Real20 completed 0/5 real trials, while Sim100 +
Real30 showed a qualitative sign of success. We believe this is due to the visual gap between
simulation and reality, so the sim-to-real sampling balance matters.

## Artifacts and acceptance gates

| Path | Required evidence |
|---|---|
| `e2e_config.json` | Stage-owned inputs and resolved workflow values |
| `run_manifest.json` | Commands, runtime-source hash, Git revision, status, attempts, and failures |
| `logs/` | Complete per-attempt stdout/stderr |
| `reconstruction/result_slam_gravity_aligned/` | Imported or reconstructed source bundle |
| `vega_retarget_qa/` | Retarget report, IK sidecar, and full-motion video |
| `human_motion_data/ego_recon/processed/` | Partitioned Vega motion |
| `train_expert/vega/logs/rsl_rl/vega_sharpa_whole_body/` | Main-path source-expert runs and checkpoints; the shortcut records the ONNX path and hash instead |
| `contracts/` | Snapshotted embodiment contract and task profile |
| `source/pilot/manifest.json` | Pilot denominator, timeout-eligible count, horizon, and diversity |
| `source/manifest.json` | Full camera-free collection wave |
| `selected/manifest.json` | Exact timeout-eligible source selection |
| `recording/data.h5` | Semantic three-camera demonstrations |
| `semantic_audit.json` | Exact selected/HDF5 contract gate |
| `gr00t_dataset/meta/stats.json` | Finite statistics for the released modality config |
| `gr00t_dataset/` and `dataset_audit.json` | Exact LeRobot Parquet/video/data audit |
| `finetune/checkpoint-*/` | Model shards, processor, statistics, and trainer state |
| `open_loop/trajectory_0.png` | Standard open-loop plot; finite metrics are in the stage log |
| `closed_loop/evaluation_<N>.json` | Task metrics, denominator, terminations, and provenance |
| `closed_loop/success_<N>/` | Retained successful episode videos |

Do not report workflow completion until source-expert training and validation, expert
demonstration collection, the exact dataset audit, a complete GR00T checkpoint, open-loop
evaluation, task-level closed-loop evaluation, and requested successful videos have all passed
independently. When using the tissue-box ONNX shortcut, its recorded validation replaces only the
source-expert training gate.

## Status, resume, and intentional reruns

```bash
./run_e2e.sh status
./run_e2e.sh status --json
```

A completed stage resumes only when its commands, declared inputs, required outputs, and runtime
source hash still match. The manifest records the Git revision for provenance; documentation and
test-only edits do not invalidate runtime stages. A failed substage preserves its earlier attempts,
and a later invocation continues through the remaining stage sequence.

If a completed stage's inputs, command, or runtime implementation changed, use a new run root or
rerun intentionally with `--force`. `--no-resume` also executes matching completed stages instead
of skipping them. Review the dry-run and artifact replacement behavior before either form of
intentional rerun.

Managed evaluation stops its server and simulator processes on exit. Preserve the run root and its
logs for provenance, and remove only artifacts that belong to the run being cleaned up.
