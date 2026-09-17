---
name: groot-finetune
description: Run and extend embodiment-aware GR00T N1.7 post-training workflows from successful robot-policy collection through semantic recording, LeRobot conversion, statistics, fine-tuning, open-loop validation, task-level closed-loop evaluation, and success-only recordings. Use for GR00T or VLA fine-tuning, policy-to-data generation, floating-hand Sharpa, Vega/Dexmate Sharpa, new robot embodiments, modality/action contracts, exact successful-episode filtering, or reproducible closed-loop evaluation.
---

# Run GR00T Fine-Tuning

Own the requested workflow through validated artifacts. Execute stages when the user asks for a
run; do not stop after printing commands. Keep source-data success, pipeline validity, and
fine-tuned task success as three separate results.

## Resolve an embodiment contract

Resolve the contract from task registration, environment configuration, HDF5 data, converter,
modality config, and the registered closed-loop adapter. Never infer it from a robot name alone.

Record these fields before launching expensive work:

```text
contract ID
record task and inference task
motion/scene source
record observation terms
state layout, units, transforms, and dimension
absolute action layout, order, semantics, and dimension
camera term -> video modality mapping
modality config path and closed-loop adapter
source-success definition
task-success evaluator and thresholds
episode horizon and FPS
reset, warmup, curriculum, and randomization behavior
safe camera-free and rendered parallelism
```

Read the matching reference:

- Floating-hand Sharpa: [references/floating-hand.md](references/floating-hand.md)
- Vega/Dexmate Sharpa: [references/vega-sharpa.md](references/vega-sharpa.md)
- Contract schema and invariants: [references/embodiment-contract.md](references/embodiment-contract.md)
- New embodiment work: [references/adding-embodiment.md](references/adding-embodiment.md)

If no contract matches, stop before collection and follow the new-embodiment procedure. Never
silently fall back to a floating-hand or Vega layout.

## Keep runtimes explicit

| Stage | Runtime |
|---|---|
| Checkpoint resolution | host or IsaacLab container |
| Source rollout, record, and rerender | `robotic_grounding` IsaacLab container |
| HDF5 to LeRobot conversion | `robotic_grounding` container |
| Statistics, training, open-loop evaluation | external Isaac-GR00T Python 3.10 environment |
| GR00T server | external Isaac-GR00T environment |
| Closed-loop client | IsaacLab container over host-network ZMQ |

Before using a container, inspect its mounts and prove it contains the current worktree. Do not
reuse a similarly named container mounted from another checkout. Keep long-running server and
simulator processes monitored and clean them up on success or failure.

## Follow the gated workflow

### 1. Preflight and plan

Record the branch, commit, worktree status, checkpoint source and SHA-256, motion source, seeds,
GPU, disk, container/image, external GR00T checkout, and resolved contract. Use one unique output
root and preserve commands and metrics in a run report.

Install or verify `pyzmq` and `msgpack-numpy` in the IsaacLab interpreter. Verify statistics and
training dependencies in the external GR00T environment.

Use the generic planner rather than estimating collection size or optimizer steps manually:

```bash
python -m groot_finetune.tools.plan_groot_run \
  --target-successes 100 \
  --measured-success-rate 0.45 \
  --collection-safety-factor 1.125 \
  --episodes 100 \
  --frames-per-episode 699 \
  --action-horizon 16 \
  --epochs 5 \
  --global-batch-size 32
```

Supply camera capacity only when it has been measured for the active renderer. Treat camera-free
collection and rendered evaluation as different capacity classes.

### 2. Collect and select timeout-eligible source episodes

Choose the source route declared by the contract:

- Native source rollout: run the registered policy in its dynamics environment.
- Native semantic rerender: use `rerender_demo_visuals.py`.
- External rollout: validate its manifest and use `replay_record.py`.

Run the planned number of camera-free environments in one parallel wave when the simulator
supports it. Do not replace a requested large wave with repeated small batches without reporting
the reason. If a measured eligibility rate is unavailable, run a small pilot, report its denominator,
then plan the full wave.

Source eligibility is distinct from closed-loop task success. For the joint-rollout route, an
episode is eligible only when it reaches its timeout. Select deterministically with:

```bash
python -m groot_finetune.tools.select_successful_episodes \
  --input <EXPORT_DIR_1> \
  --input <EXPORT_DIR_2> \
  --output <OUTPUT_ROOT>/selected \
  --target 100 \
  --expected-frames <CONTRACT_HORIZON>
```

Fail if successful inputs are insufficient, any selected episode has inconsistent time-series
lengths, or the success marker is false. Preserve source paths and counts in the selection
manifest.

### 3. Record or rerender semantic demonstrations

Write the semantic HDF5 contract:

```text
data/demo_i/obs/<semantic_term>
data/demo_i/actions
```

Do not substitute an unrelated flat-policy recording format. Replay only selected successful
episodes. Enable the contract's training visual randomization unless the user explicitly requests
a diagnostic no-randomization run.

Require:

- exact requested demo count;
- exact contract horizon for actions, state terms, and every camera;
- finite numeric arrays and expected action dimension;
- RGB `(T,H,W,3)` `uint8`;
- source/replay state and action agreement within the route's tolerance;
- nonblank, correctly mounted sample frames from every camera.

### 4. Convert and audit LeRobot data

Run the converter from `robotic_grounding/`:

```bash
python -m groot_finetune.convert_to_gr00t \
  --input <RECORDING>/data.h5 \
  --output <OUTPUT_ROOT>/gr00t_dataset \
  --contract <EMBODIMENT_CONTRACT> \
  --task-profile <TASK_PROFILE>
```

The converter exports exactly the cameras declared by the embodiment contract. Verify that
state/action slice order, camera keys, transforms, and dimensions match the config and
closed-loop adapter.

Audit the complete artifact chain:

```bash
python -m groot_finetune.tools.audit_groot_run \
  --root <OUTPUT_ROOT> \
  --episodes <N> \
  --frames <CONTRACT_HORIZON> \
  --contract <EMBODIMENT_CONTRACT> \
  --task-profile <TASK_PROFILE> \
  --selected-export selected \
  --hdf5 recording/data.h5 \
  --dataset gr00t_dataset
```

The audit must confirm exact HDF5, Parquet, and per-video frame equality. Do not accept merely
nonzero or approximately matching lengths.

### 5. Generate statistics and fine-tune

Use the same modality config for statistics, training, and open-loop evaluation. Compute
`--max-steps` from usable samples, action horizon, epochs, and global batch size with the planner.
Do not equate raw video frames with usable action-window samples.

Confirm `meta/stats.json` before training and the requested checkpoint afterward. Record consumed
samples, optimizer steps, losses, runtime, model shards, processor, statistics, and trainer state.

### 6. Gate on open-loop evaluation

Evaluate every action modality key in the order declared by `meta/modality.json`. Require finite
MSE/MAE and save the plot. Treat this as a data/model plumbing gate, not task-success evidence.
Diagnose a failed open-loop gate before spending time on closed-loop evaluation.

### 7. Evaluate task success in closed loop

Use the inference task, reset behavior, visual mode, and embodiment contract together with the
task profile's evaluator. A free-running policy must not inherit source trajectory-deviation
terminations.

Require a one-episode smoke before a large evaluation:

- first post-warmup camera observations are nonblack;
- reset frame, finger state, curriculum, and randomization match the contract;
- only intended safety/timeout terminations are active;
- the episode reaches its expected horizon;
- structured JSON reports task metrics independently of termination.

Use the lifecycle wrapper for full evaluation:

```bash
bash robotic_grounding/groot_finetune/closed_loop/run_eval.sh \
  --gr00t-dir <ISAAC_GR00T_DIR> \
  --model <CHECKPOINT> \
  --container <ISAAC_CONTAINER> \
  --client-workdir <ROBOTIC_GROUNDING_PATH_IN_CONTAINER> \
  --task <INFERENCE_TASK> \
  --contract <EMBODIMENT_CONTRACT_IN_CONTAINER> \
  --task-profile <TASK_PROFILE_IN_CONTAINER> \
  --motion-file <MOTION_DIR_IN_CONTAINER> \
  --output-json <EVAL_JSON_IN_CONTAINER> \
  --episodes 100 \
  --num-envs <RENDERER_SAFE_ENVS> \
  --episode-horizon <CONTRACT_HORIZON>
```

Pass additional client arguments through `--client-extra-arg`. Timeout survival is source
eligibility, never closed-loop manipulation success.

### 8. Retain successful recordings

Record complete episodes, decide success using the same task evaluator, discard failed frames,
and retain only requested cameras and successful rollouts. For supported lift-and-hold tasks use
`--success-video-dir`, `--success-video-camera`, and `--max-success-videos` with the lifecycle
wrapper.

Probe every retained MP4 for codec, resolution, FPS, duration, and exact contract frame count.
Visually inspect the beginning, interaction, success moment, and end.

### 9. Report and clean up

Report:

- source input/success/failure/selected counts and criterion;
- exact frame counts through every data representation;
- dataset keys, dimensions, views, and statistics;
- checkpoint path and training metrics;
- open-loop MSE/MAE;
- closed-loop denominator, task successes, rate, partial-progress distribution, and termination
  counts;
- successful recording paths;
- any requested parallelism that was reduced and the measured capacity error.

Stop servers and isolated containers. Preserve intentional run artifacts and do not delete
unrelated user files.

## Validation

Run repository contract tests for every affected embodiment, then:

```bash
python "${CODEX_HOME:-$HOME/.codex}/skills/.system/skill-creator/scripts/quick_validate.py" \
  .codex/skills/groot-finetune
bash -n robotic_grounding/groot_finetune/closed_loop/run_eval.sh
PYTHONPYCACHEPREFIX=/tmp/groot-finetune-pycache \
  python -m py_compile robotic_grounding/groot_finetune/tools/*.py
git diff --check
```

Completion requires audited data, statistics, checkpoint, open-loop result, task-level
closed-loop result, viewable successful recordings when requested, and clean process shutdown.
