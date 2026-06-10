# Experiments

Each experiment lives in its own subdirectory and is registered in `registry.yaml`.
The runner script dispatches experiments locally or to OSMO:

```bash
python experiments/run_experiment.py <id> --local
python experiments/run_experiment.py <id> --osmo
python experiments/run_experiment.py --list
```

Commands assume the current directory is `robotic_grounding/`. From the repository root, use `python robotic_grounding/experiments/run_experiment.py ...`. Add `--dry-run` to preview local commands or generated OSMO submissions without launching work. Agents without OSMO/NGC setup should read [workflow/README.md](../workflow/README.md) before attempting cloud launch commands.

---

## Directory layout

```
experiments/
  registry.yaml               # id -> directory mapping
  utils.py                    # shared helpers (build_train_command, make_entry_script, …)
  <experiment_dir>/
    config.yaml               # required — experiment definition
    workflow.py               # optional; required for custom multi-task/sweep workflows
```

Only `example_*` directories are committed to the repo.
Add your own experiment folders locally; they are gitignored (`experiments/exp*/`).

---

## Example experiments

### Relative-object merge examples

These are the committed examples to use when validating the relative-object merge path:

| ID | Purpose |
|----|---------|
| `example_fixed_post` | Stage 2 FixedTimestepCurriculum with the stage-3 object-tracking boost in the tail. |
| `example_AC_post` | Adaptive custom VOC schedule with the stage-3 object-tracking boost in the same run. |
| `example_pre_fixed_post` | Pipeline wrapper that runs collision-free stage 1, then fixed-post stage 2/3. |

Preview their generated OSMO workflows before launching:

```bash
python experiments/run_experiment.py example_fixed_post --osmo --dry-run
python experiments/run_experiment.py example_AC_post --osmo --dry-run
python experiments/run_experiment.py example_pre_fixed_post --osmo --dry-run
```

Print the generated train entry script for a single experiment:

```bash
python experiments/run_experiment.py example_fixed_post --print-workflow
```

---

### 1. Single run — `example_single`

**Directory:** `example_single_run/`

Trains on a single sequence, locally or with one OSMO task.
No `workflow.py` needed — `run_experiment.py` generates the workflow inline.

**config.yaml keys:**

| Key | Purpose |
|-----|---------|
| `motion_file` | Motion file path (e.g. `arctic/arctic_processed/arctic_s01_capsulemachine_grab_01/sharpa_wave`) |
| `train_overrides` | Hydra key=value overrides passed to `train.py` |
| `video` | Whether to record video |
| `osmo.build_image` | Whether to build+push the Docker image before submitting |

```bash
python experiments/run_experiment.py example_single --local
python experiments/run_experiment.py example_single --osmo
python experiments/run_experiment.py example_single --print-workflow  # preview entry script
```

---

### 2. Sequence list — `example_sequences`

**Directory:** `example_sequence_list/`

Spawns one independent OSMO task per sequence in `osmo_multi_task.sequence_ids`.
Requires `workflow.py` with a `generate_workflow(exp_id, config) -> str` function. Newer single-stage configs can instead use top-level `sequences:`; for those, the runner generates one OSMO task per sequence without a custom `workflow.py`.

**Additional config.yaml keys:**

| Key | Purpose |
|-----|---------|
| `osmo_multi_task.sequence_ids` | List of sequence IDs to train on |
| `osmo_multi_task.run_name_suffix_template` | Run name template; `{sequence_id}` and `{short_name}` are interpolated |

**Local single-sequence run** via `--variant <short_name>` (requires `get_variant_overrides` in `workflow.py`):

```bash
python experiments/run_experiment.py example_sequences --osmo
python experiments/run_experiment.py example_sequences --local --variant capsulemachine
```

---

### 3. Parameter sweep — `example_sweep`

**Directory:** `example_param_sweep/`

Sweeps a grid of hyperparameter values, spawning one OSMO task per combination.
Requires `workflow.py` with `generate_workflow` and optionally `get_variant_overrides`.

**Additional config.yaml keys:**

| Key | Purpose |
|-----|---------|
| `osmo_multi_task.<param>_weights` | List of values for each swept parameter |

The example sweeps `contact_force.weight × action_l1.weight` (2×2 = 4 tasks):

```bash
python experiments/run_experiment.py example_sweep --osmo
python experiments/run_experiment.py example_sweep --local --variant cf1p0_aln5e-3
```

---

## Local run path without OSMO or W&B

Use this path when the agent only has the local container and local motion data. It does not submit OSMO jobs and does not require a W&B API key.

Preview the generated train command:

```bash
python experiments/run_experiment.py example_fixed_post --local --dry-run \
  --logger tensorboard \
  --num-envs 1 \
  --max-iterations 1
```

Run a one-iteration local train smoke test with TensorBoard logging:

```bash
python scripts/rsl_rl/train.py \
  --headless \
  --task Sharpa-V2P-v0 \
  --motion_file arctic/arctic_processed/arctic_s01_box_grab_01/sharpa_wave \
  --num_envs 1 \
  --max_iterations 1 \
  --logger tensorboard \
  --run_name smoke_train \
  --use_primitive_urdfs \
  agent.num_steps_per_env=8 \
  agent.save_interval=1
```

Evaluate the checkpoint created by that smoke test with the direct eval script:

```bash
CHECKPOINT=$(find logs/rsl_rl -path '*smoke_train*/model_*.pt' | sort -V | tail -1)
python scripts/rsl_rl/eval.py \
  --headless \
  --task Sharpa-V2P-v0 \
  --motion_file arctic/arctic_processed/arctic_s01_box_grab_01/sharpa_wave \
  --num_envs 1 \
  --checkpoint "$CHECKPOINT" \
  --eval_episodes 1 \
  --use_primitive_urdfs
```

For a dummy-agent asset check, use the commands in the top-level [README.md](../README.md). If the motion data is missing and OSMO is unavailable, stop and ask for the local dataset path or a prepared asset bundle.

---

## OSMO launch checklist

Use this section only after OSMO/NGC access is configured. The setup guide is [workflow/README.md](../workflow/README.md). Real OSMO training also expects `WANDB_API_KEY`; without it, use the local TensorBoard path above.

1. Confirm the experiment is registered:
   ```bash
   python experiments/run_experiment.py --list
   ```
2. Preview before launching:
   ```bash
   python experiments/run_experiment.py <id> --osmo --dry-run
   python experiments/run_experiment.py <id> --print-workflow
   ```
3. Submit with an existing image:
   ```bash
   export WANDB_API_KEY=<key>
   python experiments/run_experiment.py <id> --osmo \
     --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<tag> \
     --pool isaac-dev-l40s-04 \
     --priority NORMAL
   ```
4. If the selected image does not contain your branch changes, build and push through the runner:
   ```bash
   python experiments/run_experiment.py <id> --osmo --build-image \
     --image nvcr.io/nvstaging/isaac-amr/robotic-grounding:<tag>
   ```
5. Inspect the workflow:
   ```bash
   osmo workflow list
   osmo workflow logs <workflow-name>
   osmo workflow cancel <workflow-name>
   ```

When `osmo.motion_data_url` is set in `config.yaml`, the runner adds the OSMO dataset input and derives the mounted `--motion_file` path in the generated workflow. Do not manually rewrite those paths unless you are debugging the generator.

---

## Adding a new experiment

1. Create `experiments/<your_exp_dir>/config.yaml` (copy an example as a starting point).
2. For simple multi-sequence runs, prefer top-level `sequences:` in `config.yaml`. Add `workflow.py` only for custom multi-task or sweep generation with `generate_workflow(exp_id, config) -> str`.
3. Register it in `experiments/registry.local.yaml` (gitignored — create it if it doesn't exist):
   ```yaml
   # experiments/registry.local.yaml  — never committed
   myexp: <your_exp_dir>
   ```
   The runner merges this with `registry.yaml` at runtime, so all your private experiments
   are immediately available without touching any tracked file.
4. Run it:
   ```bash
   python experiments/run_experiment.py myexp --local --dry-run
   ```

---

## Keeping experiments out of git

By convention, experiment directories named `exp*` (e.g. `exp5_my_run/`, `exp12_sweep/`) are
gitignored and will never be accidentally committed. This is the recommended naming convention
for personal or in-progress experiments.

If you prefer a different naming scheme, add your own pattern to `.gitignore` at the repo root:

```gitignore
# Example: ignore all directories matching your own convention
experiments/run_*/
experiments/scratch_*/
experiments/zeol_*/
```

Only `example_*` directories are tracked by git and committed to the repo.

---

## Claude Code skill: experiment workflow

You can create a Claude Code skill to automate the create → launch → monitor loop.
Skills are prompt templates that Claude follows when you type a slash command.

### What a skill file looks like

Skills live in `.claude/skills/<skill-name>/SKILL.md`.
Two scopes are useful here:

| Scope | Path | Visible to |
|-------|------|-----------|
| Project | `.claude/skills/<name>/SKILL.md` | Everyone who clones the repo |
| Personal | `~/.claude/skills/<name>/SKILL.md` | You, across all projects |

A skill file has a YAML frontmatter block followed by a Markdown body that Claude reads as instructions:

```markdown
---
name: exp
description: Create, launch, or check a robotic-grounding experiment. Invoke with /exp.
argument-hint: "[create|launch|status] [exp_id]"
disable-model-invocation: true
---

The user invoked the experiment skill with: $ARGUMENTS

Follow the steps below based on the first argument:

**create** — scaffold a new experiment
1. Ask the user for: experiment id, type (single/sequences/sweep), and what to vary.
2. Copy the closest example from experiments/example_*/config.yaml as a template.
3. Create experiments/<exp_id>/config.yaml (and workflow.py if multi-task/sweep).
4. Add the id to experiments/registry.yaml under the "Real experiments" block.
5. Show the user the generated config and confirm before writing.

**launch** — submit to OSMO
1. Read experiments/registry.yaml to resolve the experiment directory.
2. Run: python experiments/run_experiment.py <exp_id> --print-workflow
   Show the output to the user and ask for confirmation.
3. On confirmation run: python experiments/run_experiment.py <exp_id> --osmo

**status** — check progress on W&B
1. Read the run_name from experiments/<exp_dir>/config.yaml.
2. Run: wandb runs list --project v2p_hands --filter "display_name~<run_name>"
   (or use `wandb status` if available) to show recent runs.
3. Summarise: run state, latest reward metrics, wall-clock time elapsed.
4. If a run looks stuck or crashed, suggest next steps.
```

### Installing the skill

```bash
# Project-scoped (shared with teammates):
mkdir -p .claude/skills/exp
# then write .claude/skills/exp/SKILL.md with the content above

# Personal (just for you, works in any project):
mkdir -p ~/.claude/skills/exp
# then write ~/.claude/skills/exp/SKILL.md
```

### Using the skill

```
/exp create exp47_my_ablation
/exp launch exp47_my_ablation
/exp status exp47_my_ablation
```

Claude will read the skill instructions and carry out each step, using the tools
available in the session (file reads, shell commands, etc.).

### Tips

- **`disable-model-invocation: true`** prevents Claude from auto-triggering the skill
  mid-conversation — you stay in control of when launches happen.
- Add `allowed-tools: Read Write Edit Bash` in the frontmatter to pre-approve the tools
  the skill needs so you aren't prompted for each one.
- You can split into three separate skills (`exp-create`, `exp-launch`, `exp-status`)
  if you want finer-grained control or different tool permissions per action.
- The `argument-hint` field populates the slash-command autocomplete in the Claude Code UI.
