# Experiments

Each experiment lives in its own subdirectory and is registered in `registry.yaml`.
The runner script dispatches experiments locally or to OSMO:

```bash
python robotic_grounding/scripts/run_experiment.py <id> --local
python robotic_grounding/scripts/run_experiment.py <id> --osmo
python robotic_grounding/scripts/run_experiment.py --list
```

---

## Directory layout

```
experiments/
  registry.yaml               # id -> directory mapping
  utils.py                    # shared helpers (build_train_command, make_entry_script, …)
  <experiment_dir>/
    config.yaml               # required — experiment definition
    workflow.py               # required for multi-task/sweep OSMO jobs
```

Only `example_single_run` is committed to the repo as a minimal smoke example.
Add your own experiment folders locally; they are gitignored (`experiments/exp*/`).

---

## Example experiments

### 1. Single run — `example_single`

**Directory:** `example_single_run/`

Trains on a single sequence, locally or with one OSMO task.
No `workflow.py` needed — `run_experiment.py` generates the workflow inline.

**config.yaml keys:**

| Key | Purpose |
|-----|---------|
| `motion_file` | Motion file path (e.g. `arctic_processed/arctic_s01_capsulemachine_grab_01/sharpa_wave`) |
| `train_overrides` | Hydra key=value overrides passed to `train.py` |
| `video` | Whether to record video |
| `osmo.build_image` | Whether to build+push the Docker image before submitting |

```bash
python robotic_grounding/scripts/run_experiment.py example_single --local
python robotic_grounding/scripts/run_experiment.py example_single --osmo
python robotic_grounding/scripts/run_experiment.py example_single --print-workflow  # preview entry script
```

---

## Adding a new experiment

1. Create `experiments/<your_exp_dir>/config.yaml` (copy `example_single_run/config.yaml` as a starting point).
2. If multi-task or sweep, add `workflow.py` with `generate_workflow(exp_id, config) -> str`.
3. Register it in `experiments/registry.local.yaml` (gitignored — create it if it doesn't exist):
   ```yaml
   # experiments/registry.local.yaml  — never committed
   myexp: <your_exp_dir>
   ```
   The runner merges this with `registry.yaml` at runtime, so all your private experiments
   are immediately available without touching any tracked file.
4. Run it:
   ```bash
   python robotic_grounding/scripts/run_experiment.py myexp --local --dry-run
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

Only `example_single_run` is tracked by git and committed to the repo.

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
2. Copy experiments/example_single_run/config.yaml as a template.
3. Create experiments/<exp_id>/config.yaml (and workflow.py if multi-task/sweep).
4. Add the id to experiments/registry.yaml under the "Real experiments" block.
5. Show the user the generated config and confirm before writing.

**launch** — submit to OSMO
1. Read experiments/registry.yaml to resolve the experiment directory.
2. Run: python robotic_grounding/scripts/run_experiment.py <exp_id> --print-workflow
   Show the output to the user and ask for confirmation.
3. On confirmation run: python robotic_grounding/scripts/run_experiment.py <exp_id> --osmo

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
