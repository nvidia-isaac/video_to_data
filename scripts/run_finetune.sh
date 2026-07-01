#!/usr/bin/env bash
# Finetune the pretrained SimToolReal policy on the 043 cross-slot tighten task.
#
# Defaults baked into finetune.py / the cfg: screw-free + kinematic (fast), reward_tip (tip gate +
# proximity bonus), goal_noise (phase-scheduled), fixed-size success @ fixed 0.01 keypoint tolerance,
# tip-tolerance curriculum 10mm->2mm, pretrained_object_scale (2.5,0.75,0.75). Actor-only restore.
#
# Usage:
#   ./run_finetune.sh                          # 3072 envs, 3000 iters (defaults)
#   NUM_ENVS=1536 MAX_ITERS=1500 ./run_finetune.sh
#   ./run_finetune.sh --curriculum_interval 1000   # faster tip 10->2mm anneal (~1000 iters)
#   ./run_finetune.sh --learning_rate 5e-5     # any extra finetune.py flags pass through
#
# num_envs MUST be divisible by 6 (SAPG blocks); reduce if OOM.
# ~3000 iters paces the tip-tolerance curriculum (10mm->2mm); see --curriculum_interval to go faster.
set -uo pipefail

REPO=/home/cning/simtoolreal_isaaclab
ISAACLAB=/home/cning/isaaclab/IsaacLab
VENV=/home/cning/isaaclab/env_isaaclab/bin/activate
TASK=Isaac-SimToolReal-Screwdriver043-Direct-v0
NUM_ENVS="${NUM_ENVS:-3072}"
MAX_ITERS="${MAX_ITERS:-3000}"

# shellcheck disable=SC1090
source "$VENV"
cd "$ISAACLAB"

echo "[run_finetune] task=$TASK  num_envs=$NUM_ENVS  max_iterations=$MAX_ITERS  extra='$*'"
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p "$REPO/scripts/finetune.py" \
  --headless --task "$TASK" --num_envs "$NUM_ENVS" --max_iterations "$MAX_ITERS" "$@"

echo "[run_finetune] done."
echo "[run_finetune] checkpoints -> $REPO/logs/finetune/00_ft_screwdriver043/nn/*.pth"
echo "[run_finetune] evaluate with: $REPO/scripts/eval_finetuned.sh"
