#!/bin/bash
# Eval +goal state-specialist step-checkpoints (one Isaac Sim at a time, GPU-gated). --with_goal on
# the client (the env supplies goal_keypoints each step). Appends "label<TAB>success%" to a TSV.
set -u
cd /home/cning/simtoolreal_isaaclab
GROOT=/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/activate
ISAAC=/home/cning/isaaclab/env_isaaclab/bin/activate
CKDIR=/home/cning/simtoolreal_isaaclab/logs/gr00t_specialist/hammer_goal
SERVER=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_server.py
CLIENT=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_client.py
TSV=logs/eval_goal_sweep.tsv; PROG=logs/eval_goal_sweep.log
: > "$TSV"; : > "$PROG"
gpu_wait(){ for _ in $(seq 1 800); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0; sleep 15; done; }
for STEP in "$@"; do
  CKPT="$CKDIR/hammer_goal_step${STEP}.pt"; [ "$STEP" = final ] && CKPT="$CKDIR/hammer_goal.pt"
  [ -f "$CKPT" ] || { echo "step $STEP: MISSING" >> "$PROG"; continue; }
  gpu_wait; fuser -k 5602/tcp 2>/dev/null; sleep 2
  ( source "$GROOT" && python "$SERVER" --checkpoint "$CKPT" --port 5602 > /tmp/eg_srv.log 2>&1 ) & SV=$!
  for _ in $(seq 1 120); do grep -q "listening on" /tmp/eg_srv.log 2>/dev/null && break; sleep 2; done
  ( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
    ./isaaclab.sh -p "$CLIENT" --headless --num_envs 25 --episodes 100 --replan 1 --with_goal --port 5602 \
    > /tmp/eg_cli.log 2>&1 )
  R=$(grep -oE "success_rate [0-9.]+%" /tmp/eg_cli.log | tail -1 | grep -oE "[0-9.]+")
  echo -e "goal_${STEP}\t${R:-NA}" >> "$TSV"; echo "$(date +%H:%M) step $STEP = ${R:-NA}%" >> "$PROG"
  kill -9 $SV 2>/dev/null; fuser -k 5602/tcp 2>/dev/null; sleep 3
done
echo "SWEEP DONE" >> "$PROG"; cat "$TSV"
