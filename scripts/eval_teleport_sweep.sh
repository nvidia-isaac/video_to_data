#!/bin/bash
# Sequentially eval several step-checkpoints of the teleport retrain (one Isaac Sim at a time) to
# map success-vs-steps. Server (GR00T venv) + client (isaaclab venv) per checkpoint.
set -u
CKDIR=/home/cning/simtoolreal_isaaclab/logs/gr00t_specialist/hammer_str_tele
RES=/home/cning/simtoolreal_isaaclab/logs/eval_teleport_sweep.txt
GROOT=/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/activate
ISAAC=/home/cning/isaaclab/env_isaaclab/bin/activate
CLIENT=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_client.py
SERVER=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_server.py
echo "teleport step-checkpoint eval sweep (25 envs x 100 ep, replan 1)" > "$RES"
for STEP in "$@"; do
  CKPT="$CKDIR/hammer_str_tele_step${STEP}.pt"
  [ -f "$CKPT" ] || { echo "step $STEP: MISSING $CKPT" >> "$RES"; continue; }
  fuser -k 5602/tcp 2>/dev/null; sleep 2
  ( source "$GROOT" && python "$SERVER" --checkpoint "$CKPT" --port 5602 > /tmp/sw_server_$STEP.log 2>&1 ) &
  SV=$!
  for i in $(seq 1 90); do grep -q "listening on" /tmp/sw_server_$STEP.log 2>/dev/null && break; sleep 2; done
  ( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
    ./isaaclab.sh -p "$CLIENT" --headless --num_envs 25 --episodes 100 --replan 1 --port 5602 \
    > /tmp/sw_client_$STEP.log 2>&1 )
  R=$(grep "DONE:" /tmp/sw_client_$STEP.log | tail -1)
  echo "step $STEP: ${R:-NO_RESULT}" >> "$RES"
  kill -9 "$SV" 2>/dev/null; fuser -k 5602/tcp 2>/dev/null; sleep 3
done
echo "SWEEP DONE" >> "$RES"
cat "$RES"
