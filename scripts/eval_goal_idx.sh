#!/bin/bash
# #32: Does the goal ADVANCE when met during eval? Run two state-specialist policies with
# --log_goal_idx (logs base.successes = the goal/trajectory index): the +goal model (0% eval)
# and the no-goal model (14% eval). If the goal advances for the no-goal policy but stays pinned
# at 0 for the +goal policy, the +goal policy stalls. One Isaac Sim at a time, GPU-gated.
set -u
cd /home/cning/simtoolreal_isaaclab
GROOT=/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/activate
ISAAC=/home/cning/isaaclab/env_isaaclab/bin/activate
SERVER=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_server.py
CLIENT=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_client.py
PORT=5602
PROG=logs/eval_goal_idx.log; : > "$PROG"
gpu_wait(){ for _ in $(seq 1 800); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0; sleep 15; done; }

run_one(){   # $1=label  $2=checkpoint  $3=extra client flags (e.g. --with_goal)
  local LABEL="$1" CKPT="$2" EXTRA="$3"
  [ -f "$CKPT" ] || { echo "$LABEL: MISSING $CKPT" >> "$PROG"; return; }
  gpu_wait; fuser -k ${PORT}/tcp 2>/dev/null; sleep 2
  ( source "$GROOT" && python "$SERVER" --checkpoint "$CKPT" --port $PORT > /tmp/gi_srv.log 2>&1 ) & SV=$!
  for _ in $(seq 1 120); do grep -q "listening on" /tmp/gi_srv.log 2>/dev/null && break; sleep 2; done
  grep -q "listening on" /tmp/gi_srv.log || { echo "$LABEL: SERVER FAILED" >> "$PROG"; tail -15 /tmp/gi_srv.log >> "$PROG"; kill -9 $SV 2>/dev/null; return; }
  echo "$(date +%H:%M) $LABEL: server up, running client ($EXTRA --log_goal_idx)" >> "$PROG"
  ( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
    ./isaaclab.sh -p "$CLIENT" --headless --num_envs 25 --episodes 50 --replan 1 $EXTRA --log_goal_idx --port $PORT \
    > /tmp/gi_cli_${LABEL}.log 2>&1 )
  echo "--- $LABEL ---" >> "$PROG"
  grep -E "max goal index reached this run|success_rate" /tmp/gi_cli_${LABEL}.log | tail -3 >> "$PROG"
  # also a few periodic goal_idx samples to see the progression
  grep -E "goal_idx now" /tmp/gi_cli_${LABEL}.log | tail -4 >> "$PROG"
  kill -9 $SV 2>/dev/null; fuser -k ${PORT}/tcp 2>/dev/null; sleep 3
}

run_one "goal"   logs/gr00t_specialist/hammer_goal/hammer_goal.pt           "--with_goal"
run_one "nogoal" logs/gr00t_specialist/hammer_str_tele/hammer_str_tele.pt   ""
echo "GOAL_IDX_DONE" >> "$PROG"; cat "$PROG"
