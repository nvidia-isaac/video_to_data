#!/bin/bash
# Eval ONE state-specialist checkpoint with goal-index logging. Args: CKPT LABEL EXTRA_CLIENT_FLAGS [EPISODES]
# State server (GR00T venv) + env client (isaaclab, --replan 1 --log_goal_idx). One Isaac Sim, GPU-gated.
set -u
cd /home/cning/simtoolreal_isaaclab
GROOT=/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/activate
ISAAC=/home/cning/isaaclab/env_isaaclab/bin/activate
SERVER=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_server.py
CLIENT=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_client.py
CKPT="$1"; LABEL="$2"; EXTRA="${3:-}"; EPS="${4:-100}"; GPU="${5:-0}"; PORT="${6:-5602}"
PROG=logs/eval_${LABEL}.log; : > "$PROG"
[ -f "$CKPT" ] || { echo "$LABEL: MISSING $CKPT" >> "$PROG"; echo "EVAL_${LABEL}_DONE" >> "$PROG"; exit 1; }
fuser -k ${PORT}/tcp 2>/dev/null; sleep 2
( source "$GROOT" && CUDA_VISIBLE_DEVICES=$GPU python "$SERVER" --checkpoint "$CKPT" --port $PORT > /tmp/es_srv_${LABEL}.log 2>&1 ) & SV=$!
for _ in $(seq 1 120); do grep -q "listening on" /tmp/es_srv_${LABEL}.log 2>/dev/null && break; sleep 2; done
grep -q "listening on" /tmp/es_srv_${LABEL}.log || { echo "$LABEL: SERVER FAILED" >> "$PROG"; tail -15 /tmp/es_srv_${LABEL}.log >> "$PROG"; kill -9 $SV 2>/dev/null; echo "EVAL_${LABEL}_DONE" >> "$PROG"; exit 1; }
echo "$(date +%H:%M) $LABEL: server up on GPU $GPU, client (--replan 1 $EXTRA --log_goal_idx, $EPS eps)" >> "$PROG"
( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES \
  ./isaaclab.sh -p "$CLIENT" --headless --num_envs 25 --episodes $EPS --replan 1 $EXTRA --log_goal_idx --port $PORT \
  > /tmp/es_cli_${LABEL}.log 2>&1 )
grep -E "max goal index reached this run|success_rate" /tmp/es_cli_${LABEL}.log | tail -3 >> "$PROG"
grep -E "goal_idx now" /tmp/es_cli_${LABEL}.log | tail -3 >> "$PROG"
kill -9 $SV 2>/dev/null; fuser -k ${PORT}/tcp 2>/dev/null
echo "EVAL_${LABEL}_DONE" >> "$PROG"; cat "$PROG"
