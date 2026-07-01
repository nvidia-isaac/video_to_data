#!/bin/bash
# Eval the 2-cam (table+wrist) image specialist: image server (GR00T venv) + env client (isaaclab,
# --wrist --replan 1). One Isaac Sim, GPU-gated. Writes the success rate to logs/eval_2cam.log.
set -u
cd /home/cning/simtoolreal_isaaclab
GROOT=/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/activate
ISAAC=/home/cning/isaaclab/env_isaaclab/bin/activate
CKPT=/home/cning/simtoolreal_isaaclab/logs/gr00t_specialist/hammer_2cam/hammer_2cam.pt
SERVER=/home/cning/simtoolreal_isaaclab/scripts/eval_specialist_server.py
CLIENT=/home/cning/simtoolreal_isaaclab/scripts/eval_specialist_client.py
PORT=5599
PROG=logs/eval_2cam.log; : > "$PROG"
gpu_wait(){ for _ in $(seq 1 800); do [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)" ] && return 0; sleep 15; done; }
gpu_wait
fuser -k ${PORT}/tcp 2>/dev/null; sleep 2
( source "$GROOT" && python "$SERVER" --checkpoint "$CKPT" --port $PORT > /tmp/e2c_srv.log 2>&1 ) & SV=$!
for _ in $(seq 1 180); do grep -q "listening on" /tmp/e2c_srv.log 2>/dev/null && break; sleep 2; done
grep -q "listening on" /tmp/e2c_srv.log || { echo "SERVER FAILED" >> "$PROG"; tail -20 /tmp/e2c_srv.log >> "$PROG"; kill -9 $SV 2>/dev/null; exit 1; }
echo "$(date +%H:%M) server up; starting client (--wrist --replan 1)" >> "$PROG"
( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && OMNI_KIT_ACCEPT_EULA=YES \
  ./isaaclab.sh -p "$CLIENT" --headless --num_envs 25 --episodes 100 --replan 1 --wrist --port $PORT \
  > /tmp/e2c_cli.log 2>&1 )
R=$(grep -oE "success_rate [0-9.]+%" /tmp/e2c_cli.log | tail -1)
echo "$(date +%H:%M) 2cam (table+wrist, replan 1) = ${R:-NA}" >> "$PROG"
grep -E "\[client\] DONE" /tmp/e2c_cli.log >> "$PROG"
kill -9 $SV 2>/dev/null; fuser -k ${PORT}/tcp 2>/dev/null
echo "EVAL_2CAM_DONE" >> "$PROG"; cat "$PROG"
