#!/bin/bash
# Render 3x3 (9-env) one-episode grids for the BEST model, 4 seeds, success/fail marked, 1200-step budget.
# Keeps all 4 videos (videos/run_grid3x3_seed{1..4}.mp4). Args: [GPU] [PORT]
set -u
cd /home/cning/simtoolreal_isaaclab
GROOT=/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/activate
ISAAC=/home/cning/isaaclab/env_isaaclab/bin/activate
SERVER=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_server.py
CLIENT=/home/cning/simtoolreal_isaaclab/scripts/render_run_grid.py
CKPT=/home/cning/simtoolreal_isaaclab/logs/gr00t_specialist/hammer_allpert_nojv_100k/hammer_allpert_nojv_100k.pt
GPU=${1:-1}; PORT=${2:-5603}
PROG=logs/grid3x3_sweep.log; : > "$PROG"
fuser -k ${PORT}/tcp 2>/dev/null; sleep 1
( source "$GROOT" && CUDA_VISIBLE_DEVICES=$GPU python "$SERVER" --checkpoint "$CKPT" --port $PORT > /tmp/g3_srv.log 2>&1 ) & SV=$!
for _ in $(seq 1 120); do grep -q "listening on" /tmp/g3_srv.log 2>/dev/null && break; sleep 2; done
grep -q "listening on" /tmp/g3_srv.log || { echo "SERVER FAILED" >> "$PROG"; tail -15 /tmp/g3_srv.log >> "$PROG"; kill -9 $SV 2>/dev/null; echo GRID3X3_DONE >> "$PROG"; exit 1; }
for SEED in 1 2 3 4; do
  ( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES \
    ./isaaclab.sh -p "$CLIENT" --headless --num_envs 9 --no_joint_vel --table_dist 0.15 \
      --max_ep_steps 1200 --max_steps 1400 --seed $SEED --port $PORT \
      --out /home/cning/simtoolreal_isaaclab/videos/run_grid3x3_seed${SEED}.mp4 > /tmp/g3_cli_${SEED}.log 2>&1 )
  echo "seed ${SEED}: $(grep -oE '[0-9]+/9 envs succeeded' /tmp/g3_cli_${SEED}.log | tail -1)" >> "$PROG"
done
kill -9 $SV 2>/dev/null; fuser -k ${PORT}/tcp 2>/dev/null
echo GRID3X3_DONE >> "$PROG"; cat "$PROG"
