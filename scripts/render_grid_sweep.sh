#!/bin/bash
# Render the 25-env one-episode grid for the BEST model across 4 seeds; keep the best (most successes).
# Two concurrent streams (GPU0/port5602: seeds 1,2 ; GPU1/port5603: seeds 3,4).
set -u
cd /home/cning/simtoolreal_isaaclab
GROOT=/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/activate
ISAAC=/home/cning/isaaclab/env_isaaclab/bin/activate
SERVER=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_server.py
CLIENT=/home/cning/simtoolreal_isaaclab/scripts/render_run_grid.py
CKPT=/home/cning/simtoolreal_isaaclab/logs/gr00t_specialist/hammer_allpert_nojv_100k/hammer_allpert_nojv_100k.pt
PROG=logs/grid_sweep.log; : > "$PROG"

stream(){  # $1=gpu $2=port  $3...=seeds
  local GPU=$1 PORT=$2; shift 2
  fuser -k ${PORT}/tcp 2>/dev/null; sleep 1
  ( source "$GROOT" && CUDA_VISIBLE_DEVICES=$GPU python "$SERVER" --checkpoint "$CKPT" --port $PORT > /tmp/gsw_srv_${PORT}.log 2>&1 ) & local SV=$!
  for _ in $(seq 1 120); do grep -q "listening on" /tmp/gsw_srv_${PORT}.log 2>/dev/null && break; sleep 2; done
  for SEED in "$@"; do
    ( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES \
      ./isaaclab.sh -p "$CLIENT" --headless --num_envs 25 --no_joint_vel --table_dist 0.15 \
        --seed $SEED --port $PORT --out /home/cning/simtoolreal_isaaclab/videos/run_grid_seed${SEED}.mp4 \
        > /tmp/gsw_cli_${SEED}.log 2>&1 )
    echo "seed ${SEED} (GPU$GPU): $(grep -oE '[0-9]+/25 envs succeeded' /tmp/gsw_cli_${SEED}.log | tail -1)" >> "$PROG"
  done
  kill -9 $SV 2>/dev/null; fuser -k ${PORT}/tcp 2>/dev/null
}

stream 0 5602 1 2 &
stream 1 5603 3 4 &
wait
/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/python - <<'PY' >> "$PROG"
import re, shutil
best = (-1, None)
for s in [1, 2, 3, 4]:
    try:
        m = re.search(r"(\d+)/25 envs succeeded", open(f"/tmp/gsw_cli_{s}.log").read())
        n = int(m.group(1)) if m else -1
    except Exception:
        n = -1
    print(f"seed {s}: {n}/25")
    if n > best[0]:
        best = (n, s)
if best[1] is not None:
    shutil.copy(f"videos/run_grid_seed{best[1]}.mp4", "videos/run_grid_best.mp4")
    print(f"BEST seed={best[1]} {best[0]}/25 -> videos/run_grid_best.mp4")
PY
echo GRID_SWEEP_DONE >> "$PROG"; cat "$PROG"
