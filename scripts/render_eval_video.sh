#!/bin/bash
# Render a tiled-grid video of a state-specialist eval rollout (all envs, full run).
# Args: CKPT LABEL EXTRA_CLIENT_FLAGS [GPU] [VIDEO_MAX_FRAMES] [PORT]
set -u
cd /home/cning/simtoolreal_isaaclab
GROOT=/home/cning/simtoolreal_isaaclab/Isaac-GR00T/.venv/bin/activate
ISAAC=/home/cning/isaaclab/env_isaaclab/bin/activate
SERVER=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_server.py
CLIENT=/home/cning/simtoolreal_isaaclab/scripts/eval_simtoolreal_client.py
CKPT="$1"; LABEL="$2"; EXTRA="${3:-}"; GPU="${4:-0}"; VMAX="${5:-4000}"; PORT="${6:-5602}"
PROG=logs/render_${LABEL}.log; : > "$PROG"
[ -f "$CKPT" ] || { echo "MISSING $CKPT" >> "$PROG"; echo "RENDER_${LABEL}_DONE" >> "$PROG"; exit 1; }
fuser -k ${PORT}/tcp 2>/dev/null; sleep 2
( source "$GROOT" && CUDA_VISIBLE_DEVICES=$GPU python "$SERVER" --checkpoint "$CKPT" --port $PORT > /tmp/rv_srv_${LABEL}.log 2>&1 ) & SV=$!
for _ in $(seq 1 120); do grep -q "listening on" /tmp/rv_srv_${LABEL}.log 2>/dev/null && break; sleep 2; done
grep -q "listening on" /tmp/rv_srv_${LABEL}.log || { echo "SERVER FAILED" >> "$PROG"; tail -15 /tmp/rv_srv_${LABEL}.log >> "$PROG"; kill -9 $SV 2>/dev/null; echo "RENDER_${LABEL}_DONE" >> "$PROG"; exit 1; }
echo "$(date +%H:%M) $LABEL: server up GPU $GPU; rendering 100-ep 5x5 grid (vmax $VMAX, $EXTRA)" >> "$PROG"
( source "$ISAAC" && cd /home/cning/isaaclab/IsaacLab && CUDA_VISIBLE_DEVICES=$GPU OMNI_KIT_ACCEPT_EULA=YES \
  ./isaaclab.sh -p "$CLIENT" --headless --num_envs 25 --episodes 100 --replan 1 $EXTRA \
  --video --video_envs 25 --video_max_frames $VMAX --port $PORT > /tmp/rv_cli_${LABEL}.log 2>&1 )
grep -E "success_rate|wrote [0-9]+ grid frames" /tmp/rv_cli_${LABEL}.log | tail -3 >> "$PROG"
if [ -f videos/eval_simtoolreal_hammer.mp4 ]; then
  mv -f videos/eval_simtoolreal_hammer.mp4 videos/eval_${LABEL}_100ep.mp4
  echo "video -> videos/eval_${LABEL}_100ep.mp4 ($(du -h videos/eval_${LABEL}_100ep.mp4 | cut -f1))" >> "$PROG"
fi
kill -9 $SV 2>/dev/null; fuser -k ${PORT}/tcp 2>/dev/null
echo "RENDER_${LABEL}_DONE" >> "$PROG"; cat "$PROG"
