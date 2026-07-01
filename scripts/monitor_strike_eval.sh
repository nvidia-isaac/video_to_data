#!/bin/bash
# Continuously eval the strike-training checkpoints as they appear (~every 1k epochs): 100-episode
# nail_driven success rate, WITHOUT and WITH all perturbations, on GPU 1 (training uses GPU 0).
# Appends to logs/strike_eval/strike_eval.csv and refreshes logs/strike_eval/strike_eval.png.
set -u
REPO=/home/cning/simtoolreal_isaaclab
NN=$REPO/logs/simtoolreal/00_vega_right_strike/nn
EVALDIR=$REPO/logs/strike_eval
mkdir -p "$EVALDIR"
LAST="$EVALDIR/last_eval_ep.txt"; [ -f "$LAST" ] || echo 0 > "$LAST"
source /home/cning/isaaclab/env_isaaclab/bin/activate

run_eval() {  # $1=ckpt  $2=ep  $3=extra-flags
  ( cd /home/cning/isaaclab/IsaacLab && \
    CUDA_VISIBLE_DEVICES=1 OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p "$REPO/scripts/eval_strike_checkpoint.py" \
      --headless --checkpoint "$1" --step "$2" --episodes 100 $3 ) >> "$EVALDIR/eval_run.log" 2>&1
}
eval_ckpt() {  # $1=ckpt $2=ep  -> both conditions + replot
  echo "[monitor] $(date +%H:%M) evaluating ep=$2  $1" | tee -a "$EVALDIR/monitor.log"
  run_eval "$1" "$2" ""               # no perturbation
  run_eval "$1" "$2" "--perturbation" # all perturbations
  echo "$2" > "$LAST"
  python "$REPO/scripts/plot_strike_eval.py" >> "$EVALDIR/monitor.log" 2>&1
  echo "[monitor] $(date +%H:%M) done ep=$2" | tee -a "$EVALDIR/monitor.log"
}

while true; do
  best=""; bestep=0
  for c in "$NN"/last_*_ep_*.pth; do
    [ -e "$c" ] || continue
    ep=$(echo "$c" | grep -oP 'ep_\K[0-9]+')
    [ -n "$ep" ] && [ "$ep" -gt "$bestep" ] && { bestep=$ep; best=$c; }
  done
  laste=$(cat "$LAST")
  if [ -n "$best" ] && [ $((bestep - laste)) -ge 1000 ]; then
    eval_ckpt "$best" "$bestep"
  fi
  # stop once training has finished AND the final checkpoint is evaluated
  if ! pgrep -f "run_name 00_vega_right_strike" >/dev/null 2>&1; then
    laste=$(cat "$LAST")
    if [ -n "$best" ] && [ "$bestep" -gt "$laste" ]; then eval_ckpt "$best" "$bestep"; fi
    echo "[monitor] $(date +%H:%M) training finished; final eval done; exiting" | tee -a "$EVALDIR/monitor.log"
    break
  fi
  sleep 600
done
