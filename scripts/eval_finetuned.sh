#!/usr/bin/env bash
# Evaluate the finetuned 043 policy on 100 seeds with the PHYSICAL screw, and render the best-10 clips.
# Two-pass (OOM-safe): Pass 1 ranks by real screw rotation (no cameras); Pass 2 re-runs the SAME seeds
# and records only the top-10 per-env clips. Matches the eval metric the finetune trained on
# (fixed-size keypoints @ tol 0.01, object_scale 2.5/0.75/0.75 -- all via --demo + the cfg).
#
# Usage:
#   ./eval_finetuned.sh                       # latest checkpoint in logs/finetune
#   ./eval_finetuned.sh /path/to/model.pth    # a specific checkpoint
set -uo pipefail

REPO=/home/cning/simtoolreal_isaaclab
ISAACLAB=/home/cning/isaaclab/IsaacLab
VENV=/home/cning/isaaclab/env_isaaclab/bin/activate
DEPLOY="$REPO/scripts/deploy_pretrained.py"
OUT=/tmp/eval_ft; mkdir -p "$OUT"
DEST="$REPO/videos/best10_finetuned043"

# shellcheck disable=SC1090
source "$VENV"
cd "$ISAACLAB"

CKPT="${1:-$(ls -t "$REPO"/logs/finetune/*/nn/*.pth 2>/dev/null | head -1)}"
if [ -z "${CKPT:-}" ] || [ ! -f "$CKPT" ]; then
  echo "[eval] no checkpoint found -- pass one as an arg, or run run_finetune.sh first."; exit 1
fi
echo "[eval] checkpoint = $CKPT"

COMMON=(--headless --env screwdriver043 --checkpoint "$CKPT"
        --randomize_layout --demo --num_envs 102 --demo_task tighten_screw
        --steps 1600 --physical_screw --screw_friction 0.005 --screw_damping 0.005)

# ---- Pass 1: rank 100 seeds by screw rotation (no cameras -> fast, no OOM) ----
echo "[eval] Pass 1/2: rolling out + ranking (no video) ..."
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p "$DEPLOY" "${COMMON[@]}" > "$OUT/rank.log" 2>&1 || true
echo "----- ranking -----"
grep -E "FINAL|BEST_SUCCESS|TOP[0-9]+_ENVS|#  ?[0-9]+ env_" "$OUT/rank.log" | head -16 || true

# top-10 env indices, in rank order (by screw rotation)
TOP10=$(python3 -c "import re,sys;print(','.join(re.findall(r'#\s*\d+ env_(\d+):', open(sys.argv[1]).read())[:10]))" "$OUT/rank.log")
if [ -z "$TOP10" ]; then
  echo "[eval] Pass 1 produced no TOP10 (run failed?). See $OUT/rank.log"; exit 1
fi
echo "[eval] top-10 envs: $TOP10"

# ---- Pass 2: same seeds, record ONLY the top-10 clips (all cameras render; 10 encoded) ----
echo "[eval] Pass 2/2: recording top-10 clips ..."
rm -f "$REPO"/videos/per_env/*.mp4
OMNI_KIT_ACCEPT_EULA=YES ./isaaclab.sh -p "$DEPLOY" "${COMMON[@]}" \
  --per_env_cam --per_env_cam_record "$TOP10" \
  --cam_width 960 --cam_height 600 --cam_eye=-0.32,-0.58,0.92 --cam_lookat=0.10,0.0,0.61 \
  > "$OUT/record.log" 2>&1 || true

# ---- organize the clips into best10_finetuned043/, rank-named with screw deg + goals ----
rm -rf "$DEST"; mkdir -p "$DEST"
python3 - "$OUT/rank.log" "$REPO" "$DEST" <<'PY'
import re, os, shutil, sys
log, repo, dest = sys.argv[1], sys.argv[2], sys.argv[3]
rows = re.findall(r"#\s*\d+ env_(\d+): screw (\d+)deg \((\d+)/(\d+)", open(log).read())
for r, (e, deg, g, ng) in enumerate(rows[:10], 1):
    src = f"{repo}/videos/per_env/env_{int(e):03d}.mp4"
    if os.path.exists(src):
        shutil.copy(src, f"{dest}/rank{r:02d}_env{int(e):03d}_{deg}deg_{g}of{ng}.mp4")
        print(f"  rank{r:02d} env_{int(e):03d}: {deg} deg, {g}/{ng} goals")
PY
echo "[eval] best-10 clips -> $DEST"
echo "[eval] full ranking log -> $OUT/rank.log"
