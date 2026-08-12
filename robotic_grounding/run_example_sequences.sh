#!/usr/bin/env bash
# Run the full retargeting pipeline (load -> [segment] -> [urdf] -> processed -> support -> vis)
# on the v0.2 RL example sequences. See docs/EXAMPLE_SEQUENCES.md for the full list of
# sequences, prerequisites, and output layout.
#
# This script runs the pipeline in a SELF-CONTAINED example workspace,
#   $EXAMPLE_DIR  (default: $HMD/example_sequences)
# so the example inputs+outputs stay isolated from your main $HMD datasets. Everything
# the pipeline reads and writes for the examples lives under $EXAMPLE_DIR/<ds>/:
#   $EXAMPLE_DIR/<ds>/dataset/         raw data you download (input)
#   $EXAMPLE_DIR/<ds>/object_assets/   object meshes (input) + generated STLs/URDFs
#   $EXAMPLE_DIR/<ds>/<ds>_processed/  RL-ready motion parquets (output)
#   $EXAMPLE_DIR/<ds>/<ds>_html/        viser recordings + MP4 (vis output)
#
# PREREQUISITE: download each dataset FIRST, following its from-scratch guide in
#   docs/<DATASET>_SETUP.md   (arctic, hot3d, taco below), and lay it out under
#   $EXAMPLE_DIR/<ds>/. This script does NOT download anything.
# MANO models are shared from $MANO_DIR (mounted independently of the workspace).
#
# Usage (from robotic_grounding/):
#   HMD=~/datasets/human_motion_data ./run_example_sequences.sh
#   # optional: EXAMPLE_DIR=... (default $HMD/example_sequences), MANO_DIR=... (default $HMD/mano)
#
# Host orchestrator (Pattern A): each call spins the right Docker image per stage.
set -euo pipefail

HMD="${HMD:-$HOME/datasets/human_motion_data}"
EXAMPLE_DIR="${EXAMPLE_DIR:-$HMD/example_sequences}"   # self-contained example workspace
MANO_DIR="${MANO_DIR:-$HMD/mano}"                      # shared MANO (mounted separately)
cd "$(dirname "$0")"   # robotic_grounding/

echo "example workspace : $EXAMPLE_DIR   (mano: $MANO_DIR)"

# Optional knobs:
#   DRY_RUN=1   print the docker commands without running them (validate the pipeline)
#   ONLY=arctic only run this dataset's examples (arctic|hot3d|taco)
run() {  # run <dataset> <stages> <sequence_pattern> <label>
  local ds="$1" stages="$2" pat="$3" label="$4"
  if [ -n "${ONLY:-}" ] && [ "${ONLY}" != "${ds}" ]; then return 0; fi
  echo ""
  echo "============================================================"
  echo "[$ds] ${label}   (sequence_pattern: ${pat})"
  echo "============================================================"
  python scripts/run_pipeline_docker.py "$ds" \
    --hmd "$EXAMPLE_DIR" --mano-dir "$MANO_DIR" \
    --stages "$stages" --sequence-pattern "$pat" ${DRY_RUN:+--dry-run}
}

# --- ARCTIC (articulated objects -> NO 'urdf' stage; URDFs shipped via ArtiGrasp) ----------
run arctic "load,processed,support,vis" "s01_mixer_use_01"           "mixer_use (articulated)"
run arctic "load,processed,support,vis" "s07_box_grab_01"            "box_grab (single rigid)"
run arctic "load,processed,support,vis" "s01_espressomachine_use_01" "espresso_use (hand)"

# --- HOT3D ('segment' auto-inserts; seg025 is one atomic clip of source P0002_59a84a3a) ----
run hot3d  "load,urdf,processed,support,vis" "P0002_59a84a3a"        "P0002_59a84a3a -> seg025 (multiple objects)"

# --- TACO (rigid). Patterns are regex (re.search) over the derived sequence_id, where the
#     loader maps a triplet "(a b, c, d)" -> "a_b__c__d" (spaces/punct -> '_', via
#     _triplet_to_safe_id). Each example is pinned to one sequence by its full derived id
#     (triplet + capture id), so it matches exactly that sequence.
run taco   "load,urdf,processed,support,vis" "empty__cup__bowl_20231006_280"     "empty_cup_bowl / 20231006_280"
run taco   "load,urdf,processed,support,vis" "skim_off__spoon__pan_20230926_011" "skim_off_spoon_pan / 20230926_011"

echo ""
echo "=== done — RL-ready parquets under \$EXAMPLE_DIR/<ds>/<ds>_processed/sequence_id=.../robot_name=sharpa_wave/"
echo "           viser recordings + MP4 under \$EXAMPLE_DIR/<ds>/<ds>_html/ ==="
