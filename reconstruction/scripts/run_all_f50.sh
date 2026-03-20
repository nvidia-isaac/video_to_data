#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/.."

# python experiments/wooden_spatula_f50.py
# python experiments/electric_drill_toy_f50.py
# python experiments/dust_brush_f50.py
python experiments/airplane_f50.py
python experiments/yellow_spray_f50.py
