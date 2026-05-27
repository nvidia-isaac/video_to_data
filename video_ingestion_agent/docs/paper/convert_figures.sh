#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-4.0 AND Apache-2.0
# Convert SVG figures from docs/images/ to PDF for LaTeX inclusion.
# Requires one of: inkscape, rsvg-convert, or cairosvg (Python).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMG_DIR="$SCRIPT_DIR/../images"
OUT_DIR="$SCRIPT_DIR"

for svg in "$IMG_DIR"/*.svg; do
    base="$(basename "$svg" .svg)"
    pdf="$OUT_DIR/$base.pdf"
    echo "Converting $base.svg -> $base.pdf"

    if command -v inkscape &>/dev/null; then
        inkscape "$svg" --export-type=pdf --export-filename="$pdf"
    elif command -v rsvg-convert &>/dev/null; then
        rsvg-convert -f pdf -o "$pdf" "$svg"
    elif python3 -c "import cairosvg" 2>/dev/null; then
        python3 -c "import cairosvg; cairosvg.svg2pdf(url='$svg', write_to='$pdf')"
    elif [ -f "$SCRIPT_DIR/../../.venv/bin/python" ]; then
        "$SCRIPT_DIR/../../.venv/bin/python" -c "import cairosvg; cairosvg.svg2pdf(url='$svg', write_to='$pdf')"
    else
        echo "ERROR: No SVG-to-PDF converter found."
        echo "Install one of: inkscape, librsvg2-bin, or pip install cairosvg"
        exit 1
    fi
done

echo "Done. PDFs written to $OUT_DIR"
