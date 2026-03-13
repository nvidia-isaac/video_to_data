#!/usr/bin/env bash
set -euo pipefail

sudo apt update
sudo apt install -y git-lfs pipx
pipx ensurepath
pipx install pre-commit
chmod +x workflow/run.sh

echo "Done. You may need to restart your shell for pipx PATH changes."
