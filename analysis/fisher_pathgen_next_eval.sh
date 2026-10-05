#!/usr/bin/env bash
set -euo pipefail
: "${FLEXBAGEL_PYTHON:?Use the next-eval PBS wrapper}"
FISHER_EXPERT=pathgen exec bash "$(dirname "$0")/fisher_commands.sh"
