#!/usr/bin/env bash
# Submit this command file again to resume Fisher without rerunning checkpoint methods.
set -euo pipefail
# All snapshots are preflighted before submission; avoid remote shard resolution.
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec bash "$(dirname "$0")/example_commands.sh" fisher
