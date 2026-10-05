#!/usr/bin/env bash
# Run the two checkpoint-only analyses in one next-eval allocation.
set -euo pipefail
command_file="$(dirname "$0")/example_commands.sh"
bash "$command_file" task-vectors
bash "$command_file" sign-conflicts
