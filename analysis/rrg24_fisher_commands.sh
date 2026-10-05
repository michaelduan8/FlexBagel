#!/usr/bin/env bash
# Submit this command file again to resume Fisher without rerunning checkpoint methods.
set -euo pipefail
exec bash "$(dirname "$0")/rrg24_example_commands.sh" fisher
