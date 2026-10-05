#!/bin/bash -l
#PBS -A MatSciAI
#PBS -q capacity
#PBS -l select=1
#PBS -l walltime=18:00:00
#PBS -l filesystems=flare
#PBS -l place=scatter
#PBS -k doe
#PBS -j oe
set -euo pipefail
: "${COMMAND_SCRIPT:?Absolute command file required}"
: "${ANALYSIS_OUTPUT_DIR:?Isolated output directory required}"
[[ "$COMMAND_SCRIPT" == /* && -f "$COMMAND_SCRIPT" ]]
set +u
module reset
module load frameworks/2026.1.0
source /flare/MatSciAI/xinxil/setup.sh
set -u
unset CONDA_PREFIX PYTHONHOME
export VIRTUAL_ENV=/home/xinxil/venvs/flexolmo-next-eval
export PATH="$VIRTUAL_ENV/bin:${PATH:-/usr/bin:/bin}"
export FLEXBAGEL_PYTHON="$VIRTUAL_ENV/bin/python"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FISHER_ACTIVATION_CHECKPOINTING=1
cd /flare/MatSciAI/xinxil/codes/FlexBagel
printf 'PBS job: %s\nHost: %s\nCommand: %s\nOutput directory: %s\n' "$PBS_JOBID" "$(hostname)" "$COMMAND_SCRIPT" "$ANALYSIS_OUTPUT_DIR"
"$FLEXBAGEL_PYTHON" -c 'import torch, transformers; print("torch", torch.__version__, "transformers", transformers.__version__, "XPUs", torch.xpu.device_count())'
exec bash "$COMMAND_SCRIPT"
