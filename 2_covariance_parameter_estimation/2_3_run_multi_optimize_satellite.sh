#!/bin/bash --login
#SBATCH --account=pawsey0106
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=3G
#SBATCH --time=03:00:00
#SBATCH --array=0-1  # Placeholder, will exit if index is too high
#SBATCH --output=log/job_output_%A_%a.out
#SBATCH --error=log/job_error_%A_%a.log

# Activate virtual environment — use full path or ensure MYSOFTWARE is defined
VENV_PATH=".venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtualenv not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"

declare -A TIME_MAP

TIME_MAP["time_24h"]="2023-09-22T04:00:00 2023-12-18T01:00:00"

# Build all combinations
COMBINATIONS=()

for label in "${!TIME_MAP[@]}"; do
    steps=(${TIME_MAP[$label]})  # Split the space-separated string into an array
    for step_index in "${!steps[@]}"; do
        step="${steps[$step_index]}"
        COMBINATIONS+=("$step $step_index")
    done
done

Bounds check for SLURM_ARRAY_TASK_ID
if [[ -z "$SLURM_ARRAY_TASK_ID" ]] || [[ -z "${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}" ]]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID). Max index: $((${#COMBINATIONS[@]} - 1))."
    exit 1
fi

# Extract combination
combo="${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"
read time step_index <<< "$combo"

echo "Running with: time=$time, index=$step_index"

# Run your optimization script
.venv/bin/python 2_covariance_parameter_estimation/sbatch_scripts/optimize_satellite.py "$time" "$step_index"
