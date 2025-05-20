#!/bin/bash --login
#SBATCH --account=pawsey0106
#SBATCH --partition=work
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=3G
#SBATCH --time=03:00:00
#SBATCH --array=0-800  # Placeholder, will exit if index is too high
#SBATCH --output=log/job_output_%A_%a.out
#SBATCH --error=log/job_error_%A_%a.log

# Activate virtual environment — use full path or ensure MYSOFTWARE is defined
VENV_PATH=".venv/bin/activate"
if [ ! -f "$VENV_PATH" ]; then
    echo "Error: Virtualenv not found at $VENV_PATH"
    exit 1
fi
source "$VENV_PATH"

# Define step/model combinations for each test_type
declare -A STEP_MAP
declare -A MODEL_MAP

STEP_MAP["time_24h"]="$(seq 0 86400 8640000)"
MODEL_MAP["time_24h"]="gos gprm_e gprm optimum"

STEP_MAP["time_1h"]="$(seq 0 3600 172800)"
MODEL_MAP["time_1h"]="gos gprm_e gprm optimum"

STEP_MAP["noise"]="$(seq 0 0.001 0.015)"
MODEL_MAP["noise"]="gos gprm_e gprm"

STEP_MAP["cloud_sparse"]="$(seq 0 0.03 0.75)"
MODEL_MAP["cloud_sparse"]="gprm_e gprm"

STEP_MAP["cloud_dense"]="$(seq 0 0.03 0.75)"
MODEL_MAP["cloud_dense"]="gprm_e gprm"

# Build all combinations
COMBINATIONS=()

for test_type in "${!STEP_MAP[@]}"; do
    steps=(${STEP_MAP[$test_type]})
    models=(${MODEL_MAP[$test_type]})
    for step_index in "${!steps[@]}"; do
        step="${steps[$step_index]}"
        for model in "${models[@]}"; do
            COMBINATIONS+=("$step $model $test_type $step_index")
        done
    done
done

# Bounds check for SLURM_ARRAY_TASK_ID
if [[ -z "$SLURM_ARRAY_TASK_ID" ]] || [[ -z "${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}" ]]; then
    echo "Error: Invalid SLURM_ARRAY_TASK_ID ($SLURM_ARRAY_TASK_ID). Max index: $((${#COMBINATIONS[@]} - 1))."
    exit 1
fi

# Extract combination
combo="${COMBINATIONS[$SLURM_ARRAY_TASK_ID]}"
read step model_type test_type step_index <<< "$combo"

echo "Running with: step=$step, model_type=$model_type, test_type=$test_type, index=$step_index"

# Run your optimization script
.venv/bin/python scripts/optimize_osse.py $step "$model_type" "$test_type" "$step_index"
