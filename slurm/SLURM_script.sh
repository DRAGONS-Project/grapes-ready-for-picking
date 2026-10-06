#!/bin/bash
set -e

# Show help message
show_help() {
    cat << EOF
Usage: $0 --script SCRIPT [OPTIONS]

Submit jobs to SLURM cluster with customizable parameters.

Required:
  --script SCRIPT          Script to run

Optional:
  --params "PARAMS"        Parameters to pass to script (quoted)
  --time HH:MM:SS          Time limit (default: 24:00:00)
  --partition NAME         Partition name (default: short)
  --gpu N                  Number of GPUs (default: 1)
  --gpu-type TYPE          GPU type (e.g., a100, v100)
  --cpus N                 CPUs per task (default: 4)
  --mem SIZE               Memory limit (default: 25GB)
  --dependency JOBID       Job dependency (afterok:JOBID)
  --array RANGE            Array job range (e.g., 1-10)
  --log-dir DIR            Directory for logs (default: logs)
  --track-mem              Enable RAM and VRAM logging to CSV
  --print-only             Print script without submitting
  --help                   Show this help message

Examples:
  $0 --script train.py --params "--config config.py" --time 48:00:00
  $0 --script analyze.py --track-mem --mem 1TB
EOF
}

# Default values for variables
SCRIPT=""
PARAMS=""
PRINT_ONLY=false
TIME="24:00:00"
PARTITION="short"
GPU="1"
GPU_TYPE=""
CPUS="1"
MEM="16GB"
DEPENDENCY=""
ARRAY=""
LOG_DIR="logs"
TRACK_MEM=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --script)
            SCRIPT=$2
            shift 2
            ;;
        --params)
            PARAMS=$2
            shift 2
            ;;
        --time)
            TIME=$2
            shift 2
            ;;
        --partition)
            PARTITION=$2
            shift 2
            ;;
        --gpu)
            GPU=$2
            shift 2
            ;;
        --gpu-type)
            GPU_TYPE=$2
            shift 2
            ;;
        --cpus)
            CPUS=$2
            shift 2
            ;;
        --mem)
            MEM=$2
            shift 2
            ;;
        --dependency)
            DEPENDENCY=$2
            shift 2
            ;;
        --array)
            ARRAY=$2
            shift 2
            ;;
        --log-dir)
            LOG_DIR=$2
            shift 2
            ;;
        --track-mem)
            TRACK_MEM=true
            shift
            ;;
        --print-only)
            PRINT_ONLY=true
            shift
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$SCRIPT" ]]; then
    echo "Error: You must specify the script to run with --script."
    echo "Use --help for usage information"
    exit 1
fi

# Validate script exists
if [[ ! -f "$SCRIPT" ]]; then
    echo "Error: Script '$SCRIPT' not found."
    exit 1
fi

# Validate time format
if ! [[ $TIME =~ ^[0-9]{1,3}:[0-9]{2}:[0-9]{2}$ ]]; then
    echo "Error: Invalid time format '$TIME'. Use HH:MM:SS or HHH:MM:SS"
    exit 1
fi

# Validate numeric parameters
if ! [[ $GPU =~ ^[0-9]+$ ]]; then
    echo "Error: --gpu must be a positive number"
    exit 1
fi

if ! [[ $CPUS =~ ^[0-9]+$ ]]; then
    echo "Error: --cpus must be a positive number"
    exit 1
fi

# Generate a unique job script based on the script name and parameters
JOB_NAME=$(basename "${SCRIPT%.py}" .sh)
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_SCRIPT="tmp/job_${JOB_NAME}_${TIMESTAMP}.slurm"

# Create necessary directories
mkdir -p tmp
mkdir -p "$LOG_DIR/${JOB_NAME}"

# Create the SLURM job script
(
    echo "#!/bin/bash"
    echo "#SBATCH --job-name=$JOB_NAME"
    echo "#SBATCH --account=FILL_MANUALLY"
    echo "#SBATCH --partition=$PARTITION"
    echo "#SBATCH --nodes=1"
    echo "#SBATCH --cpus-per-task=$CPUS"
    echo "#SBATCH --mem=$MEM"

    # GPU configuration
    if [[ -n "$GPU_TYPE" ]]; then
        echo "#SBATCH --gres=gpu:$GPU_TYPE:$GPU"
    else
        echo "#SBATCH --gres=gpu:$GPU"
    fi

    echo "#SBATCH --time=$TIME"

    # Standard Output and Error — logs/SCRIPTNAME/JOBID[_TASKID]/JOBID[_TASKID].{out,err}
    if [[ -n "$ARRAY" ]]; then
        echo "#SBATCH --output=$LOG_DIR/${JOB_NAME}/%A_%a/%A_%a.out"
        echo "#SBATCH --error=$LOG_DIR/${JOB_NAME}/%A_%a/%A_%a.err"
    else
        echo "#SBATCH --output=$LOG_DIR/${JOB_NAME}/%j/%j.out"
        echo "#SBATCH --error=$LOG_DIR/${JOB_NAME}/%j/%j.err"
    fi

    # Job dependencies
    if [[ -n "$DEPENDENCY" ]]; then
        echo "#SBATCH --dependency=afterok:$DEPENDENCY"
    fi

    # Array jobs
    if [[ -n "$ARRAY" ]]; then
        echo "#SBATCH --array=$ARRAY"
    fi

    echo
    echo "# --- Setup Job Directory ---"
    echo "JOB_ID=\${SLURM_ARRAY_JOB_ID:-\$SLURM_JOB_ID}"
    echo "SUB_ID=\${SLURM_ARRAY_TASK_ID:-\"\"}"
    echo "if [ -n \"\$SUB_ID\" ]; then FOLDER_ID=\"\${JOB_ID}_\${SUB_ID}\"; else FOLDER_ID=\"\$JOB_ID\"; fi"
    echo "JOB_LOG_DIR=\"$LOG_DIR/${JOB_NAME}/\$FOLDER_ID\""
    echo "mkdir -p \"\$JOB_LOG_DIR\""
    echo

    echo 'echo "Job started on $(date) at $HOSTNAME"'
    echo 'echo "Job ID: $SLURM_JOB_ID"'
    echo 'echo "Logs being saved to: $JOB_LOG_DIR"'
    echo

    # Optional Resource Tracker
    if $TRACK_MEM; then
        echo "# --- Start Resource Tracking ---"
        echo "RESOURCE_LOG=\"\$JOB_LOG_DIR/resource_usage.csv\""
        echo "echo 'Starting RAM/VRAM logger...'"
        # Uses uv to pull dependencies on the fly without affecting main env
        echo "uv run --with psutil --with nvidia-ml-py --with pandas --with matplotlib \\"
        echo "   python resource_logger.py --output \"\$RESOURCE_LOG\" --pid \$$ --interval 5 --plot &"
        echo "LOGGER_PID=\$!"
        echo
    fi

    # Execute the script
    echo 'echo "Executing command:"'
    echo "export OUTPUT_DIR=\"\$JOB_LOG_DIR\""

    # Disable auto-exit on error so we can catch OOM crashes and still plot
    echo "set +e"
    
    if [[ "$SCRIPT" == *.py ]]; then
        echo "echo \"uv run python $SCRIPT $PARAMS\""
        echo "uv run python \"$SCRIPT\" $PARAMS"
    else
        echo "echo \"bash $SCRIPT $PARAMS\""
        echo "bash \"$SCRIPT\" $PARAMS"
    fi

    # Capture the exit code
    echo "EXIT_CODE=\$?"
    echo "set -e"

    echo
    # Clean up tracker
    if $TRACK_MEM; then
        echo "echo 'Stopping logger and waiting for plot generation...'"
        echo "kill \$LOGGER_PID 2>/dev/null || true"
        ## Wait for the logger to finish plotting before closing the job
        echo "wait \$LOGGER_PID"
    fi
    
    echo 'echo "Job finished on $(date)"'
    ## Exit with the actual status code of your script
    echo "exit \$EXIT_CODE"
) > "$JOB_SCRIPT"

if $PRINT_ONLY; then
    echo "========================================"
    echo "Generated SLURM script: $JOB_SCRIPT"
    echo "========================================"
    cat "$JOB_SCRIPT"
    echo
    echo "========================================"
    echo "To submit this job, remove --print-only flag"
    echo "========================================"
else
    # Before submitting, we need to ensure the parent log dir exists 
    # but the job-specific folder is created inside the slurm task itself.
    
    OUTPUT=$(sbatch "$JOB_SCRIPT")

    # Extract job ID from output
    if [[ $OUTPUT =~ ([0-9]+) ]]; then
        JOB_ID="${BASH_REMATCH[1]}"
    else
        echo "Error: Failed to extract job ID from sbatch output:"
        echo "$OUTPUT"
        exit 1
    fi

    FINAL_LOG_DIR="$LOG_DIR/${JOB_NAME}/${JOB_ID}"
    mkdir -p "$FINAL_LOG_DIR"
    FINAL_SCRIPT_PATH="$FINAL_LOG_DIR/job_${JOB_NAME}_${TIMESTAMP}.slurm"
    mv "$JOB_SCRIPT" "$FINAL_SCRIPT_PATH"

    echo "========================================"
    echo "Job submitted successfully!"
    echo "========================================"
    echo "Job ID: $JOB_ID"
    echo "Folder: $LOG_DIR/${JOB_NAME}/${JOB_ID}/"
    echo "Script: $SCRIPT"
    echo "Parameters: $PARAMS"
    echo "Time limit: $TIME"
    echo "Partition: $PARTITION"
    echo "GPUs: $GPU"
    [[ -n "$GPU_TYPE" ]] && echo "GPU Type: $GPU_TYPE"
    [[ -n "$ARRAY" ]] && echo "Array: $ARRAY"
    echo "========================================"
    if [[ -n "$ARRAY" ]]; then
        echo "Output logs: $LOG_DIR/${JOB_NAME}/${JOB_ID}_<task>/<task>.out"
        echo "Error logs:  $LOG_DIR/${JOB_NAME}/${JOB_ID}_<task>/<task>.err"
    else
        echo "Output log: $LOG_DIR/${JOB_NAME}/${JOB_ID}/${JOB_ID}.out"
        echo "Error log:  $LOG_DIR/${JOB_NAME}/${JOB_ID}/${JOB_ID}.err"
    fi
    echo "========================================"
    echo "Monitor with: squeue -j $JOB_ID"
    echo "Cancel with: scancel $JOB_ID"
    if [[ -n "$ARRAY" ]]; then
        echo "View output: tail -f $LOG_DIR/${JOB_NAME}/${JOB_ID}_0/${JOB_ID}_0.out"
    else
        echo "View output: tail -f $LOG_DIR/${JOB_NAME}/${JOB_ID}/${JOB_ID}.out"
    fi
fi
