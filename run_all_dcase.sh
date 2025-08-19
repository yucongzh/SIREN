#!/bin/bash

# Parallel DCASE Evaluation Script for SIREN Toolkit
# Author: Yucong Zhang
# Email: yucong0428@outlook.com
#
# This script runs DCASE evaluation for all years (2020-2025) in parallel
# with each year assigned to a different GPU for maximum efficiency.
# Usage: bash run_all_dcase.sh your_extractor_path

# Assign GPUs to each evaluation
# You can modify these GPU assignments based on your system
declare -A gpu_assignments=(
    ["2020"]="0"
    ["2021"]="1" 
    ["2022"]="2"
    ["2023"]="3"
    ["2024"]="4"
    ["2025"]="5"
)

# Set your conda environment path - replace with your actual environment path
# Example: CONDA_ENV="/path/to/your/env"
CONDA_ENV="/path/to/your/env"
extractor_path=$1

BASE_DIR=$(pwd)
LOG_DIR=${BASE_DIR}/logs
mkdir -p ${LOG_DIR}
# Run all DCASE years
for year in 2020 2021 2022 2023 2024 2025; do
    nohup bash -c "
        source ~/.bashrc
        conda activate ${CONDA_ENV}
        export CUDA_VISIBLE_DEVICES=${gpu_assignments[$year]}
        cd ${BASE_DIR}
        echo 'DCASE ${year} evaluation started on GPU ${gpu_assignments[$year]}'
        python evaluate_dcase.py --extractor_path $extractor_path --dcase_year $year > ${LOG_DIR}/dcase_evaluation_${year}.log 2>&1
        echo 'DCASE ${year} evaluation completed on GPU ${gpu_assignments[$year]}' >> ${LOG_DIR}/dcase_evaluation_${year}.log
    " &
done

# Function to show status
show_status() {
    echo "=== DCASE Evaluation Status ==="
    for year in 2020 2021 2022 2023 2024 2025; do
        # Check if the process is actually running using ps -ef
        if ps -ef | grep -v grep | grep -q "evaluate.py.*${extractor_path}.*${year}"; then
            # Get the actual PID from ps output
            actual_pid=$(ps -ef | grep -v grep | grep "evaluate.py.*${extractor_path}.*${year}" | awk '{print $2}' | head -1)
            echo "DCASE ${year}: Running (PID: $actual_pid)"
        else
            # Check if PID file exists and if the process was completed
            pid_file="${LOG_DIR}/dcase_evaluation_${year}.log"
            if [ -f "$pid_file" ]; then
                echo "DCASE ${year}: Completed"
            else
                echo "DCASE ${year}: Not started"
            fi
        fi
    done
    echo "================================"
}

show_gpu_usage() {
    echo "=== GPU Usage ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r index name mem_used mem_total util; do
        echo "GPU $index ($name): ${mem_used}MB/${mem_total}MB (${util}% util)"
    done
    echo "================="
}

while true; do
    show_status
    echo
    echo
    show_gpu_usage
    sleep 5
    clear
done
