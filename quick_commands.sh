#!/bin/bash

# Quick Evaluation Commands for SIREN Toolkit
# Author: Yucong Zhang
# Email: yucong0428@outlook.com
#
# This script provides comprehensive evaluation across all supported datasets and tasks.
# Usage: bash quick_commands.sh your_extractor_name

# 0. setting up environment
# Set your CUDA devices - replace with your actual GPU configuration
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Set your conda environment path - replace with your actual environment path
# Example: source ~/anaconda3/bin/activate /path/to/your/env
source ~/anaconda3/bin/activate /path/to/your/env
root_path=./ # enter the path contains the extractors
your_extractor_name=${1:-audioMAE} # name of your model, default to audioMAE

# 1. test DCASE series (2020-2025)
# 1.1 test single year
# echo "testing single year DCASE..."
# python evaluate_dcase.py --dataset_root /path/to/machine_signal_data \
#                          --extractor_path ${root_path}/${your_extractor_name}_extractor.py \
#                          --dcase_year 202X

# 1.2 test all years
# 1.2.1 single process
echo "testing DCASE all years sequentially..."
python evaluate_dcase.py --dataset_root /path/to/machine_signal_data \
                         --extractor_path ${root_path}/${your_extractor_name}_extractor.py \
                         --dcase_year all

# 1.2.2 multi process (adjust GPU, need to modify run_all_dcase.sh)
# echo "testing all years DCASE in parallel..."
# ./run_all_dcase.sh ${your_extractor_name}_extractor.py

# 2. test MAFAULDA dataset
echo "testing MAFAULDA dataset..."
python test_fault_classification.py --dataset_root /path/to/MAFAULDA \
                                    --extractor_name ${your_extractor_name}_extractor \
                                    --dataset_type mafaulda \
                                    --loocv --per_channel_knn

# 3. test CWRU dataset
echo "testing CWRU dataset..."
python test_fault_classification.py --dataset_root /path/to/CWRU_Bearing_Dataset/CWRU-dataset \
                                    --extractor_name ${your_extractor_name}_extractor \
                                    --dataset_type cwru \
                                    --loocv

# 4 test IIEE dataset (external test + per-condition)
echo "testing IIEE dataset (external test + per-condition)..."
python test_fault_classification.py --dataset_root /path/to/IDMT-ISA-Electric-Engine \
                                    --extractor_name ${your_extractor_name}_extractor \
                                    --dataset_type iiee \
                                    --no_kfold \
                                    --external_test --group_by_condition

# 5. test IICA dataset (k-fold)
echo "testing IICA dataset (k-fold)..."
python test_fault_classification.py --dataset_root /path/to/IDMT-ISA-Compressed-Air \
                                    --extractor_name ${your_extractor_name}_extractor \
                                    --dataset_type iica \
                                    --use_kfold --n_splits 5


