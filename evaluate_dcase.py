"""
DCASE Evaluation Script

Author: Yucong Zhang
Email: yucong0428@outlook.com

This script evaluates feature extractors on the DCASE dataset series.
It supports both single-year and all-years evaluation with multiprocessing options.
"""

import multiprocessing as mp
from multiprocessing import freeze_support
from siren import DCASETester
import os
import argparse

# Define paths - replace with your actual paths
# Example: ROOT = "/path/to/your/workspace"
ROOT = '/path/to/your/workspace'

def main(dataset_root: str, extractor_path: str, dcase_year: str, use_multiprocessing: bool = True):
    print('Testing {} with extractor from {}...'.format(
        'multiprocessing' if use_multiprocessing else 'sequential', 
        extractor_path
    ))
    model_name = os.path.basename(extractor_path).split('.')[0]
    tester = DCASETester(dataset_root, 
                         dcase_year, 
                         extractor_path, 
                         cache_enabled=True, 
                         multiprocessing_enabled=use_multiprocessing,  # Use parameter control
                         results_dir=os.path.join(ROOT, f'test_results/{model_name}'),
                         cache_dir=os.path.join(ROOT, f'feature_cache/{model_name}'))
    
    mode = 'multiprocessing' if use_multiprocessing else 'sequential'
    print(f'Starting {mode} evaluation...')
    tester.run_evaluation()
    print(f'{mode.capitalize()} evaluation completed')

if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description='Evaluate a feature extractor on the DCASE dataset')
    parser.add_argument('--dataset_root', type=str, default='/path/to/machine_signal_data', help='Path to the dataset root')
    parser.add_argument('--extractor_path', type=str, default='./audioMAE_extractor.py', help='Path to the extractor module')
    parser.add_argument('--dcase_year', type=str, default='2023', help='Year of the DCASE dataset')
    parser.add_argument('--no_multiprocessing', action='store_true', help='Disable multiprocessing (use sequential processing)')
    
    args = parser.parse_args()
    
    if args.dcase_year == 'all':
        main(args.dataset_root, args.extractor_path, 'all', not args.no_multiprocessing)
    else:
        main(args.dataset_root, args.extractor_path, int(args.dcase_year), not args.no_multiprocessing)
