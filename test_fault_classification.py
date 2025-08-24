#!/usr/bin/env python3
"""
Test script for FaultClassificationTester.

Author: Yucong Zhang
Email: yucong0428@outlook.com

This script provides comprehensive fault classification evaluation across multiple datasets
including MAFAULDA, CWRU, IIEE, and IICA with various evaluation strategies.
"""

import os
import argparse
import logging
from siren import FaultClassificationTester

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main(dataset_root: str, extractor_name: str = 'simple_feature_extractor', 
         use_kfold: bool = True, n_splits: int = 5, dataset_type: str = 'mafaulda',
         use_loocv: bool = False,
         external_test: bool = False,
         group_by_condition: bool = False):
    """
    Test fault classification with specified dataset and extractor.
    
    Args:
        dataset_root: Path to the dataset root
        extractor_name: Name of the feature extractor
        use_kfold: Whether to use k-fold cross validation
        n_splits: Number of folds for k-fold cross validation
        dataset_type: Type of dataset (mafaulda, cwru)
    """
    if use_loocv:
        eval_mode = "Leave-One-Out Cross Validation (LOOCV)"
    elif use_kfold:
        eval_mode = f"k-fold cross validation (n_splits={n_splits})"
    else:
        eval_mode = "train-test split"
    if external_test:
        eval_mode = "external test (train=train_cut, test=test)"
    print(f'Testing fault classification with {extractor_name} on {dataset_root} using {eval_mode}...')
    
    # Check if dataset exists
    if not os.path.exists(dataset_root):
        logger.error(f"Dataset not found at {dataset_root}")
        logger.info("Please ensure the MAFAULDA dataset is available at the specified path")
        return
    
    # Create feature extractor based on extractor_name using importlib
    import importlib
    
    try:
        # Import the extractor module
        extractor_module = importlib.import_module(extractor_name)
        FeatureExtractor = getattr(extractor_module, 'FeatureExtractor')
        
        # Create feature extractor with default parameters
        # Each extractor should handle its own default parameters
        feature_extractor = FeatureExtractor()
            
    except ImportError:
        raise ValueError(f"Extractor module '{extractor_name}' not found. Make sure the module exists and is in the Python path.")
    except AttributeError:
        raise ValueError(f"FeatureExtractor class not found in module '{extractor_name}'. All extractors must have a FeatureExtractor class.")
    
    # Create tester configuration
    config = {
        'dataset_root': dataset_root,
        'dataset_type': dataset_type,
        'feature_extractor': feature_extractor,
        'extractor_name': extractor_name,
        'k': 5,
        'metric': 'euclidean',
        'use_kfold': use_kfold,
        'use_loocv': use_loocv,
        'n_splits': n_splits,
        'test_size': 0.2,  # Only used when use_kfold=False
        'random_state': 1,
        'use_external_test': external_test,
        'group_by_condition': group_by_condition,
        'return_per_channel_knn': args.per_channel_knn
    }
    
    try:
        # Create and run tester
        logger.info("Creating FaultClassificationTester...")
        tester = FaultClassificationTester(config)
        
        # Get dataset stats
        stats = tester.get_dataset_stats()
        logger.info(f"Dataset stats: {stats}")
        
        # Run evaluation
        logger.info("Running fault classification evaluation...")
        results = tester.run_evaluation()
        
        # Print summary
        logger.info("=" * 60)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Evaluation Mode: {results.get('evaluation_mode', 'train-test')}")
        logger.info(f"Overall Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Overall F1-Score: {results['f1_score']:.4f}")
        logger.info(f"Number of Classes: {len(results['classes'])}")
        logger.info(f"Classes: {results['classes']}")
        
        # Show k-fold specific summary if available
        if results.get('evaluation_mode') == 'k-fold':
            fold_stats = results['fold_statistics']
            logger.info(f"Cross-validation accuracy: {fold_stats['accuracy_mean']:.4f} ± {fold_stats['accuracy_std']:.4f}")
            logger.info(f"Cross-validation F1-score: {fold_stats['f1_mean']:.4f} ± {fold_stats['f1_std']:.4f}")
            logger.info(f"Number of folds: {results['n_splits']}")
        
        print('Fault classification evaluation completed successfully!')
        return results
        
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate fault classification on MAFAULDA dataset')
    parser.add_argument('--dataset_root', type=str, 
                       default='/DKUdata/machine_signal_data/MAFAULDA', 
                       help='Path to the MAFAULDA dataset root')
    parser.add_argument('--extractor_name', type=str, default='simple_feature_extractor', 
                       help='Name of the feature extractor')
    parser.add_argument('--use_kfold', action='store_true', default=True,
                       help='Use k-fold cross validation (default: True)')
    parser.add_argument('--no_kfold', dest='use_kfold', action='store_false',
                       help='Use train-test split instead of k-fold')
    parser.add_argument('--loocv', action='store_true', default=False,
                       help='Use Leave-One-Out Cross Validation (overrides k-fold and train-test)')
    parser.add_argument('--n_splits', type=int, default=5,
                       help='Number of folds for k-fold cross validation (default: 5)')
    parser.add_argument('--dataset_type', type=str, default='mafaulda',
                        help='Dataset type: mafaulda (CSV), cwru (MAT with dynamic rates), iiee/iica (WAV)')
    parser.add_argument('--per_channel_knn', action='store_true', default=False,
                        help='Also compute per-channel KNN predictions for multi-channel features')
    parser.add_argument('--external_test', action='store_true', default=False,
                        help='For IIEE: train on train_cut, test on test (noise-robust evaluation)')
    parser.add_argument('--group_by_condition', action='store_true', default=False,
                        help='Group IIEE external test results by condition (talking/atmo/whitenoise/stresstest)')
    args = parser.parse_args()
    
    main(args.dataset_root, args.extractor_name, args.use_kfold, args.n_splits, args.dataset_type, args.loocv,
         args.external_test, args.group_by_condition)
