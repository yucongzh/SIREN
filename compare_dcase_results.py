#!/usr/bin/env python3
"""
DCASE Results Comparison Script

Author: Yucong Zhang
Email: yucong0428@outlook.com

This script compares DCASE evaluation results across different systems and years,
generating comprehensive comparison tables and visualizations.
"""

import os
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from scipy.stats import hmean
from enhanced_dcase_comparison import DCASEDetailedComparator

def find_overall_summaries(root_dir):
    """Find all dcaseYYYY_overall_summary.csv files in the test_results directory."""
    summaries = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith('_overall_summary.csv') and file.startswith('dcase'):
                year = file.split('_')[0].replace('dcase', '')
                system = Path(root).parts[-3]  # Extract system name from path
                summaries.append({
                    'year': year,
                    'system': system,
                    'path': os.path.join(root, file)
                })
    return summaries

def load_and_compare(summaries):
    """Load all summary files and create pivoted comparisons for each score type."""
    all_data = []
    for summary in summaries:
        try:
            df = pd.read_csv(summary['path'])
            data = {
                'year': summary['year'],
                'system': summary['system']
            }
            metric_map = {
                'Dev Overall': 'dev_score',
                'Eval Overall': 'eval_score',
                'Overall Year': 'overall_year_score'
            }
            for _, row in df.iterrows():
                metric = row['metric_type']
                if metric in metric_map:
                    data[metric_map[metric]] = row['score']
            all_data.append(data)
        except Exception as e:
            print(f"Error loading {summary['path']}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No valid data found to compare")
    
    combined = pd.DataFrame(all_data)
    
    # Create pivoted tables for each score
    score_types = ['dev_score', 'eval_score', 'overall_year_score']
    pivots = {}
    hmean_rows = {}
    for score in score_types:
        if score in combined.columns:
            pivot = combined.pivot_table(
                index='year',
                columns='system',
                values=score,
                aggfunc='first'
            )
            # Add hmean row
            hmean_row = pivot.apply(lambda x: hmean(x.dropna()) if len(x.dropna()) > 0 else float('nan'))
            hmean_row.name = 'hmean'
            pivot = pd.concat([pivot, hmean_row.to_frame().T])
            pivots[score] = pivot
            hmean_rows[score] = hmean_row
    
    if not pivots:
        return {}
    
    # Determine reference for sorting
    reference_score = 'overall_year_score' if 'overall_year_score' in hmean_rows else next(iter(hmean_rows))
    reference_hmean = hmean_rows[reference_score]
    
    # Sort systems by reference hmean descending
    sorted_columns = reference_hmean.sort_values(ascending=False).index
    
    # Apply consistent ordering to all pivots
    for score in pivots:
        pivot = pivots[score]
        # Reorder columns, filling missing with NaN
        pivot = pivot.reindex(columns=sorted_columns)
        pivots[score] = pivot
    
    return pivots

def save_comparison(pivots, output_dir):
    """Save the pivoted comparison results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    for score, pivot in pivots.items():
        output_path = os.path.join(output_dir, f'dcase_{score}_comparison.csv')
        pivot.to_csv(output_path)
        print(f"{score.capitalize().replace('_', ' ')} comparison saved to {output_path}")

def plot_comparisons(pivots, output_dir):
    """Generate and save comparison plots for each score type."""
    os.makedirs(output_dir, exist_ok=True)
    for score, pivot in pivots.items():
        if pivot.empty:
            continue
        
        # Separate data rows and hmean row
        data_pivot = pivot.iloc[:-1]
        hmean_row = pivot.iloc[-1:]
        
        # Create bar plot for years
        ax = data_pivot.plot(
            kind='bar',
            figsize=(12, 6),
            width=0.8,
            rot=0
        )
        
        # Customize plot
        ax.set_title(f'DCASE {score.replace("_", " ").title()} Comparison')
        ax.set_xlabel('DCASE Year')
        ax.set_ylabel('Score')
        ax.legend(title='System', bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels for years
        for container in ax.containers:
            for rect in container:
                height = rect.get_height()
                ax.text(
                    rect.get_x() + rect.get_width() / 2,
                    height,
                    f'{height:.4f}',
                    ha='center',
                    va='bottom',
                    fontsize=8
                )
        
        # Add hmean as text or separate annotation
        y_max = ax.get_ylim()[1]
        ax.text(
            len(data_pivot) + 0.5,  # Position after the last bar
            y_max * 0.5,
            'HMean:\n' + '\n'.join([f"{sys}: {val:.4f}" for sys, val in hmean_row.iloc[0].items()]),
            va='center',
            ha='left',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5')
        )
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure
        output_path = os.path.join(output_dir, f'dcase_{score}_comparison.png')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        print(f"{score.capitalize().replace('_', ' ')} plot saved to {output_path}")

if __name__ == "__main__":
    root_dir = "test_results"
    output_dir = "comparison_results"
    
    print("="*60)
    print("DCASE COMPREHENSIVE COMPARISON TOOL")
    print("="*60)
    
    # Run original overall comparison
    print("\n1. RUNNING OVERALL SUMMARY COMPARISON...")
    print("-" * 40)
    
    print("Finding all DCASE overall summary files...")
    summaries = find_overall_summaries(root_dir)
    
    if not summaries:
        print("No summary files found!")
        exit(1)
    
    print(f"Found {len(summaries)} summary files from {len({s['system'] for s in summaries})} systems")
    
    print("Loading and comparing results...")
    try:
        pivots = load_and_compare(summaries)
        print("\nQuick Preview:")
        for score, pivot in pivots.items():
            print(f"\n{score.upper().replace('_', ' ')} COMPARISON:")
            print(pivot.to_string())
        
        print("\nSaving comparisons...")
        save_comparison(pivots, output_dir)
        
        print("\nGenerating plots...")
        plot_comparisons(pivots, output_dir)
        
        print("Overall comparison completed!")
    except Exception as e:
        print(f"Error during overall comparison: {str(e)}")
        print("Continuing with detailed analysis...")
    
    # Run enhanced detailed comparison
    print("\n\n2. RUNNING ENHANCED DETAILED COMPARISON...")
    print("-" * 40)
    
    try:
        enhanced_output_dir = "enhanced_comparison_results"
        comparator = DCASEDetailedComparator(root_dir, enhanced_output_dir)
        comparator.run_enhanced_comparison()
    except Exception as e:
        print(f"Error during enhanced comparison: {str(e)}")
    
    print("\n" + "="*60)
    print("COMPARISON ANALYSIS COMPLETED!")
    print("="*60)
    print(f"Results available in:")
    print(f"  - {output_dir}/ (overall comparisons)")
    print(f"  - enhanced_comparison_results/ (detailed analysis)")
    print("    ├── dcase20XX_detailed/ (year-specific details)")
    print("    ├── cross_year_analysis/ (trends)")
    print("    └── visualizations/ (charts and graphs)")
