#!/usr/bin/env python3
"""
Enhanced DCASE Comparison Script

Author: Yucong Zhang
Email: yucong0428@outlook.com

This script provides enhanced DCASE evaluation comparison with detailed machine type analysis,
generating comprehensive visualizations and statistical comparisons across systems and years.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hmean
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

class DCASEDetailedComparator:
    """Enhanced DCASE comparison tool with detailed machine type analysis."""
    
    def __init__(self, root_dir="test_results", output_dir="enhanced_comparison_results"):
        self.root_dir = root_dir
        self.output_dir = output_dir
        self.systems = []
        self.years = []
        
    def find_files(self):
        """Find all DCASE files (both summary and detailed)."""
        summaries = []
        detailed_files = []
        
        for root, _, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('_overall_summary.csv') and file.startswith('dcase'):
                    year = file.split('_')[0].replace('dcase', '')
                    system = Path(root).parts[-3]  # Extract system name from path
                    summaries.append({
                        'year': year,
                        'system': system,
                        'type': 'summary',
                        'path': os.path.join(root, file)
                    })
                elif file.endswith('_detailed_groups.csv') and file.startswith('dcase'):
                    year = file.split('_')[0].replace('dcase', '')
                    system = Path(root).parts[-3]
                    detailed_files.append({
                        'year': year,
                        'system': system,
                        'type': 'detailed',
                        'path': os.path.join(root, file)
                    })
        
        self.systems = sorted(list(set([s['system'] for s in summaries])))
        self.years = sorted(list(set([s['year'] for s in summaries])))
        
        return summaries, detailed_files
    
    def load_detailed_data(self, detailed_files):
        """Load and process detailed machine type data."""
        all_detailed_data = []
        
        for file_info in detailed_files:
            try:
                df = pd.read_csv(file_info['path'])
                df['year'] = file_info['year']
                df['system'] = file_info['system']
                all_detailed_data.append(df)
            except Exception as e:
                print(f"Error loading {file_info['path']}: {str(e)}")
                continue
        
        if not all_detailed_data:
            return pd.DataFrame()
            
        return pd.concat(all_detailed_data, ignore_index=True)
    
    def create_machine_type_comparison(self, detailed_df):
        """Create machine type comparison across systems and years."""
        if detailed_df.empty:
            return {}
        
        comparisons = {}
        
        # Group by year for yearly analysis
        for year in self.years:
            year_data = detailed_df[detailed_df['year'] == year]
            if year_data.empty:
                continue
                
            year_comparisons = {}
            
            # For each metric type (AUC, pAUC)
            for metric in year_data['metric_type'].unique():
                metric_data = year_data[year_data['metric_type'] == metric]
                
                # Create pivot tables for different views
                pivots = {}
                
                # 1. Machine type vs System (overall)
                if 'domain' in metric_data.columns:
                    # For AUC data with source/target domains
                    if metric == 'AUC':
                        for domain in ['source', 'target']:
                            domain_data = metric_data[metric_data['domain'] == domain]
                            if not domain_data.empty:
                                pivot = domain_data.pivot_table(
                                    index='machine_type',
                                    columns='system',
                                    values='score',
                                    aggfunc='mean'
                                )
                                pivots[f'{metric}_{domain}'] = pivot
                    else:
                        # For pAUC data (combined domain)
                        combined_data = metric_data[metric_data['domain'] == 'combined']
                        if not combined_data.empty:
                            pivot = combined_data.pivot_table(
                                index='machine_type',
                                columns='system',
                                values='score',
                                aggfunc='mean'
                            )
                            pivots[f'{metric}_combined'] = pivot
                
                year_comparisons[metric] = pivots
            
            comparisons[year] = year_comparisons
        
        return comparisons
    
    def create_system_ranking_by_machine(self, detailed_df):
        """Create system rankings for each machine type."""
        if detailed_df.empty:
            return {}
        
        rankings = {}
        
        for year in self.years:
            year_data = detailed_df[detailed_df['year'] == year]
            if year_data.empty:
                continue
                
            year_rankings = {}
            
            # Focus on AUC source domain for ranking (most common)
            auc_source = year_data[
                (year_data['metric_type'] == 'AUC') & 
                (year_data['domain'] == 'source')
            ]
            
            if not auc_source.empty:
                # For each machine type, rank systems
                for machine in auc_source['machine_type'].unique():
                    machine_data = auc_source[auc_source['machine_type'] == machine]
                    machine_avg = machine_data.groupby('system')['score'].mean().sort_values(ascending=False)
                    year_rankings[machine] = machine_avg
            
            rankings[year] = year_rankings
        
        return rankings
    
    def generate_comprehensive_reports(self, summaries, detailed_files):
        """Generate comprehensive comparison reports."""
        os.makedirs(self.output_dir, exist_ok=True)
        
        print("Loading detailed data...")
        detailed_df = self.load_detailed_data(detailed_files)
        
        if detailed_df.empty:
            print("No detailed data found!")
            return
        
        print("Creating machine type comparisons...")
        machine_comparisons = self.create_machine_type_comparison(detailed_df)
        
        print("Creating system rankings...")
        system_rankings = self.create_system_ranking_by_machine(detailed_df)
        
        # Generate reports for each year
        for year in self.years:
            print(f"Generating reports for DCASE {year}...")
            self.generate_year_specific_report(
                year, 
                machine_comparisons.get(year, {}),
                system_rankings.get(year, {}),
                detailed_df
            )
        
        # Generate cross-year analysis
        print("Generating cross-year analysis...")
        self.generate_cross_year_analysis(machine_comparisons, detailed_df)
        
        print("Generating visualizations...")
        self.generate_visualizations(machine_comparisons, system_rankings, detailed_df)
    
    def generate_year_specific_report(self, year, comparisons, rankings, detailed_df):
        """Generate detailed report for a specific year."""
        year_dir = os.path.join(self.output_dir, f"dcase{year}_detailed")
        os.makedirs(year_dir, exist_ok=True)
        
        year_data = detailed_df[detailed_df['year'] == year]
        
        # Save machine type comparison tables
        for metric, pivots in comparisons.items():
            for pivot_name, pivot in pivots.items():
                if not pivot.empty:
                    # Add hmean row
                    hmean_row = pivot.apply(lambda x: hmean(x.dropna()) if len(x.dropna()) > 0 else np.nan)
                    hmean_row.name = 'hmean'
                    pivot_with_hmean = pd.concat([pivot, hmean_row.to_frame().T])
                    
                    output_path = os.path.join(year_dir, f'{metric}_{pivot_name}_comparison.csv')
                    pivot_with_hmean.to_csv(output_path)
                    print(f"  Saved {metric} {pivot_name} comparison to {output_path}")
        
        # Save system rankings
        if rankings:
            rankings_df = pd.DataFrame(rankings).T
            rankings_df.index.name = 'machine_type'
            rankings_path = os.path.join(year_dir, 'system_rankings_by_machine.csv')
            rankings_df.to_csv(rankings_path)
            print(f"  Saved system rankings to {rankings_path}")
        
        # Generate summary statistics
        self.generate_year_summary_stats(year, year_data, year_dir)
    
    def generate_year_summary_stats(self, year, year_data, year_dir):
        """Generate summary statistics for a year."""
        if year_data.empty:
            return
        
        stats = {}
        
        # Basic statistics
        stats['total_machines'] = year_data['machine_type'].nunique()
        stats['total_systems'] = year_data['system'].nunique()
        stats['available_metrics'] = year_data['metric_type'].unique().tolist()
        
        # Performance statistics by system
        system_stats = {}
        for system in year_data['system'].unique():
            system_data = year_data[year_data['system'] == system]
            auc_source = system_data[
                (system_data['metric_type'] == 'AUC') & 
                (system_data['domain'] == 'source')
            ]
            if not auc_source.empty:
                system_stats[system] = {
                    'mean_auc_source': auc_source['score'].mean(),
                    'std_auc_source': auc_source['score'].std(),
                    'min_auc_source': auc_source['score'].min(),
                    'max_auc_source': auc_source['score'].max(),
                    'machines_tested': auc_source['machine_type'].nunique()
                }
        
        stats['system_performance'] = system_stats
        
        # Save statistics
        stats_path = os.path.join(year_dir, 'summary_statistics.txt')
        with open(stats_path, 'w') as f:
            f.write(f"DCASE {year} Summary Statistics\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Total Machine Types: {stats['total_machines']}\n")
            f.write(f"Total Systems: {stats['total_systems']}\n")
            f.write(f"Available Metrics: {', '.join(stats['available_metrics'])}\n\n")
            
            f.write("System Performance (AUC Source Domain):\n")
            f.write("-" * 40 + "\n")
            for system, perf in system_stats.items():
                f.write(f"{system}:\n")
                f.write(f"  Mean: {perf['mean_auc_source']:.4f}\n")
                f.write(f"  Std:  {perf['std_auc_source']:.4f}\n")
                f.write(f"  Range: {perf['min_auc_source']:.4f} - {perf['max_auc_source']:.4f}\n")
                f.write(f"  Machines: {perf['machines_tested']}\n\n")
        
        print(f"  Saved summary statistics to {stats_path}")
    
    def generate_cross_year_analysis(self, machine_comparisons, detailed_df):
        """Generate cross-year analysis."""
        cross_year_dir = os.path.join(self.output_dir, "cross_year_analysis")
        os.makedirs(cross_year_dir, exist_ok=True)
        
        if detailed_df.empty:
            print("  No data available for cross-year analysis")
            return
        
        # 1. Create overall system performance trends across years
        auc_source_data = detailed_df[
            (detailed_df['metric_type'] == 'AUC') & 
            (detailed_df['domain'] == 'source')
        ]
        
        if not auc_source_data.empty:
            # Calculate average performance per system per year
            system_year_avg = auc_source_data.groupby(['year', 'system'])['score'].agg(['mean', 'std', 'count']).reset_index()
            system_year_avg.columns = ['year', 'system', 'mean_score', 'std_score', 'machine_count']
            
            # Create pivot table for better viewing
            performance_pivot = system_year_avg.pivot_table(
                index='year',
                columns='system',
                values='mean_score',
                aggfunc='first'
            )
            
            # Add hmean row
            hmean_row = performance_pivot.apply(lambda x: hmean(x.dropna()) if len(x.dropna()) > 0 else np.nan)
            hmean_row.name = 'hmean'
            performance_with_hmean = pd.concat([performance_pivot, hmean_row.to_frame().T])
            
            # Save system performance trends
            trends_path = os.path.join(cross_year_dir, 'system_performance_trends_by_year.csv')
            performance_with_hmean.to_csv(trends_path)
            print(f"  Saved system performance trends to {trends_path}")
        
        # 2. Find machines that appear in multiple years
        machine_year_counts = detailed_df.groupby('machine_type')['year'].nunique().sort_values(ascending=False)
        common_machines = machine_year_counts[machine_year_counts >= 3].index.tolist()  # Machines in 3+ years
        
        if common_machines:
            print(f"  Found {len(common_machines)} machines appearing in 3+ years: {', '.join(common_machines)}")
            
            # Create detailed trend analysis for common machines
            trend_data = []
            for year in self.years:
                year_data = detailed_df[detailed_df['year'] == year]
                auc_source = year_data[
                    (year_data['metric_type'] == 'AUC') & 
                    (year_data['domain'] == 'source') &
                    (year_data['machine_type'].isin(common_machines))
                ]
                
                for system in self.systems:
                    system_data = auc_source[auc_source['system'] == system]
                    if not system_data.empty:
                        for machine in common_machines:
                            machine_scores = system_data[system_data['machine_type'] == machine]['score']
                            if not machine_scores.empty:
                                trend_data.append({
                                    'year': year,
                                    'system': system,
                                    'machine_type': machine,
                                    'score': machine_scores.iloc[0]
                                })
            
            if trend_data:
                trend_df = pd.DataFrame(trend_data)
                detailed_trends_path = os.path.join(cross_year_dir, 'common_machines_detailed_trends.csv')
                trend_df.to_csv(detailed_trends_path, index=False)
                print(f"  Saved detailed machine trends to {detailed_trends_path}")
                
                # Create machine-specific pivot tables
                for machine in common_machines:
                    machine_data = trend_df[trend_df['machine_type'] == machine]
                    if not machine_data.empty:
                        machine_pivot = machine_data.pivot_table(
                            index='year',
                            columns='system',
                            values='score',
                            aggfunc='first'
                        )
                        
                        if not machine_pivot.empty:
                            # Add hmean row
                            hmean_row = machine_pivot.apply(lambda x: hmean(x.dropna()) if len(x.dropna()) > 0 else np.nan)
                            hmean_row.name = 'hmean'
                            machine_with_hmean = pd.concat([machine_pivot, hmean_row.to_frame().T])
                            
                            machine_path = os.path.join(cross_year_dir, f'{machine}_trends_by_year.csv')
                            machine_with_hmean.to_csv(machine_path)
                            print(f"  Saved {machine} trends to {machine_path}")
        
        # 3. Create system comparison summary across all years
        if not auc_source_data.empty:
            system_summary = auc_source_data.groupby('system')['score'].agg([
                'count', 'mean', 'std', 'min', 'max'
            ]).round(4)
            system_summary.columns = ['total_tests', 'overall_mean', 'overall_std', 'min_score', 'max_score']
            system_summary = system_summary.sort_values('overall_mean', ascending=False)
            
            summary_path = os.path.join(cross_year_dir, 'overall_system_summary.csv')
            system_summary.to_csv(summary_path)
            print(f"  Saved overall system summary to {summary_path}")
        
        # 4. Generate yearly statistics summary
        year_stats = []
        for year in self.years:
            year_data = detailed_df[detailed_df['year'] == year]
            if not year_data.empty:
                year_auc_source = year_data[
                    (year_data['metric_type'] == 'AUC') & 
                    (year_data['domain'] == 'source')
                ]
                if not year_auc_source.empty:
                    year_stats.append({
                        'year': year,
                        'total_machines': year_data['machine_type'].nunique(),
                        'total_systems': year_data['system'].nunique(),
                        'avg_performance': year_auc_source['score'].mean(),
                        'std_performance': year_auc_source['score'].std(),
                        'total_tests': len(year_auc_source)
                    })
        
        if year_stats:
            year_stats_df = pd.DataFrame(year_stats)
            year_stats_path = os.path.join(cross_year_dir, 'yearly_statistics_summary.csv')
            year_stats_df.to_csv(year_stats_path, index=False)
            print(f"  Saved yearly statistics to {year_stats_path}")
    
    def generate_visualizations(self, machine_comparisons, system_rankings, detailed_df):
        """Generate various visualizations."""
        viz_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(viz_dir, exist_ok=True)
        
        # Generate heatmaps for each year
        for year in self.years:
            if year in machine_comparisons:
                self.create_heatmaps(year, machine_comparisons[year], viz_dir)
        
        # Generate radar charts (passing detailed_df for hmean calculation)
        self.create_radar_charts(system_rankings, viz_dir, detailed_df)
        
        # Generate trend plots
        self.create_trend_plots(detailed_df, viz_dir)
        
        # Generate AUC+pAUC hmean visualizations
        self.create_hmean_visualizations(detailed_df, viz_dir)
    
    def create_heatmaps(self, year, comparisons, viz_dir):
        """Create heatmaps for machine type vs system performance."""
        year_viz_dir = os.path.join(viz_dir, f"dcase{year}")
        os.makedirs(year_viz_dir, exist_ok=True)
        
        for metric, pivots in comparisons.items():
            for pivot_name, pivot in pivots.items():
                if pivot.empty:
                    continue
                
                plt.figure(figsize=(12, 8))
                sns.heatmap(
                    pivot, 
                    annot=True, 
                    fmt='.3f', 
                    cmap='RdYlBu_r',
                    center=0.5,
                    cbar_kws={'label': 'Score'}
                )
                plt.title(f'DCASE {year} - {metric} {pivot_name.replace("_", " ").title()}')
                plt.xlabel('System')
                plt.ylabel('Machine Type')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                output_path = os.path.join(year_viz_dir, f'{metric}_{pivot_name}_heatmap.png')
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"  Saved heatmap to {output_path}")
    
    def create_radar_charts(self, system_rankings, viz_dir, detailed_df):
        """Create radar charts for system performance across machine types using AUC+pAUC hmean."""
        # We'll calculate hmean data for radar charts instead of using just AUC source
        for year in self.years:
            year_data = detailed_df[detailed_df['year'] == year]
            if year_data.empty:
                continue
            
            # Calculate hmean data for this year
            hmean_results = self.calculate_machine_hmean(year_data)
            if hmean_results.empty:
                continue
            
            # Filter valid hmean scores for radar chart
            valid_hmean = hmean_results[hmean_results['hmean_score'].notna()]
            if valid_hmean.empty:
                continue
            
            # Get unique machines and systems
            machines = valid_hmean['machine_type'].unique().tolist()
            systems = valid_hmean['system'].unique().tolist()
            
            if len(machines) < 3 or len(systems) < 2:
                continue
            
            # Create pivot table for easier access
            hmean_pivot = valid_hmean.pivot_table(
                index='machine_type',
                columns='system',
                values='hmean_score',
                aggfunc='first'
            )
            
            # Create radar chart
            angles = np.linspace(0, 2 * np.pi, len(machines), endpoint=False).tolist()
            angles += angles[:1]  # Complete the circle
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
            
            colors = plt.cm.Set3(np.linspace(0, 1, len(systems)))
            
            for i, system in enumerate(systems):
                values = []
                for machine in machines:
                    if system in hmean_pivot.columns and machine in hmean_pivot.index:
                        score = hmean_pivot.loc[machine, system]
                        values.append(score if not pd.isna(score) else 0)
                    else:
                        values.append(0)
                
                values += values[:1]  # Complete the circle
                
                ax.plot(angles, values, 'o-', linewidth=2, label=system, color=colors[i])
                ax.fill(angles, values, alpha=0.25, color=colors[i])
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(machines)
            ax.set_ylim(0, 1)
            ax.set_title(f'DCASE {year} - System Performance by Machine Type (AUC+pAUC Harmonic Mean)', y=1.08)
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
            
            output_path = os.path.join(viz_dir, f'dcase{year}_radar_chart_hmean.png')
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  Saved hmean radar chart to {output_path}")
    
    def create_trend_plots(self, detailed_df, viz_dir):
        """Create trend plots showing performance evolution across years."""
        # Overall system performance trends
        auc_source_data = detailed_df[
            (detailed_df['metric_type'] == 'AUC') & 
            (detailed_df['domain'] == 'source')
        ]
        
        if auc_source_data.empty:
            return
        
        # Calculate average performance per system per year
        system_year_avg = auc_source_data.groupby(['year', 'system'])['score'].mean().reset_index()
        
        plt.figure(figsize=(12, 8))
        for system in self.systems:
            system_data = system_year_avg[system_year_avg['system'] == system]
            if not system_data.empty:
                plt.plot(system_data['year'], system_data['score'], 'o-', label=system, linewidth=2, markersize=6)
        
        plt.xlabel('DCASE Year')
        plt.ylabel('Average AUC Score (Source Domain)')
        plt.title('System Performance Trends Across DCASE Years')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(viz_dir, 'system_performance_trends.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved trend plot to {output_path}")
    
    def create_hmean_visualizations(self, detailed_df, viz_dir):
        """Create visualizations for AUC+pAUC harmonic mean by machine type."""
        if detailed_df.empty:
            return
        
        hmean_dir = os.path.join(viz_dir, "hmean_analysis")
        os.makedirs(hmean_dir, exist_ok=True)
        
        # Generate hmean analysis for each year
        for year in self.years:
            year_data = detailed_df[detailed_df['year'] == year]
            if year_data.empty:
                continue
            
            hmean_results = self.calculate_machine_hmean(year_data)
            if hmean_results.empty:
                continue
                
            # Save hmean results to CSV
            hmean_csv_path = os.path.join(hmean_dir, f'dcase{year}_machine_hmean.csv')
            hmean_results.to_csv(hmean_csv_path)
            print(f"  Saved machine hmean data to {hmean_csv_path}")
            
            # Create heatmap for hmean results
            self.create_hmean_heatmap(year, hmean_results, hmean_dir)
            
            # Create bar chart for hmean results
            self.create_hmean_bar_chart(year, hmean_results, hmean_dir)
        
        # Create cross-year hmean comparison
        self.create_cross_year_hmean_comparison(detailed_df, hmean_dir)
    
    def calculate_machine_hmean(self, year_data):
        """Calculate harmonic mean of AUC and pAUC for each machine-system combination."""
        hmean_data = []
        
        # Get unique machine types and systems for this year
        machines = year_data['machine_type'].unique()
        systems = year_data['system'].unique()
        
        for machine in machines:
            machine_data = year_data[year_data['machine_type'] == machine]
            
            for system in systems:
                system_machine_data = machine_data[machine_data['system'] == system]
                
                if system_machine_data.empty:
                    continue
                
                # Collect AUC and pAUC scores for this machine-system combination
                scores = []
                
                # Get AUC scores (both source and target if available)
                auc_data = system_machine_data[system_machine_data['metric_type'] == 'AUC']
                if not auc_data.empty:
                    # Use source domain AUC as primary, fall back to any AUC if source not available
                    auc_source = auc_data[auc_data['domain'] == 'source']['score']
                    if not auc_source.empty:
                        scores.append(auc_source.iloc[0])
                    else:
                        # If no source domain, use any available AUC score
                        auc_any = auc_data['score']
                        if not auc_any.empty:
                            scores.append(auc_any.iloc[0])
                
                # Get pAUC scores
                pauc_data = system_machine_data[system_machine_data['metric_type'] == 'pAUC']
                if not pauc_data.empty:
                    # pAUC usually has combined domain
                    pauc_combined = pauc_data[pauc_data['domain'] == 'combined']['score']
                    if not pauc_combined.empty:
                        scores.append(pauc_combined.iloc[0])
                    else:
                        # If no combined domain, use any available pAUC score
                        pauc_any = pauc_data['score']
                        if not pauc_any.empty:
                            scores.append(pauc_any.iloc[0])
                
                # Calculate harmonic mean if we have both AUC and pAUC
                if len(scores) >= 2:
                    try:
                        machine_hmean = hmean(scores)
                        hmean_data.append({
                            'machine_type': machine,
                            'system': system,
                            'auc_score': scores[0] if len(scores) > 0 else np.nan,
                            'pauc_score': scores[1] if len(scores) > 1 else np.nan,
                            'hmean_score': machine_hmean,
                            'score_count': len(scores)
                        })
                    except:
                        # Handle any calculation errors
                        continue
                elif len(scores) == 1:
                    # If only one metric available, record it but no hmean
                    hmean_data.append({
                        'machine_type': machine,
                        'system': system,
                        'auc_score': scores[0] if 'AUC' in system_machine_data['metric_type'].values else np.nan,
                        'pauc_score': scores[0] if 'pAUC' in system_machine_data['metric_type'].values else np.nan,
                        'hmean_score': np.nan,  # Can't calculate hmean with only one score
                        'score_count': len(scores)
                    })
        
        if hmean_data:
            return pd.DataFrame(hmean_data)
        else:
            return pd.DataFrame()
    
    def create_hmean_heatmap(self, year, hmean_results, hmean_dir):
        """Create heatmap for machine hmean results."""
        if hmean_results.empty:
            return
        
        # Create pivot table for hmean scores
        hmean_pivot = hmean_results.pivot_table(
            index='machine_type',
            columns='system',
            values='hmean_score',
            aggfunc='first'
        )
        
        if hmean_pivot.empty:
            return
        
        # Create heatmap
        plt.figure(figsize=(12, 8))
        mask = hmean_pivot.isna()  # Mask NaN values
        
        sns.heatmap(
            hmean_pivot,
            annot=True,
            fmt='.3f',
            cmap='RdYlBu_r',
            center=0.5,
            cbar_kws={'label': 'Harmonic Mean Score'},
            mask=mask,
            cbar=True
        )
        
        plt.title(f'DCASE {year} - Machine Type Performance (AUC+pAUC Harmonic Mean)')
        plt.xlabel('System')
        plt.ylabel('Machine Type')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        output_path = os.path.join(hmean_dir, f'dcase{year}_machine_hmean_heatmap.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved hmean heatmap to {output_path}")
    
    def create_hmean_bar_chart(self, year, hmean_results, hmean_dir):
        """Create bar chart comparing systems for each machine type."""
        if hmean_results.empty:
            return
        
        # Filter out rows with NaN hmean scores
        valid_hmean = hmean_results[hmean_results['hmean_score'].notna()]
        if valid_hmean.empty:
            return
        
        # Get unique machines that have hmean scores
        machines_with_hmean = valid_hmean['machine_type'].unique()
        
        if len(machines_with_hmean) == 0:
            return
        
        # Create subplots for each machine type
        n_machines = len(machines_with_hmean)
        cols = 3
        rows = (n_machines + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 4*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        if n_machines == 1:
            axes = axes.reshape(1, 1)
        
        for i, machine in enumerate(machines_with_hmean):
            row = i // cols
            col = i % cols
            ax = axes[row, col] if rows > 1 else axes[col]
            
            machine_data = valid_hmean[valid_hmean['machine_type'] == machine]
            machine_data = machine_data.sort_values('hmean_score', ascending=False)
            
            bars = ax.bar(machine_data['system'], machine_data['hmean_score'])
            ax.set_title(f'{machine}')
            ax.set_ylabel('Harmonic Mean')
            ax.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, score in zip(bars, machine_data['hmean_score']):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{score:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Hide empty subplots
        for i in range(n_machines, rows * cols):
            row = i // cols
            col = i % cols
            fig.delaxes(axes[row, col] if rows > 1 else axes[col])
        
        plt.suptitle(f'DCASE {year} - System Performance by Machine Type (AUC+pAUC Harmonic Mean)')
        plt.tight_layout()
        
        output_path = os.path.join(hmean_dir, f'dcase{year}_machine_hmean_bar_chart.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved hmean bar chart to {output_path}")
    
    def create_cross_year_hmean_comparison(self, detailed_df, hmean_dir):
        """Create cross-year comparison of machine hmean performance."""
        if detailed_df.empty:
            return
        
        # Calculate hmean for all years
        all_year_hmeans = []
        for year in self.years:
            year_data = detailed_df[detailed_df['year'] == year]
            if year_data.empty:
                continue
            
            year_hmean = self.calculate_machine_hmean(year_data)
            if not year_hmean.empty:
                year_hmean['year'] = year
                all_year_hmeans.append(year_hmean)
        
        if not all_year_hmeans:
            return
        
        combined_hmean = pd.concat(all_year_hmeans, ignore_index=True)
        
        # Filter valid hmean scores
        valid_combined = combined_hmean[combined_hmean['hmean_score'].notna()]
        if valid_combined.empty:
            return
        
        # Create system performance trends across years
        system_year_hmean = valid_combined.groupby(['year', 'system'])['hmean_score'].mean().reset_index()
        
        plt.figure(figsize=(12, 8))
        for system in self.systems:
            system_data = system_year_hmean[system_year_hmean['system'] == system]
            if not system_data.empty:
                plt.plot(system_data['year'], system_data['hmean_score'], 
                        'o-', label=system, linewidth=2, markersize=6)
        
        plt.xlabel('DCASE Year')
        plt.ylabel('Average Harmonic Mean Score (AUC+pAUC)')
        plt.title('System Performance Trends (Machine-Level AUC+pAUC Harmonic Mean)')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        output_path = os.path.join(hmean_dir, 'cross_year_hmean_trends.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved cross-year hmean trends to {output_path}")
        
        # Save cross-year hmean summary
        cross_year_summary = valid_combined.groupby(['year', 'system'])['hmean_score'].agg(['mean', 'std', 'count']).reset_index()
        cross_year_summary.columns = ['year', 'system', 'mean_hmean', 'std_hmean', 'machine_count']
        
        summary_path = os.path.join(hmean_dir, 'cross_year_hmean_summary.csv')
        cross_year_summary.to_csv(summary_path, index=False)
        print(f"  Saved cross-year hmean summary to {summary_path}")
    
    def run_enhanced_comparison(self):
        """Run the enhanced comparison analysis."""
        print("Enhanced DCASE Comparison Analysis")
        print("=" * 50)
        
        print("Finding files...")
        summaries, detailed_files = self.find_files()
        
        if not summaries or not detailed_files:
            print("No files found!")
            return
        
        print(f"Found {len(summaries)} summary files and {len(detailed_files)} detailed files")
        print(f"Systems: {', '.join(self.systems)}")
        print(f"Years: {', '.join(self.years)}")
        
        print("\nGenerating enhanced reports...")
        self.generate_comprehensive_reports(summaries, detailed_files)
        
        print(f"\nEnhanced comparison completed! Results saved in {self.output_dir}")
        print(f"Check subdirectories for year-specific analyses:")
        for year in self.years:
            print(f"  - dcase{year}_detailed/")
        print("  - cross_year_analysis/")
        print("  - visualizations/")

if __name__ == "__main__":
    comparator = DCASEDetailedComparator()
    comparator.run_enhanced_comparison()
