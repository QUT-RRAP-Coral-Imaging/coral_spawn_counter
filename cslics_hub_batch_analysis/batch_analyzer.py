#! /usr/bin/env python3

# Dorian Tsai
# 2026/02/02
# Batch Analysis Class for CSLICS Hub
# Combines manual counts and automated plotting functionality

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from typing import Dict, List, Optional


class BatchAnalyzer:
    """
    A class to analyze CSLICS batch data, process manual counts,
    and generate plots comparing manual counts with CSLICS estimates.
    """
    
    def __init__(self, data_dir: str, manual_counts_file: str):
        """
        Initialize the BatchAnalyzer.
        
        Parameters:
        -----------
        data_dir : str
            Path to the directory containing batch export data
        manual_counts_file : str
            Name of the CSV file containing manual counts
        """
        self.data_dir = data_dir
        self.manual_counts_file = manual_counts_file
        self.manual_counts_path = os.path.join(data_dir, manual_counts_file)
        
        # Scaling factors for different counting methods
        self.tank_volume = 440.0 # litres
        self.scale_5x50ml = 1.0 / 50.0 * 1000.0 * self.tank_volume  # scale to tank volume
        self.scale_3x1000ml = 1.0 / 1000.0 * 1000.0 * self.tank_volume  # scale to tank volume

        # Data storage
        self.mc = None  # manual counts dataframe
        self.batch_dirs = []  # list of batch directories
        self.sample_data = {}  # dictionary to store sample data per tank
        
        self.sample_data_file = 'sample_data.csv'
        self.configuration_history_file = 'configuration_history.csv'
        self.batch_metadata_file = 'batch_metadata.csv'
        
        # Load and process data
        self._load_manual_counts()
        self._find_batch_directories()
    
    def _load_manual_counts(self):
        """Load and preprocess manual counts data."""
        print(f"Loading manual counts from: {self.manual_counts_path}")
        self.mc = pd.read_csv(self.manual_counts_path)
        
        # Convert relevant columns from string to float64
        self.mc['Manual'] = pd.to_numeric(self.mc['Manual'], errors='coerce')
        self.mc['CSLICS'] = pd.to_numeric(self.mc['CSLICS'], errors='coerce')
        self.mc['subcount1'] = pd.to_numeric(self.mc['subcount1'], errors='coerce')
        self.mc['subcount2'] = pd.to_numeric(self.mc['subcount2'], errors='coerce')
        self.mc['subcount3'] = pd.to_numeric(self.mc['subcount3'], errors='coerce')
        self.mc['subcount4'] = pd.to_numeric(self.mc['subcount4'], errors='coerce')
        self.mc['subcount5'] = pd.to_numeric(self.mc['subcount5'], errors='coerce')

        print(f"Loaded {len(self.mc)} manual count records")
    
    def _find_batch_directories(self):
        """Find all batch directories (starting with 'LAR') in data_dir."""
        self.batch_dirs = [d for d in glob.glob(os.path.join(self.data_dir, 'LAR*')) 
                          if os.path.isdir(d)]
        print(f"\nFound {len(self.batch_dirs)} batch directories:")
        for b in self.batch_dirs:
            print(f" - {os.path.basename(b)}")
    
    def process_tank(self, tank_name: str) -> pd.DataFrame:
        """
        Process data for a specific tank, calculating standard deviations and differences.
        
        Parameters:
        -----------
        tank_name : str
            Name of the tank to process
            
        Returns:
        --------
        pd.DataFrame
            Processed dataframe for the tank with calculated metrics
        """
        # Filter manual counts for this tank
        tank_data = self.mc[self.mc['Tank'] == tank_name].copy()
        
        if tank_data.empty:
            print(f"Warning: No data found for tank {tank_name}")
            return tank_data
        
        # Calculate standard deviation across subcounts
        tank_data['Std'] = np.nanstd(
            tank_data[['subcount1', 'subcount2', 'subcount3', 'subcount4', 'subcount5']], 
            axis=1
        )
        
        # Scale standard deviation based on method
        tank_data['Std_scaled'] = tank_data.apply(
            lambda row: row['Std'] * self.scale_5x50ml if row['Method'] == '5x50ml' 
            else row['Std'] * self.scale_3x1000ml if row['Method'] == '3x1000ml' 
            else row['Std'], 
            axis=1
        )
        
        # Compute difference between manual count and CSLICS estimate
        tank_data['diff'] = abs(tank_data['Manual'] - tank_data['CSLICS'])
        tank_data['percent_diff'] = tank_data['diff'] / tank_data['Manual'] * 100.0
        
        return tank_data
    
    def generate_plot(self, tank_name: str, save_dir: Optional[str] = None):
        """
        Generate a plot for a specific tank with manual counts overlayed on CSLICS data.
        
        Parameters:
        -----------
        tank_name : str
            Name of the tank to plot
        save_dir : str, optional
            Directory to save the plot. If None, uses the tank's batch directory
        """
        # Set seaborn style
        sns.set_style("whitegrid")
        sns.set_context("notebook", font_scale=1.1)
        
        # Find the batch directory for this tank
        batch_dir = None
        for bdir in self.batch_dirs:
            if tank_name in os.path.basename(bdir):
                batch_dir = bdir
                break
        
        if batch_dir is None:
            print(f"Warning: No batch directory found for tank {tank_name}")
            return
        
        # Read configuration history file
        config_history_file = os.path.join(batch_dir, self.configuration_history_file)
        if not os.path.exists(config_history_file):
            print(f"Warning: configuration_history.csv not found in {batch_dir}")
            return
        cf = pd.read_csv(config_history_file)
        species = cf["coral_species"].iloc[0]
        
        # Read camera UUID from batch_metadata file
        batch_metadata_file = os.path.join(batch_dir, self.batch_metadata_file)
        if not os.path.exists(batch_metadata_file):
            print(f"Warning: batch_metadata.csv not found in {batch_dir}")
            return
        bf = pd.read_csv(batch_metadata_file)
        camera_uuid = bf["camera"].iloc[0]

        # Read sample data
        sample_data_file = os.path.join(batch_dir, self.sample_data_file)
        if not os.path.exists(sample_data_file):
            print(f"Warning: sample_data.csv not found in {batch_dir}")
            return
        
        df = pd.read_csv(sample_data_file)
        
        # Extract sample times and tank estimates
        df['timestamp'] = pd.to_datetime(df["timestamp"])
        t_sample = df["timestamp"]
        tank_est = df["Coral (Tank ~)"]

        # do a running average, since we care more about long-term trends, rather than instantaneous fluctuations
        # the imaging rate was all over the place for November, but we want ~ 1 hr
        df = df.set_index('timestamp').sort_index()
        df["tank_est_rolling"] = df['Coral (Tank ~)'].rolling(window=10, min_periods=1).mean()

        # extract coral image counts, and scale them to some calibration (eg. in Nov, runs not scaled properly)
        # TODO
        
        # Get manual counts for this tank
        tank_manual = self.process_tank(tank_name)
        
        # Get seaborn color palette
        colors = sns.color_palette()
        
        # Create plot with seaborn styling
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot CSLICS estimates
        ax.plot(t_sample, tank_est, marker=None, linestyle='-', 
                color=colors[0], label='CSLICS Estimates', 
                alpha=0.8, linewidth=1)
        ax.plot(t_sample, df["tank_est_rolling"], marker=None, linestyle='-',
                color=colors[1], label='CSLICS Estimate Mean',
                alpha=1.0, linewidth=1)
        
        # Overlay manual counts if available
        if not tank_manual.empty:
            # Combine Date and Time columns to create datetime (using day/month/year format)
            manual_times = pd.to_datetime(tank_manual['Date'] + ' ' + tank_manual['Time'], 
                                         format='%d/%m/%Y %H:%M:%S')
            manual_counts = tank_manual['Manual']
            std_scaled = tank_manual['Std_scaled']
            
            # Plot manual counts with error bars
            ax.errorbar(manual_times, manual_counts, yerr=std_scaled, 
                       fmt='o', color=colors[1], markersize=10, capsize=5, capthick=2,
                       label='Manual Counts', alpha=0.9, elinewidth=2.5, linewidth=2)
        
        # Formatting
        ax.set_title(f'CSLICS Tank Estimates vs Manual Counts\nTank: {tank_name} | Camera: {camera_uuid} | Species: {species}', 
                    fontsize=12, fontweight='bold', pad=20)
        ax.set_xlabel('Time', fontsize=12, fontweight='normal')
        ax.set_ylabel('Coral Count', fontsize=12, fontweight='normal')
        ax.legend(loc='best', frameon=True, shadow=True, fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        if save_dir is None:
            save_dir = batch_dir
        
        output_path = os.path.join(save_dir, f'{tank_name}_comparison_plot.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
        plt.close()
        
        # Reset to default style
        sns.reset_defaults()
    
    def print_tank_summary(self, tank_name: str):
        """
        Print a summary of manual counts and CSLICS estimates for a tank.
        
        Parameters:
        -----------
        tank_name : str
            Name of the tank to summarize
        """
        tank_data = self.process_tank(tank_name)
        
        if tank_data.empty:
            return
        
        print(f'\n{"="*80}')
        print(f'Tank: {tank_name}')
        print(f'{"="*80}')
        
        for idx, row in tank_data.iterrows():
            date = row['Date']
            time = row['Time']
            species = row['Species']
            manual_count = row['Manual']
            std_dev = row['Std_scaled']
            meth = row['Method']
            cslics_estimate = row['CSLICS']
            diff = row['diff']
            pdiff = row['percent_diff']
            
            print(f'Date: {date} Time: {time}\tSpecies: {species}\t'
                  f'Manual: {manual_count:06.0f}\tStd Dev: {std_dev:05.0f}\t'
                  f'CSLICS: {cslics_estimate:06.0f}\tDiff: {diff:06.0f}\t'
                  f'% Diff: {pdiff:02.2f}%')
    
    def run_all_tanks(self, generate_plots: bool = True, print_summaries: bool = True):
        """
        Process all tanks, optionally generating plots and printing summaries.
        
        Parameters:
        -----------
        generate_plots : bool, default=True
            Whether to generate comparison plots for each tank
        print_summaries : bool, default=True
            Whether to print summary statistics for each tank
        """
        # Get unique tank names from manual counts
        tanks = self.mc['Tank'].unique()
        
        print(f"\n{'='*80}")
        print(f"Processing {len(tanks)} tanks")
        print(f"{'='*80}")
        
        for tank in tanks:
            if print_summaries:
                self.print_tank_summary(tank)
            
            if generate_plots:
                self.generate_plot(tank)
        
        print(f"\n{'='*80}")
        print("Processing complete!")
        print(f"{'='*80}")


def main():
    """Main function to run the batch analyzer."""
    # Configuration
    # data_dir = '/home/dtsai/Data/cslics_datasets/cslics_batch_export_2025/2025dec'
    # manual_counts_file = 'Dec2025_CSLICS_Larval Cultures(Culture Density).csv'
    
    data_dir = '/home/dtsai/Data/cslics_datasets/cslics_batch_export_2025/2025nov'
    manual_counts_file = 'Nov2025_CSLICS_Larval Cultures(Culture Density).csv'
    
    # Create analyzer instance
    analyzer = BatchAnalyzer(data_dir, manual_counts_file)
    
    # Run analysis on all tanks
    analyzer.run_all_tanks(generate_plots=True, print_summaries=True)
    
    # Optional: Interactive mode for exploration
    import code
    code.interact(local=dict(globals(), **locals()))


if __name__ == '__main__':
    main()
