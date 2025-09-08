# a new coral_spawn_countrer.py class
# This class will handle the processing and plotting of coral spawn data with respect to manual counts
# based on CslicsDataProcessor.py

# read in a configuration file, which defines what we want to plot, where the detection data is, what models were used, where the manual counts are
# two modes of operation, surface and subsurface

# for the manual counts
# read in the manual counts file (the method is shown in read_manual_counts in CslicsDataProcessor.py)
# plot_manual_counts (plot_manual_counts in CslicsDataProcessor.py)

# for the surface plot
# - read in the surface model weights
# - extract the number of classes and the names of the classes using methods from model_manager.py
# - read in the detection data files from the specified directory os.path.join(det_dir, "detections_text") folder
# - for each detection file, read in the data, extract the counts per class, and store in a dictionary
# - create a pandas dataframe from the dictionary
# - create plots using matplotlib
# the plot should be titled with the cslics_uuid, the model name, "surface detections", and the coral species and tank number and the date
# - I want to plot the counts per class over time, with one line per class
# in the x-axis, I want to show time since spawning, where 0 is the time of the first image, and then it should be in hours up to the end of the data
# - save the plots to the specified output directory

# for the subsurface plot
# - read in the subsurface model weights
# - it should be the same as the subsurface, but with "surface detections" in the title replaced with "subsurface detections"

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from datetime import datetime, timedelta
import traceback
import json

# Add paths for imports
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))

# Import our custom classes
from config.plot_config_manager import PlotConfigManager, CoralSpawnConfig
from manual_counts.manual_counts_processor import ManualCountsProcessor
from visualisation.tank_count_plotter import TankCountPlotter
from processing.invalid_times_processor import InvalidTimesProcessor
from models.model_manager import ModelManager

class CoralSpawnCounter:
    """
    Main class for coral spawn counting analysis.
    Handles processing and plotting of coral spawn data with respect to manual counts.
    Supports both surface and subsurface detection modes.
    """
    
    def __init__(self, config_path: str):
        """
        Initialize the coral spawn counter with a configuration file.
        
        Args:
            config_path: Path to the JSON configuration file
        """
        self.config_manager = PlotConfigManager(config_path)
        self.config = self.config_manager.get_config()
        
        # Initialize processors
        self.manual_processor = ManualCountsProcessor(self.config_manager)
        self.tank_plotter = TankCountPlotter(self.config_manager)
        self.invalid_processor = InvalidTimesProcessor.from_config_file(config_path)
        
        # Initialize device (no GPU needed for plotting)
        self.device = torch.device('cpu')
        
        # Initialize model manager
        self.model_manager = ModelManager(self.config, self.device)
        
        
        # Store manual counts data
        self.manual_data = None
        self.nearest_day = None
        
    def process_manual_counts(self, show_plot=False):
        """
        Process manual counts data and generate plots.
        
        Args:
            show_plot: Whether to display plots interactively
            
        Returns:
            dict: Manual counts data
        """
        print("Processing manual counts...")
        self.manual_data = self.manual_processor.process_all_manual_counts(show_plot=show_plot)
        
        if self.manual_data:
            # Calculate nearest day for time reference
            counts_time = self.manual_data['counts_time']
            self.nearest_day = counts_time[0].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
            print(f"Manual counts processed successfully. Reference day: {self.nearest_day}")
        else:
            print("Failed to process manual counts.")
            
        return self.manual_data
    
    
    
    def read_detection_files(self, is_surface=True):
        """
        Read surface detection files from the configured surface detection directory.
        
        Returns:
            list: List of detection file paths
        """
        if is_surface:
            det_dir = self.config_manager.get_surface_detection_dir()
        else:
            det_dir = self.config_manager.get_subsurface_detection_dir()
        
        # Look for detections_text subdirectory first
        detections_text_dir = os.path.join(det_dir, "detections_text")
        if os.path.exists(detections_text_dir):
            det_dir = detections_text_dir

        if not os.path.exists(det_dir):
            raise FileNotFoundError(f"Detection directory not found: {det_dir}")
        # recursively search for all .json files in and below the directory
        detection_files = sorted(Path(det_dir).rglob('*.json'))
        print(f"Found {len(detection_files)} detection files in {det_dir}")
        return detection_files
    
    def parse_detection_file(self, file_path):
        """
        Parse a single surface detection file.
        
        Args:
            file_path: Path to the detection file
            
        Returns:
            dict: Dictionary with timestamp and class counts
        """
        try:
            # Extract timestamp from filename
            filename = Path(file_path).stem
            # Adjust based on your filename format
            time_str = filename[9:-15] if len(filename) > 24 else filename
            timestamp = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
            
            # Read detection data
            
            class_counts = {}
            if os.path.getsize(file_path) > 0:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Adjust this based on your JSON structure
                    if 'detections' in data:
                        for detection in data['detections']:
                            class_id = detection.get('class_id', detection.get('class'))
                            if class_id is not None:
                                class_counts[class_id] = class_counts.get(class_id, 0) + 1
            return {
                'timestamp': timestamp,
                'class_counts': class_counts
            }
        
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return None
    
    def process_surface_detections(self, show_plots=False):
        """
        Process surface detections and create plots.
        
        Args:
            show_plots: Whether to display plots interactively
            
        Returns:
            pd.DataFrame: DataFrame with surface detection data
        """
        if not self.manual_data or not self.nearest_day:
            raise ValueError("Manual counts must be processed first. Call process_manual_counts().")
            
        print("Processing surface detections...")
        
        # Read detection files
        try:
            detection_files = self.read_detection_files()
        except FileNotFoundError as e:
            print(f"Surface detection files not found: {e}")
            return None
            
        # Process each file
        detection_data = []
        class_names = self.model_manager.surface_classes
        
        # Process detection files
        for file_path in detection_files:
            parsed_data = self.parse_detection_file(file_path)
            if parsed_data:
                detection_data.append(parsed_data)
        
        if not detection_data:
            print("No valid detection data found.")
            return None
            
        # Create DataFrame
        df_data = []
        for data in detection_data:
            row = {
                'timestamp': data['timestamp'],
                'hours_since_spawning': (data['timestamp'] - self.nearest_day).total_seconds() / 3600
            }
            
            # Add class counts - map class_id to class_name using list index
            for class_id, count in data['class_counts'].items():
                # Convert class_id to int if it's not already
                class_id = int(class_id)
                # Use class_id as index into the class_names list
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                else:
                    class_name = f'class_{class_id}'  # fallback for out-of-range IDs
                row[class_name] = count
                
            df_data.append(row)
        
        df = pd.DataFrame(df_data).fillna(0)
        
        # Create plots - pass the class_names list instead of dict
        self.plot_surface_detections(df, class_names, show_plots)
        
        # Save data
        self.save_surface_detection_data(df)
        
        print("Surface detection analysis completed successfully.")
        return df
    
    def plot_surface_detections(self, df, class_names, show_plots=False):
        """
        Create plots for surface detections.
        
        Args:
            df: DataFrame with surface detection data
            class_names: List of class names (index corresponds to class ID)
            show_plots: Whether to display plots interactively
        """
        # Get class columns (exclude timestamp and hours_since_spawning)
        class_columns = [col for col in df.columns if col not in ['timestamp', 'hours_since_spawning']]
        
        if not class_columns:
            print("No class data found for plotting.")
            return
            
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each class
        for class_col in class_columns:
            ax.plot(df['hours_since_spawning'], df[class_col], 
                   marker='o', label=class_col, linewidth=2, markersize=4)
        
        # Get surface model name for title
        surface_model_name = self.config_manager.get_surface_model_name()
        
        # Customize plot
        ax.set_xlabel('Hours since spawning')
        ax.set_ylabel('Detection counts')
        ax.set_title(f'Surface Detections - {self.config.cslics_uuid}\n'
                    f'Model: {surface_model_name}\n'
                    f'Species: {self.config.coral_species} - Tank: {self.config.tank_sheet_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(self.config.base_detection_dir, self.config.cslics_uuid, "plots")
        os.makedirs(output_dir, exist_ok=True)
        
        plot_filename = f'Surface_detections_{self.config.tank_sheet_name}_{self.config.cslics_uuid}.png'
        plot_path = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Surface detection plot saved to: {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    def save_surface_detection_data(self, df):
        """
        Save surface detection data to CSV.
        
        Args:
            df: DataFrame with surface detection data
        """
        output_dir = os.path.join(self.config.base_detection_dir, self.config.cslics_uuid, "plots")
        os.makedirs(output_dir, exist_ok=True)
        
        csv_filename = f'Surface_detection_data_{self.config.tank_sheet_name}_{self.config.cslics_uuid}.csv'
        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Surface detection data saved to: {csv_path}")
    
    def process_subsurface_detections(self, show_plots=False):
        """
        Process subsurface detections using the tank count plotter.
        
        Args:
            show_plots: Whether to display plots interactively
            
        Returns:
            dict: Analysis results
        """
        if not self.manual_data or not self.nearest_day:
            raise ValueError("Manual counts must be processed first. Call process_manual_counts().")
            
        print("Processing subsurface detections...")
        
        # Load invalid time ranges
        # invalid_indices = self.invalid_processor.find_invalid_file_indices(
        #     self.tank_plotter.read_detections()
        # )
        
        # Run full subsurface analysis
        results = self.tank_plotter.run_full_analysis(
            det_dir=self.config_manager.get_subsurface_detection_dir(),
            manual_counts=self.manual_data['counts'],
            manual_std=self.manual_data['std'],
            manual_times=self.manual_processor.convert_to_decimal_days(
                self.manual_data['counts_time'], self.nearest_day
            ),
            nearest_day=self.nearest_day,
            invalid_indices=None, # temporary disable invalid times
            show_plots=show_plots
        )
        
        if results:
            print("Subsurface detection analysis completed successfully.")
        else:
            print("Failed to process subsurface detections.")
            
        return results
    
    def run_full_analysis(self, show_plots=False, include_surface=True, include_subsurface=True):
        """
        Run the complete analysis pipeline.
        
        Args:
            show_plots: Whether to display plots interactively
            include_surface: Whether to process surface detections
            include_subsurface: Whether to process subsurface detections
            
        Returns:
            dict: Complete analysis results
        """
        results = {}
        
        try:
            # Process manual counts first
            print("="*60)
            print("CORAL SPAWN COUNTER - FULL ANALYSIS")
            print("="*60)
            print(f"Configuration: {self.config.cslics_uuid} - {self.config.coral_species}")
            print(f"Surface Model: {self.config_manager.get_surface_model_name()}")
            print(f"Subsurface Model: {self.config_manager.get_subsurface_model_name()}")
            print(f"Tank: {self.config.tank_sheet_name}")
            print(f"Surface Detection Dir: {self.config_manager.get_surface_detection_dir()}")
            print(f"Subsurface Detection Dir: {self.config_manager.get_subsurface_detection_dir()}")
            print("="*60)
            
            manual_data = self.process_manual_counts(show_plot=show_plots)
            if not manual_data:
                raise ValueError("Failed to process manual counts.")
            results['manual_data'] = manual_data
            
            # Process surface detections
            if include_surface:
                print("\n" + "-"*40)
                print("SURFACE DETECTION ANALYSIS")
                print("-"*40)
                surface_df = self.process_surface_detections(show_plots=show_plots)
                results['surface_data'] = surface_df
            
            # Process subsurface detections
            if include_subsurface:
                print("\n" + "-"*40)
                print("SUBSURFACE DETECTION ANALYSIS")
                print("-"*40)
                subsurface_results = self.process_subsurface_detections(show_plots=show_plots)
                results['subsurface_results'] = subsurface_results
            
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            
            # Print summary
            self.print_analysis_summary(results)
            
            return results
            
        except Exception as e:
            print(f"Error in full analysis: {e}")
            return results
    
    def print_analysis_summary(self, results):
        """
        Print a summary of the analysis results.
        
        Args:
            results: Dictionary containing analysis results
        """
        print("\nANALYSIS SUMMARY:")
        print("-" * 30)
        
        if 'manual_data' in results and results['manual_data']:
            manual_count = len(results['manual_data']['counts'])
            print(f"Manual counts processed: {manual_count} data points")
        
        if 'subsurface_results' in results and results['subsurface_results']:
            subsurface_count = len(results['subsurface_results']['image_counts'])
            print(f"Subsurface detections processed: {subsurface_count} batches")
            
            if 'errors' in results['subsurface_results']:
                error_count = len(results['subsurface_results']['errors'])
                if error_count > 0:
                    print(f"Subsurface detection errors: {error_count}")
        
        if 'surface_data' in results and results['surface_data'] is not None:
            surface_count = len(results['surface_data'])
            print(f"Surface detections processed: {surface_count} time points")
        
        print(f"\nOutput directory: {os.path.join(self.config.base_detection_dir, self.config.cslics_uuid, 'plots')}")
        print(f"Surface detection directory: {self.config_manager.get_surface_detection_dir()}")
        print(f"Subsurface detection directory: {self.config_manager.get_subsurface_detection_dir()}")
    
    def get_config_summary(self):
        """Get a summary of the current configuration."""
        return self.manual_processor.get_config_summary()


# Example usage and main execution
if __name__ == "__main__":
    # Configuration file path
    config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plot_config_202312_t4_alor_cslics08.json"
    
    try:
        # Initialize the coral spawn counter
        counter = CoralSpawnCounter(config_path)
        
        # Print configuration summary
        print("Configuration Summary:")
        config_summary = counter.get_config_summary()
        for key, value in config_summary.items():
            print(f"  {key}: {value}")
        
        # Run full analysis
        results = counter.run_full_analysis(
            show_plots=False,  # Set to True to display plots interactively
            include_surface=True,
            include_subsurface=True
        )
        
        if results:
            print("\nAnalysis completed successfully!")
        else:
            print("\nAnalysis failed or returned no results.")
            
    except Exception as e:
        print(f"Error running coral spawn counter: {e}")
        traceback.print_exc()




