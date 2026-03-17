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
        
        # Initialize separate model managers for surface and subsurface
        self.surface_model_manager = None
        self.subsurface_model_manager = None
        
        # Initialize model managers if weights are available
        if os.path.exists(self.config.surface_weights_path):
            self.surface_model_manager = ModelManager(self.config, self.device, model_type='surface')
        
        if os.path.exists(self.config.subsurface_weights_path):
            self.subsurface_model_manager = ModelManager(self.config, self.device, model_type='subsurface')
        
        # Store manual counts data
        self.manual_data = None
        self.nearest_day = None

    def process_manual_counts(self, show_plot=False):
        """
        Process manual counts data and generate plots.
        
        Args:
            show_plot: Whether to display plots interactively
            
        Returns:
            dict: Manual counts dictionary with keys: counts, std, decimal_days, counts_time, camera_uuid, species
        """
        print("Processing manual counts...")
        self.manual_data = self.manual_processor.process_all_manual_counts(show_plot=show_plot)
        
        if self.manual_data is not None and isinstance(self.manual_data, dict):
            if 'counts' in self.manual_data and len(self.manual_data['counts']) > 0:
                # Calculate nearest day for time reference
                counts_time = self.manual_data['counts_time']
                if isinstance(counts_time[0], datetime):
                    self.nearest_day = counts_time[0].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
                else:
                    # If counts_time is already in decimal days, we need the nearest_day from elsewhere
                    # For now, assume it's the first day
                    self.nearest_day = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                print(f"Manual counts processed successfully. Reference day: {self.nearest_day}")
                print(f"Manual data keys: {list(self.manual_data.keys())}")
                print(f"Manual counts: {len(self.manual_data['counts'])} data points")
                print(f"Camera UUID: {self.manual_data.get('camera_uuid', 'N/A')}")
                print(f"Species: {self.manual_data.get('species', 'N/A')}")
            else:
                print("Manual counts dictionary is empty or missing 'counts' key.")
                return None
        else:
            print("Failed to process manual counts or unexpected data type.")
            return None
            
        return self.manual_data

    def read_detection_files(self, detection_type='surface'):
        """
        Read detection files from the configured detection directory.
        
        Args:
            detection_type: Either 'surface' or 'subsurface'
            
        Returns:
            list: List of detection file paths
        """
        if detection_type == 'surface':
            det_dir = self.config_manager.get_surface_detection_dir()
        elif detection_type == 'subsurface':
            det_dir = self.config_manager.get_subsurface_detection_dir()
        else:
            raise ValueError(f"Invalid detection_type: {detection_type}. Must be 'surface' or 'subsurface'")
        
        # Look for detections_text subdirectory for JSON files
        # if detection_type == 'surface':
        #     detections_text_dir = os.path.join(det_dir, "detections_text")
        #     if os.path.exists(detections_text_dir):
        #         det_dir = detections_text_dir
        #     file_pattern = '*_det.json'  # Surface uses text files
        # else:
        file_pattern = '*_det.json'  # Subsurface uses JSON files

        if not os.path.exists(det_dir):
            raise FileNotFoundError(f"{detection_type.capitalize()} detection directory not found: {det_dir}")
        
        # Recursively search for detection files
        detection_files = sorted(Path(det_dir).rglob(file_pattern))
        print(f"Found {len(detection_files)} {detection_type} detection files in {det_dir}")
        return detection_files

    def get_class_names(self, detection_type='surface'):
        """
        Get class names for the specified detection type.
        
        Args:
            detection_type: Either 'surface' or 'subsurface'
            
        Returns:
            list: List of class names
        """
        if detection_type == 'surface' and self.surface_model_manager:
            return self.surface_model_manager.get_class_names()
        elif detection_type == 'subsurface' and self.subsurface_model_manager:
            return self.subsurface_model_manager.get_class_names()
        else:
            print(f"Warning: No model manager available for {detection_type} detections")
            return []

    def get_model_name(self, detection_type='surface'):
        """
        Get model name for the specified detection type.
        
        Args:
            detection_type: Either 'surface' or 'subsurface'
            
        Returns:
            str: Model name
        """
        if detection_type == 'surface' and self.surface_model_manager:
            return self.surface_model_manager.get_model_name()
        elif detection_type == 'subsurface' and self.subsurface_model_manager:
            return self.subsurface_model_manager.get_model_name()
        else:
            print(f"Warning: No model manager available for {detection_type} detections")
            return f"{detection_type}_model"

    def parse_detection_file(self, file_path, detection_type='surface'):
        """
        Parse a single detection file.
        
        Args:
            file_path: Path to the detection file
            detection_type: Either 'surface' or 'subsurface'
            
        Returns:
            dict: Dictionary with timestamp and class counts
        """
        try:
            # Extract timestamp from filename
            filename = Path(file_path).stem
            
            # Extract timestamp - adjust pattern based on your filename format
            # Common patterns: cslics08_20231205_142030_det.json or similar
            import re
            time_match = re.search(r'(\d{8}_\d{6})', filename)
            if time_match:
                time_str = time_match.group(1)
                timestamp = datetime.strptime(time_str, "%Y%m%d_%H%M%S")
            else:
                print(f"Warning: Could not extract timestamp from {filename}")
                return None
            
            class_counts = {}
            
            # Read JSON detection data
            if os.path.getsize(file_path) > 0:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                detections = data.get('detections', [])
                
                for detection in detections:
                    confidence = detection.get('confidence', 0)
                    if confidence >= self.config.confidence_threshold:
                        class_id = detection.get('class_id', 0)
                        class_counts[class_id] = class_counts.get(class_id, 0) + 1
            
            if self.config.verbose:
                print(f"Parsed {filename}: {len(class_counts)} class types, {sum(class_counts.values())} total detections")
            
            return {
                'timestamp': timestamp,
                'class_counts': class_counts
            }
            
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            traceback.print_exc()
            return None

    def process_detections(self, detection_type='surface', show_plots=False):
        """
        Generic method to process detections and create plots.
        
        Args:
            detection_type: Either 'surface' or 'subsurface'
            show_plots: Whether to display plots interactively
            
        Returns:
            pd.DataFrame: DataFrame with detection data
        """
        if not self.manual_data or not self.nearest_day:
            raise ValueError("Manual counts must be processed first. Call process_manual_counts().")
            
        print(f"Processing {detection_type} detections...")
        
        # Check if model manager exists
        model_manager = (self.surface_model_manager if detection_type == 'surface' 
                        else self.subsurface_model_manager)
        if not model_manager:
            print(f"Error: No {detection_type} model manager available. Check model weights path.")
            return None
        
        # Read detection files
        try:
            detection_files = self.read_detection_files(detection_type)
        except FileNotFoundError as e:
            print(f"{detection_type.capitalize()} detection files not found: {e}")
            return None
            
        # Get class names
        class_names = self.get_class_names(detection_type)
        print(f"Model has {len(class_names)} classes: {class_names}")
        
        # Process each file
        detection_data = []
        successful_parses = 0
        for file_path in detection_files:
            parsed_data = self.parse_detection_file(file_path, detection_type)
            if parsed_data:
                detection_data.append(parsed_data)
                successful_parses += 1
        
        print(f"Successfully parsed {successful_parses} out of {len(detection_files)} files")
        
        if not detection_data:
            print(f"No valid {detection_type} detection data found.")
            return None
            
        # Create DataFrame
        df_data = []
        for data in detection_data:
            row = {
                'timestamp': data['timestamp'],
                'hours_since_spawning': (data['timestamp'] - self.nearest_day).total_seconds() / 3600
            }
            
            # Initialize all class columns to 0
            for i, class_name in enumerate(class_names):
                row[class_name] = 0
            
            # Add class counts - map class_id to class_name using list index
            for class_id, count in data['class_counts'].items():
                # Convert class_id to int if it's not already
                class_id = int(class_id)
                # Use class_id as index into the class_names list
                if 0 <= class_id < len(class_names):
                    class_name = class_names[class_id]
                    row[class_name] = count
                else:
                    # Handle out-of-range class IDs
                    unknown_class = f'class_{class_id}'
                    row[unknown_class] = count
                    print(f"Warning: Unknown class_id {class_id} found, added as {unknown_class}")
                
            df_data.append(row)
        
        df = pd.DataFrame(df_data).fillna(0)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Created DataFrame with {len(df)} rows and columns: {list(df.columns)}")
        
        # Check if we have any class data
        class_columns = [col for col in df.columns if col not in ['timestamp', 'hours_since_spawning']]
        if not class_columns:
            print(f"Warning: No class columns found in DataFrame. Available columns: {list(df.columns)}")
            print(f"Class names from model: {class_names}")
            return df
        
        # Create plots
        self.plot_detections(df, class_names, detection_type, show_plots)
        
        # Save data
        self.save_detection_data(df, detection_type)
        
        print(f"{detection_type.capitalize()} detection analysis completed successfully.")
        return df

    def plot_detections(self, df, class_names, detection_type='surface', show_plots=False):
        """
        Create plots for detections.
        
        Args:
            df: DataFrame with detection data
            class_names: List of class names (index corresponds to class ID)
            detection_type: Either 'surface' or 'subsurface'
            show_plots: Whether to display plots interactively
        """
        # Get class columns (exclude timestamp and hours_since_spawning)
        class_columns = [col for col in df.columns if col not in ['timestamp', 'hours_since_spawning']]
        
        if not class_columns:
            print(f"No class data found for {detection_type} plotting.")
            return
            
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each class
        for class_col in class_columns:
            ax.plot(df['hours_since_spawning'], df[class_col], 
                   marker='o', label=class_col, linewidth=2, markersize=4)
        
        # Get model name for title
        model_name = self.get_model_name(detection_type)
        
        # Customize plot
        ax.set_xlabel('Hours since spawning')
        ax.set_ylabel('Detection counts')
        ax.set_title(f'{detection_type.capitalize()} Detections - {self.config.cslics_uuid}\n'
                    f'Model: {model_name}\n'
                    f'Species: {self.config.coral_species} - Tank: {self.config.tank_sheet_name}')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # Use same directory structure as coral_spawn_predictor.py
        # Create plots directory structure: base_detection_dir/cslics_uuid/plots
        plots_dir = os.path.join(self.config.base_detection_dir, self.config.cslics_uuid, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        plot_filename = f'{detection_type}_detections_{self.config.tank_sheet_name}_{self.config.cslics_uuid}.png'
        plot_path = os.path.join(plots_dir, plot_filename)
        plt.savefig(plot_path, dpi=600, bbox_inches='tight')
        print(f"{detection_type.capitalize()} detection plot saved to: {plot_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()

    def save_detection_data(self, df, detection_type='surface'):
        """
        Save detection data to CSV using same directory structure as coral_spawn_predictor.py.
        
        Args:
            df: DataFrame with detection data
            detection_type: Either 'surface' or 'subsurface'
        """
        # Use same directory structure as coral_spawn_predictor.py
        # Create data directory structure: base_detection_dir/cslics_uuid/data
        data_dir = os.path.join(self.config.base_detection_dir, self.config.cslics_uuid, 'data')
        os.makedirs(data_dir, exist_ok=True)
        
        csv_filename = f'{detection_type}_detection_data_{self.config.tank_sheet_name}_{self.config.cslics_uuid}.csv'
        csv_path = os.path.join(data_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"{detection_type.capitalize()} detection data saved to: {csv_path}")

    def process_surface_detections(self, show_plots=False):
        """
        Process surface detections and create plots.
        
        Args:
            show_plots: Whether to display plots interactively
            
        Returns:
            pd.DataFrame: DataFrame with surface detection data
        """
        return self.process_detections('surface', show_plots)

    def process_subsurface_detections(self, show_plots=False):
        """
        Process subsurface detections using the tank cousurface nt plotter.
        
        Args:
            show_plots: Whether to display plots interactively
            
        Returns:
            dict: Analysis results
        """
        # Check if manual data is available (expecting dictionary)
        if (not isinstance(self.manual_data, dict)):
            raise ValueError("Manual counts must be processed first. Call process_manual_counts().")
            
        print("Processing subsurface detections...")
        
        # Extract arrays from manual data dictionary
        manual_counts = self.manual_data['counts']
        manual_std = self.manual_data['std']
        manual_times = self.manual_processor.convert_to_decimal_days(
            self.manual_data['counts_time'], self.nearest_day
        )
        
        # Create batch histogram directory
        batch_histogram_dir = os.path.join(
            self.config.base_detection_dir, 
            self.config.cslics_uuid, 
            'plots', 
            'batch_histogram'
        )
        os.makedirs(batch_histogram_dir, exist_ok=True)
        
        # Run full subsurface analysis using tank plotter
        results = self.tank_plotter.run_full_analysis(
            det_dir=self.config_manager.get_subsurface_detection_dir(),
            manual_counts=manual_counts,
            manual_std=manual_std,
            manual_times=manual_times,
            nearest_day=self.nearest_day,
            invalid_indices=None,  # temporary disable invalid times
            show_plots=show_plots,
            batch_histogram_dir=batch_histogram_dir,
            detection_type='subsurface'
        )
        
        if results:
            print("Subsurface detection analysis completed successfully.")
            print(f"Batch histograms saved to: {batch_histogram_dir}")
        else:
            print("Failed to process subsurface detections.")
            
        return results

    def run_full_analysis(self, show_plots=False, include_surface=True, include_subsurface=True, export_counts=True):
        """
        Run the complete analysis pipeline.
        
        Args:
            show_plots: Whether to display plots interactively
            include_surface: Whether to process surface detections
            include_subsurface: Whether to process subsurface detections
            export_counts: Whether to export count data to JSON file
            
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
            
            # Print calibration parameters
            print(f"Surface Calibration - Index: {self.config_manager.get_surface_calibration_idx()}, "
                  f"Window Size: {self.config_manager.get_surface_calibration_window_size()}, "
                  f"Window Shift: {self.config_manager.get_surface_calibration_window_shift()}")
            print(f"Subsurface Calibration - Index: {self.config_manager.get_subsurface_calibration_idx()}, "
                  f"Window Size: {self.config_manager.get_subsurface_calibration_window_size()}, "
                  f"Window Shift: {self.config_manager.get_subsurface_calibration_window_shift()}")
            print("="*60)
            
            manual_data = self.process_manual_counts(show_plot=show_plots)
            if manual_data is None:
                raise ValueError("Failed to process manual counts.")
            results['manual_data'] = manual_data
            
            # Process surface detections
            if include_surface:
                print("\n" + "-"*40)
                print("SURFACE DETECTION ANALYSIS")
                print("-"*40)
                surface_df = self.process_surface_detections(show_plots=show_plots)
                results['surface_data'] = surface_df

                # Plot surface tank estimates with manual data
                if surface_df is not None:
                    class_names = self.get_class_names('surface')
                    
                    # Extract manual data for surface tank estimate (dictionary)
                    manual_counts = manual_data['counts']
                    manual_std = manual_data['std']
                    manual_times = manual_data['counts_time']
                    
                    # Add submersion_time and surface calibration parameters to tank plotter
                    if hasattr(self.config, 'submersion_time'):
                        self.tank_plotter.submersion_time = self.config.submersion_time
                        self.tank_plotter.submersion_time_decimal = self.manual_processor.convert_to_decimal_days(
                            [pd.to_datetime(self.config.submersion_time, format="%Y-%m-%d_%H-%M-%S")], self.nearest_day)
                        
                    # Set surface calibration parameters
                    self.tank_plotter.surface_calibration_idx = self.config_manager.get_surface_calibration_idx()
                    self.tank_plotter.surface_calibration_window_size = self.config_manager.get_surface_calibration_window_size()
                    self.tank_plotter.surface_calibration_window_shift = self.config_manager.get_surface_calibration_window_shift()
                    
                    times_hours, total_counts_scaled, scale_factor = self.tank_plotter.surface_tank_estimate(
                        surface_df, class_names, manual_counts, manual_std, manual_times, 
                        self.nearest_day, show_plots, apply_calibration=True, 
                        calibration_idx=self.config_manager.get_surface_calibration_idx()
                    )
                    
                    
                    print(f"Surface tank estimate completed with scale factor: {scale_factor:.4f}")
    
            # Process subsurface detections
            if include_subsurface:
                print("\n" + "-"*40)
                print("SUBSURFACE DETECTION ANALYSIS")
                print("-"*40)
                
                # Set subsurface calibration parameters
                self.tank_plotter.subsurface_calibration_idx = self.config_manager.get_subsurface_calibration_idx()
                self.tank_plotter.subsurface_calibration_window_size = self.config_manager.get_subsurface_calibration_window_size()
                self.tank_plotter.subsurface_calibration_window_shift = self.config_manager.get_subsurface_calibration_window_shift()
                
                subsurface_results = self.process_subsurface_detections(show_plots=show_plots)
                results['subsurface_results'] = subsurface_results
            
            print("\n" + "="*60)
            print("ANALYSIS COMPLETE")
            print("="*60)
            
            # Print summary
            self.print_analysis_summary(results)
            
            # Export counts data to JSON if requested
            if export_counts:
                print("\n" + "-"*40)
                print("EXPORTING COUNT DATA")
                print("-"*40)
                self.export_counts_to_json(
                    manual_counts=results.get('manual_data'),
                    subsurface_counts=results.get('subsurface_results'),
                    surface_counts=results.get('surface_data'),
                    include_metadata=True
                )
            
            return results
            
        except Exception as e:
            print(f"Error in full analysis: {e}")
            traceback.print_exc()
            return results

    def print_analysis_summary(self, results):
        """
        Print a summary of the analysis results.
        
        Args:
            results: Dictionary containing analysis results
        """
        print("\nANALYSIS SUMMARY:")
        print("-" * 30)
        
        if 'manual_data' in results and results['manual_data'] is not None:
            # Handle dictionary only
            manual_count = len(results['manual_data'].get('counts', []))
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
        
        # Update output directory paths to match coral_spawn_predictor.py structure
        plots_dir = os.path.join(self.config.base_detection_dir, self.config.cslics_uuid, 'plots')
        data_dir = os.path.join(self.config.base_detection_dir, self.config.cslics_uuid, 'data')
        batch_histogram_dir = os.path.join(plots_dir, 'batch_histogram')
        
        print(f"\nOutput directories:")
        print(f"  Plots: {plots_dir}")
        print(f"  Batch Histograms: {batch_histogram_dir}")
        print(f"  Data: {data_dir}")
        print(f"Surface detection directory: {self.config_manager.get_surface_detection_dir()}")
        print(f"Subsurface detection directory: {self.config_manager.get_subsurface_detection_dir()}")

    def get_config_summary(self):
        """Get a summary of the current configuration."""
        return self.manual_processor.get_config_summary()

    def export_counts_to_json(self, manual_counts, subsurface_counts=None, surface_counts=None, 
                         output_path=None, include_metadata=True):
        """
        Export manual counts, subsurface counts, and surface counts to a JSON file.
        This allows recreation of the plots produced for surface and subsurface analysis.
        
        Args:
            manual_counts: Manual count data (dict or array-like)
            subsurface_counts: Subsurface detection data (dict or DataFrame, optional)
            surface_counts: Surface detection data (dict or DataFrame, optional)
            output_path: Path to save JSON file. If None, uses default location.
            include_metadata: Whether to include configuration metadata
            
        Returns:
            str: Path to the saved JSON file
        """
        try:
            # Prepare the export data dictionary
            export_data = {
                "export_timestamp": datetime.now().isoformat(),
                "data_version": "1.0"
            }
            
            # Add metadata if requested
            if include_metadata and hasattr(self, 'config'):
                export_data["metadata"] = {
                    "cslics_uuid": self.config.cslics_uuid,
                    "coral_species": self.config.coral_species,
                    "tank_sheet_name": self.config.tank_sheet_name,
                    "confidence_threshold": self.config.confidence_threshold,
                    "submersion_time": getattr(self.config, 'submersion_time', None),
                    "surface_model_name": self.config_manager.get_surface_model_name() if hasattr(self, 'config_manager') else None,
                    "subsurface_model_name": self.config_manager.get_subsurface_model_name() if hasattr(self, 'config_manager') else None,
                    "surface_calibration": {
                        "idx": self.config_manager.get_surface_calibration_idx() if hasattr(self, 'config_manager') else None,
                        "window_size": self.config_manager.get_surface_calibration_window_size() if hasattr(self, 'config_manager') else None,
                        "window_shift": self.config_manager.get_surface_calibration_window_shift() if hasattr(self, 'config_manager') else None
                    },
                    "subsurface_calibration": {
                        "idx": self.config_manager.get_subsurface_calibration_idx() if hasattr(self, 'config_manager') else None,
                        "window_size": self.config_manager.get_subsurface_calibration_window_size() if hasattr(self, 'config_manager') else None,
                        "window_shift": self.config_manager.get_subsurface_calibration_window_shift() if hasattr(self, 'config_manager') else None
                    }
                }
            
            # Process manual counts data
            if manual_counts is not None:
                if isinstance(manual_counts, dict):
                    # Handle dictionary format (from self.manual_data)
                    manual_data = {
                        "counts": manual_counts.get('counts', []),
                        "std": manual_counts.get('std', []),
                        "camera_uuid": manual_counts.get('camera_uuid'),
                        "species": manual_counts.get('species')
                    }
                    
                    # Convert datetime objects to ISO format strings
                    if 'counts_time' in manual_counts:
                        times = manual_counts['counts_time']
                        if isinstance(times[0], datetime):
                            manual_data["counts_time"] = [t.isoformat() for t in times]
                        else:
                            manual_data["counts_time"] = times
                    
                    # Add decimal days if available
                    if hasattr(self, 'nearest_day') and self.nearest_day:
                        manual_data["nearest_day"] = self.nearest_day.isoformat()
                        decimal_days = []
                        for t in manual_counts['counts_time']:
                            if isinstance(t, datetime):
                                decimal_days.append((t - self.nearest_day).total_seconds() / (24 * 3600))
                            else:
                                decimal_days.append(t)  # Assume already in decimal days
                        manual_data["decimal_days"] = decimal_days
                    
                    export_data["manual_counts"] = manual_data
                else:
                    # Handle array-like format
                    export_data["manual_counts"] = {
                        "counts": list(manual_counts) if hasattr(manual_counts, '__iter__') else [manual_counts]
                    }
            
            # Process subsurface counts data
            if subsurface_counts is not None:
                subsurface_data = {}
                
                if isinstance(subsurface_counts, dict):
                    # Handle results dictionary from tank plotter
                    if 'image_counts' in subsurface_counts:
                        # Ensure it's a list
                        image_counts = subsurface_counts['image_counts']
                        if isinstance(image_counts, (list, tuple)):
                            subsurface_data["image_counts"] = list(image_counts)
                        elif hasattr(image_counts, 'tolist'):  # numpy array
                            subsurface_data["image_counts"] = image_counts.tolist()
                        else:
                            subsurface_data["image_counts"] = [image_counts]
                    if 'tank_counts_cal' in subsurface_counts:
                        # Ensure it's a list
                        tank_counts_cal = subsurface_counts['tank_counts_cal']
                        if isinstance(tank_counts_cal, (list, tuple)):
                            subsurface_data["tank_counts_cal"] = list(tank_counts_cal)
                        elif hasattr(tank_counts_cal, 'tolist'):  # numpy array
                            subsurface_data["tank_counts_cal"] = tank_counts_cal.tolist()
                        else:
                            subsurface_data["tank_counts_cal"] = [tank_counts_cal]
                    if 'tank_std_cal' in subsurface_counts:
                        # Ensure it's a list
                        tank_std_cal = subsurface_counts['tank_std_cal']
                        if isinstance(tank_std_cal, (list, tuple)):
                            subsurface_data["tank_std_cal"] = list(tank_std_cal)
                        elif hasattr(tank_std_cal, 'tolist'):  # numpy array
                            subsurface_data["tank_std_cal"] = tank_std_cal.tolist()
                        else:
                            subsurface_data["tank_std_cal"] = [tank_std_cal]
                    if 'image_times' in subsurface_counts:
                        # Ensure it's a list
                        image_times = subsurface_counts['image_times']
                        if isinstance(image_times, (list, tuple)):
                            subsurface_data["image_times"] = list(image_times)
                        elif hasattr(image_times, 'tolist'):  # numpy array
                            subsurface_data["image_times"] = image_times.tolist()
                        else:
                            subsurface_data["image_times"] = [image_times]
                    
                    if 'decimal_days' in subsurface_counts:
                        # Ensure it's a list
                        decimal_days = subsurface_counts['decimal_days']
                        if isinstance(decimal_days, (list, tuple)):
                            subsurface_data["decimal_days"] = list(decimal_days)
                        elif hasattr(decimal_days, 'tolist'):  # numpy array
                            subsurface_data["decimal_days"] = decimal_days.tolist()
                        else:
                            subsurface_data["decimal_days"] = [decimal_days]
                    
                    if 'total_counts_scaled' in subsurface_counts:
                        # Ensure it's a list
                        total_counts_scaled = subsurface_counts['total_counts_scaled']
                        if isinstance(total_counts_scaled, (list, tuple)):
                            subsurface_data["total_counts_scaled"] = list(total_counts_scaled)
                        elif hasattr(total_counts_scaled, 'tolist'):  # numpy array
                            subsurface_data["total_counts_scaled"] = total_counts_scaled.tolist()
                        else:
                            subsurface_data["total_counts_scaled"] = [total_counts_scaled]
                    
                    if 'scale_factor' in subsurface_counts:
                        subsurface_data["scale_factor"] = subsurface_counts['scale_factor']
                    
                    if 'errors' in subsurface_counts:
                        # Ensure it's a list
                        errors = subsurface_counts['errors']
                        if isinstance(errors, (list, tuple)):
                            subsurface_data["errors"] = list(errors)
                        elif hasattr(errors, 'tolist'):  # numpy array
                            subsurface_data["errors"] = errors.tolist()
                        else:
                            subsurface_data["errors"] = [errors] if errors is not None else []
                
                elif hasattr(subsurface_counts, 'to_dict'):
                    # Handle DataFrame
                    subsurface_data = subsurface_counts.to_dict('list')
                    # Convert any datetime columns to ISO strings
                    for key, values in subsurface_data.items():
                        if values and isinstance(values[0], datetime):
                            subsurface_data[key] = [v.isoformat() for v in values]
                
                export_data["subsurface_counts"] = subsurface_data
            
            # Process surface counts data
            if surface_counts is not None:
                surface_data = {}
                
                if isinstance(surface_counts, dict):
                    surface_data = surface_counts.copy()
                    # Convert datetime objects to ISO strings
                    for key, values in surface_data.items():
                        if isinstance(values, list) and values and isinstance(values[0], datetime):
                            surface_data[key] = [v.isoformat() for v in values]
                elif hasattr(surface_counts, 'to_dict'):
                    # Handle DataFrame
                    surface_data = surface_counts.to_dict('list')
                    # Convert any datetime columns to ISO strings
                    for key, values in surface_data.items():
                        if values and isinstance(values[0], datetime):
                            surface_data[key] = [v.isoformat() for v in values]
                
                export_data["surface_counts"] = surface_data
            
            # Determine output path
            if output_path is None:
                # Use same directory structure as other outputs
                data_dir = os.path.join(self.config.base_detection_dir, self.config.cslics_uuid, 'data')
                os.makedirs(data_dir, exist_ok=True)
                filename = f'counts_export_{self.config.tank_sheet_name}_{self.config.cslics_uuid}.json'
                output_path = os.path.join(data_dir, filename)
            
            # Write to JSON file
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"Counts data exported to: {output_path}")
            
            # Print summary of exported data
            print("\nExported data summary:")
            if "manual_counts" in export_data:
                manual_count = len(export_data["manual_counts"].get("counts", []))
                print(f"  Manual counts: {manual_count} data points")
            if "subsurface_counts" in export_data:
                subsurface_count = len(export_data["subsurface_counts"].get("image_counts", []))
                print(f"  Subsurface counts: {subsurface_count} data points")
            if "surface_counts" in export_data:
                # Count non-metadata columns
                surface_data = export_data["surface_counts"]
                metadata_cols = ['timestamp', 'hours_since_spawning']
                data_cols = [k for k in surface_data.keys() if k not in metadata_cols]
                surface_count = len(surface_data.get("timestamp", [])) if "timestamp" in surface_data else 0
                print(f"  Surface counts: {surface_count} data points across {len(data_cols)} classes")
            
            return output_path
            
        except Exception as e:
            print(f"Error exporting counts to JSON: {e}")
            traceback.print_exc()
            return None


# Example usage and main execution
if __name__ == "__main__":
    # Configuration file path
    
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202311_t3_aant_cslics02.json"
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202311_t3_aant_cslics06.json"
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202311_t4_amag_cslics01.json"
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202311_t4_amag_cslics08.json"
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202311_t4_amag_cslics09.json"
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202312_t3_alor_cslics02.json"
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202312_t3_alor_cslics04.json"
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202312_t3_alor_cslics06.json"
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202312_t4_alor_cslics01.json"
    config_path = "Corals/cslic/coral_spawn_counter/data_yaml_files/plotting/plot_config_20251227_aken_LAR01.json"
    # config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plotting/plot_config_202312_t4_alor_cslics09.json"
    
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
            include_subsurface=True,
            export_counts=True  # Automatically export count data to JSON
        )
        
        if results:
            print("\nAnalysis completed successfully!")
            
            # Optionally export to a custom location
            # custom_export_path = "/path/to/custom/export.json"
            # counter.export_analysis_results(results, output_path=custom_export_path)
        else:
            print("\nAnalysis failed or returned no results.")
            
    except Exception as e:
        print(f"Error running coral spawn counter: {e}")
        traceback.print_exc()




