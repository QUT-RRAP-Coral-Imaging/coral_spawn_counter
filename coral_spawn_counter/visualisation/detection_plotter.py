# visualization/plotter.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from pathlib import Path


class DetectionPlotter:
    """Handles visualization and plotting of detection data."""
    
    def __init__(self, config, detection_data_manager, time_utils):
        """
        Initialize the detection plotter.
        
        Args:
            config: ConfigManager instance
            detection_data_manager: DetectionDataManager instance
            time_utils: TimeUtils instance
        """
        self.config = config
        self.detection_data_manager = detection_data_manager
        self.time_utils = time_utils
        
        # Configuration parameters
        self.cslics_uuid = config.cslics_uuid
        self.mode = config.mode
        self.submersion_time = config.submersion_time
        self.submersion_datetime = config.submersion_datetime
        self.surface_model_id = config.surface_model_id
        self.subsurface_model_id = config.subsurface_model_id
        
        # Get output directory for plots
        self.plot_dir = os.path.join(
            config.save_dir, config.cslics_uuid, "plots"
        )
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # Plot styling
        self._setup_plot_style()
    
    def _setup_plot_style(self):
        """Set up matplotlib plotting style."""
        plt.style.use('default')
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['lines.linewidth'] = 2
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
    
    def plot_surface_detections(self, save_path=None):
        """
        Plot surface detections over time.
        
        Args:
            save_path: Optional custom save path for the plot
            
        Returns:
            str: Path to the saved plot file
        """
        if self.mode not in ["surface", "both"]:
            print("Surface plotting not enabled for current mode")
            return None
        
        # Load surface detection data
        surface_data = self.detection_data_manager.load_detection_data("surface")
        if not surface_data or not surface_data.get("detections_by_timestamp"):
            print("No surface detection data available for plotting")
            return None
        
        # Prepare data for plotting
        timestamps, counts = self._prepare_time_series_data(
            surface_data["detections_by_timestamp"]
        )
        
        if not timestamps:
            print("No valid timestamps found in surface data")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot detection counts over time
        ax.plot(timestamps, counts, 'b-o', markersize=4, linewidth=2, 
                label=f'Surface Detections (Total: {surface_data["total_detections"]})')
        
        # Add submersion time line
        ax.axvline(x=self.submersion_datetime, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Submersion Time: {self.submersion_time}')
        
        # Customize the plot
        self._customize_time_plot(ax, "Surface Coral Spawn Detections", 
                                 surface_data["total_detections"])
        
        # Add statistics text box
        self._add_statistics_box(ax, surface_data, "surface")
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(self.plot_dir, f"{self.cslics_uuid}_surface_detections.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Surface detection plot saved to: {save_path}")
        return save_path
    
    def plot_subsurface_detections(self, save_path=None):
        """
        Plot subsurface detections over time.
        
        Args:
            save_path: Optional custom save path for the plot
            
        Returns:
            str: Path to the saved plot file
        """
        if self.mode not in ["subsurface", "both"]:
            print("Subsurface plotting not enabled for current mode")
            return None
        
        # Load subsurface detection data
        subsurface_data = self.detection_data_manager.load_detection_data("subsurface")
        if not subsurface_data or not subsurface_data.get("detections_by_timestamp"):
            print("No subsurface detection data available for plotting")
            return None
        
        # Prepare data for plotting
        timestamps, counts = self._prepare_time_series_data(
            subsurface_data["detections_by_timestamp"]
        )
        
        if not timestamps:
            print("No valid timestamps found in subsurface data")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot detection counts over time
        ax.plot(timestamps, counts, 'g-o', markersize=4, linewidth=2, 
                label=f'Subsurface Detections (Total: {subsurface_data["total_detections"]})')
        
        # Add submersion time line
        ax.axvline(x=self.submersion_datetime, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Submersion Time: {self.submersion_time}')
        
        # Customize the plot
        self._customize_time_plot(ax, "Subsurface Coral Spawn Detections", 
                                 subsurface_data["total_detections"])
        
        # Add statistics text box
        self._add_statistics_box(ax, subsurface_data, "subsurface")
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(self.plot_dir, f"{self.cslics_uuid}_subsurface_detections.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Subsurface detection plot saved to: {save_path}")
        return save_path
    
    def plot_all_detections(self, save_path=None):
        """
        Plot both surface and subsurface detections on the same plot.
        
        Args:
            save_path: Optional custom save path for the plot
            
        Returns:
            str: Path to the saved plot file
        """
        if self.mode != "both":
            print("Combined plotting only available when mode is 'both'")
            return None
        
        # Load both datasets
        surface_data = self.detection_data_manager.load_detection_data("surface")
        subsurface_data = self.detection_data_manager.load_detection_data("subsurface")
        
        if not surface_data and not subsurface_data:
            print("No detection data available for plotting")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(16, 10))
        
        total_detections = 0
        
        # Plot surface data if available
        if surface_data and surface_data.get("detections_by_timestamp"):
            timestamps, counts = self._prepare_time_series_data(
                surface_data["detections_by_timestamp"]
            )
            if timestamps:
                ax.plot(timestamps, counts, 'b-o', markersize=4, linewidth=2, 
                        label=f'Surface Detections (Total: {surface_data["total_detections"]})')
                total_detections += surface_data["total_detections"]
        
        # Plot subsurface data if available
        if subsurface_data and subsurface_data.get("detections_by_timestamp"):
            timestamps, counts = self._prepare_time_series_data(
                subsurface_data["detections_by_timestamp"]
            )
            if timestamps:
                ax.plot(timestamps, counts, 'g-o', markersize=4, linewidth=2, 
                        label=f'Subsurface Detections (Total: {subsurface_data["total_detections"]})')
                total_detections += subsurface_data["total_detections"]
        
        # Add submersion time line
        ax.axvline(x=self.submersion_datetime, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Submersion Time: {self.submersion_time}')
        
        # Customize the plot
        self._customize_time_plot(ax, "All Coral Spawn Detections", total_detections)
        
        # Add combined statistics box
        self._add_combined_statistics_box(ax, surface_data, subsurface_data)
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(self.plot_dir, f"{self.cslics_uuid}_all_detections.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Combined detection plot saved to: {save_path}")
        return save_path
    
    def plot_detection_rate_analysis(self, save_path=None):
        """
        Plot detection rate analysis (detections per hour, etc.).
        
        Args:
            save_path: Optional custom save path for the plot
            
        Returns:
            str: Path to the saved plot file
        """
        # Load data based on mode
        datasets = {}
        if self.mode in ["surface", "both"]:
            surface_data = self.detection_data_manager.load_detection_data("surface")
            if surface_data:
                datasets["surface"] = surface_data
        
        if self.mode in ["subsurface", "both"]:
            subsurface_data = self.detection_data_manager.load_detection_data("subsurface")
            if subsurface_data:
                datasets["subsurface"] = subsurface_data
        
        if not datasets:
            print("No data available for rate analysis")
            return None
        
        # Create subplots
        n_plots = len(datasets)
        fig, axes = plt.subplots(n_plots, 1, figsize=(14, 6*n_plots))
        if n_plots == 1:
            axes = [axes]
        
        for i, (detection_type, data) in enumerate(datasets.items()):
            ax = axes[i]
            
            # Calculate hourly detection rates
            hourly_rates = self._calculate_hourly_rates(data["detections_by_timestamp"])
            
            if hourly_rates:
                hours, rates = zip(*hourly_rates)
                color = 'blue' if detection_type == 'surface' else 'green'
                
                ax.bar(hours, rates, alpha=0.7, color=color, 
                       label=f'{detection_type.capitalize()} Detection Rate')
                
                ax.set_xlabel('Hours from Start')
                ax.set_ylabel('Detections per Hour')
                ax.set_title(f'{detection_type.capitalize()} Detection Rate Analysis')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Add submersion time line
                submersion_hours = (self.submersion_datetime - min(
                    datetime.fromisoformat(ts) for ts in data["detections_by_timestamp"].keys()
                )).total_seconds() / 3600
                ax.axvline(x=submersion_hours, color='red', linestyle='--', 
                          alpha=0.7, label='Submersion Time')
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(self.plot_dir, f"{self.cslics_uuid}_detection_rate_analysis.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Detection rate analysis plot saved to: {save_path}")
        return save_path
    
    def plot_cumulative_detections(self, save_path=None):
        """
        Plot cumulative detections over time.
        
        Args:
            save_path: Optional custom save path for the plot
            
        Returns:
            str: Path to the saved plot file
        """
        # Load data based on mode
        datasets = {}
        if self.mode in ["surface", "both"]:
            surface_data = self.detection_data_manager.load_detection_data("surface")
            if surface_data:
                datasets["surface"] = surface_data
        
        if self.mode in ["subsurface", "both"]:
            subsurface_data = self.detection_data_manager.load_detection_data("subsurface")
            if subsurface_data:
                datasets["subsurface"] = subsurface_data
        
        if not datasets:
            print("No data available for cumulative plot")
            return None
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for detection_type, data in datasets.items():
            # Calculate cumulative detections
            timestamps, cumulative_counts = self._calculate_cumulative_detections(
                data["detections_by_timestamp"]
            )
            
            if timestamps:
                color = 'blue' if detection_type == 'surface' else 'green'
                ax.plot(timestamps, cumulative_counts, color=color, linewidth=3,
                        label=f'{detection_type.capitalize()} (Total: {data["total_detections"]})')
        
        # Add submersion time line
        ax.axvline(x=self.submersion_datetime, color='red', linestyle='--', 
                   linewidth=2, alpha=0.7, label=f'Submersion Time: {self.submersion_time}')
        
        # Customize the plot
        ax.set_xlabel('Time')
        ax.set_ylabel('Cumulative Detection Count')
        ax.set_title(f'Cumulative Coral Spawn Detections - {self.cslics_uuid}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
        
        # Save the plot
        if save_path is None:
            save_path = os.path.join(self.plot_dir, f"{self.cslics_uuid}_cumulative_detections.png")
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Cumulative detections plot saved to: {save_path}")
        return save_path
    
    def _prepare_time_series_data(self, detections_by_timestamp):
        """
        Prepare timestamp and count data for plotting.
        
        Args:
            detections_by_timestamp: Dictionary of timestamp -> detection data
            
        Returns:
            tuple: (timestamps, counts) lists
        """
        timestamps = []
        counts = []
        
        for timestamp_str, data in sorted(detections_by_timestamp.items()):
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                timestamps.append(timestamp)
                counts.append(data["count"])
            except ValueError:
                continue  # Skip invalid timestamps
        
        return timestamps, counts
    
    def _calculate_hourly_rates(self, detections_by_timestamp):
        """
        Calculate hourly detection rates.
        
        Args:
            detections_by_timestamp: Dictionary of timestamp -> detection data
            
        Returns:
            list: List of (hour, rate) tuples
        """
        if not detections_by_timestamp:
            return []
        
        # Group detections by hour
        hourly_counts = {}
        start_time = None
        
        for timestamp_str, data in detections_by_timestamp.items():
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if start_time is None:
                    start_time = timestamp
                
                hour = int((timestamp - start_time).total_seconds() // 3600)
                hourly_counts[hour] = hourly_counts.get(hour, 0) + data["count"]
                
            except ValueError:
                continue
        
        # Convert to sorted list
        return sorted(hourly_counts.items())
    
    def _calculate_cumulative_detections(self, detections_by_timestamp):
        """
        Calculate cumulative detections over time.
        
        Args:
            detections_by_timestamp: Dictionary of timestamp -> detection data
            
        Returns:
            tuple: (timestamps, cumulative_counts) lists
        """
        timestamps, counts = self._prepare_time_series_data(detections_by_timestamp)
        
        if not timestamps:
            return [], []
        
        # Calculate cumulative counts
        cumulative_counts = []
        running_total = 0
        
        for count in counts:
            running_total += count
            cumulative_counts.append(running_total)
        
        return timestamps, cumulative_counts
    
    def _customize_time_plot(self, ax, title, total_detections):
        """Customize a time-series plot with standard formatting."""
        ax.set_xlabel('Time')
        ax.set_ylabel('Detection Count')
        ax.set_title(f'{title} - {self.cslics_uuid}\nTotal Detections: {total_detections}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis to show dates and times nicely
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
        plt.xticks(rotation=45)
    
    def _add_statistics_box(self, ax, data, detection_type):
        """Add a statistics text box to the plot."""
        stats_text = self._generate_statistics_text(data, detection_type)
        
        # Position the text box
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=9)
    
    def _add_combined_statistics_box(self, ax, surface_data, subsurface_data):
        """Add a combined statistics text box for both surface and subsurface data."""
        stats_lines = [f"CSLICS UUID: {self.cslics_uuid}"]
        
        if surface_data:
            stats_lines.extend([
                f"Surface Model: {surface_data.get('model_id', 'Unknown')}",
                f"Surface Detections: {surface_data.get('total_detections', 0)}",
                f"Surface Files: {surface_data.get('total_files_processed', 0)}"
            ])
        
        if subsurface_data:
            stats_lines.extend([
                f"Subsurface Model: {subsurface_data.get('model_id', 'Unknown')}",
                f"Subsurface Detections: {subsurface_data.get('total_detections', 0)}",
                f"Subsurface Files: {subsurface_data.get('total_files_processed', 0)}"
            ])
        
        total_detections = 0
        if surface_data:
            total_detections += surface_data.get('total_detections', 0)
        if subsurface_data:
            total_detections += subsurface_data.get('total_detections', 0)
        
        stats_lines.append(f"Total Detections: {total_detections}")
        stats_text = '\n'.join(stats_lines)
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='wheat', alpha=0.8), fontsize=9)
    
    def _generate_statistics_text(self, data, detection_type):
        """Generate statistics text for a data set."""
        stats_lines = [
            f"CSLICS UUID: {self.cslics_uuid}",
            f"Model: {data.get('model_id', 'Unknown')}",
            f"Type: {detection_type.capitalize()}",
            f"Total Detections: {data.get('total_detections', 0)}",
            f"Files Processed: {data.get('total_files_processed', 0)}",
            f"Unique Timestamps: {len(data.get('detections_by_timestamp', {}))}",
            f"Last Updated: {data.get('last_updated', 'Unknown')}"
        ]
        
        return '\n'.join(stats_lines)
    
    def generate_all_plots(self):
        """
        Generate all available plots based on the current mode.
        
        Returns:
            dict: Dictionary of plot types to file paths
        """
        generated_plots = {}
        
        if self.mode in ["surface", "both"]:
            surface_plot = self.plot_surface_detections()
            if surface_plot:
                generated_plots["surface"] = surface_plot
        
        if self.mode in ["subsurface", "both"]:
            subsurface_plot = self.plot_subsurface_detections()
            if subsurface_plot:
                generated_plots["subsurface"] = subsurface_plot
        
        if self.mode == "both":
            combined_plot = self.plot_all_detections()
            if combined_plot:
                generated_plots["combined"] = combined_plot
        
        # Generate additional analysis plots
        rate_analysis = self.plot_detection_rate_analysis()
        if rate_analysis:
            generated_plots["rate_analysis"] = rate_analysis
        
        cumulative_plot = self.plot_cumulative_detections()
        if cumulative_plot:
            generated_plots["cumulative"] = cumulative_plot
        
        return generated_plots
    
    def export_plot_data(self, output_dir=None):
        """
        Export the underlying data used for plotting to CSV files.
        
        Args:
            output_dir: Directory to save CSV files (default: plot directory)
        """
        if output_dir is None:
            output_dir = self.plot_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        for detection_type in ["surface", "subsurface"]:
            if self.mode in [detection_type, "both"]:
                data = self.detection_data_manager.load_detection_data(detection_type)
                if data and data.get("detections_by_timestamp"):
                    csv_path = os.path.join(output_dir, 
                                          f"{self.cslics_uuid}_{detection_type}_plot_data.csv")
                    self._export_plot_data_csv(data["detections_by_timestamp"], csv_path)
                    print(f"Exported {detection_type} plot data to: {csv_path}")
    
    def _export_plot_data_csv(self, detections_by_timestamp, csv_path):
        """Export detection data to CSV format for external analysis."""
        import csv
        
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'detection_count', 'files_count', 
                           'datetime_parsed', 'hours_from_start'])
            
            timestamps, counts = self._prepare_time_series_data(detections_by_timestamp)
            start_time = timestamps[0] if timestamps else None
            
            for timestamp_str, data in sorted(detections_by_timestamp.items()):
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    hours_from_start = ((timestamp - start_time).total_seconds() / 3600) if start_time else 0
                    
                    writer.writerow([
                        timestamp_str,
                        data["count"],
                        data["files"],
                        timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                        f"{hours_from_start:.2f}"
                    ])
                except ValueError:
                    continue
    
    def print_plotting_summary(self):
        """Print a summary of the plotting capabilities."""
        print("\nDetection Plotter Summary:")
        print(f"CSLICS UUID: {self.cslics_uuid}")
        print(f"Processing mode: {self.mode}")
        print(f"Plot output directory: {self.plot_dir}")
        print(f"Submersion time: {self.submersion_time}")
        
        # Check data availability
        data_summary = self.detection_data_manager.get_detection_summary()
        
        if data_summary:
            print("\nAvailable data for plotting:")
            for detection_type, data in data_summary.items():
                print(f"  {detection_type.capitalize()}: {data['total_detections']} detections, "
                      f"{data['unique_timestamps']} timestamps")
        else:
            print("\nNo detection data available for plotting")
        
        print("\nAvailable plot types:")
        if self.mode in ["surface", "both"]:
            print("  - Surface detections over time")
        if self.mode in ["subsurface", "both"]:
            print("  - Subsurface detections over time")
        if self.mode == "both":
            print("  - Combined surface and subsurface detections")
        print("  - Detection rate analysis")
        print("  - Cumulative detections over time")