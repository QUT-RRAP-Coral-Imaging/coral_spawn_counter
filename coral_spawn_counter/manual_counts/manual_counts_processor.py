import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import json
from datetime import timedelta
from pathlib import Path
import sys

# Add the parent directory to the path to import the config manager
sys.path.append(str(Path(__file__).parent.parent))
from config.plot_config_manager import PlotConfigManager, CoralSpawnConfig


class ManualCountsProcessor:
    """Processor for manual counts data using configuration management."""
    
    def __init__(self, config_manager: PlotConfigManager):
        """
        Initialize with a configuration manager.
        
        Args:
            config_manager: PlotConfigManager instance with loaded configuration
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # Set up convenience attributes for backward compatibility
        self.manual_counts_file = self.config.manual_counts_file
        self.tank_sheet_name = self.config.tank_sheet_name
        self.spawning_sheet_name = self.config.spawning_sheet_name
        self.cslics_associations_file = self.config.cslics_associations_file
        self.save_manual_plot_dir = self.config.save_manual_plot_dir
        
    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Create processor from configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            ManualCountsProcessor instance
        """
        config_manager = PlotConfigManager(config_path)
        return cls(config_manager)
    
    def read_manual_counts(self):
        """
        Reads manual count data from an Excel file and processes it.

        This function reads data from an Excel file specified by `self.manual_counts_file` 
        and extracts manual count information from the sheet specified by `self.tank_sheet_name`. 
        It calculates the corresponding decimal days and timestamps for the counts.

        Returns:
            tuple: A tuple containing:
                - pd.Series: The manual counts from the 'Count (500L)' column.
                - pd.Series: The standard deviations from the 'Std Dev' column.
                - pd.Series: The decimal days calculated relative to the nearest day.
                - pd.Series: The datetime objects representing the counts' timestamps.
        """
        df = pd.read_excel(self.manual_counts_file, sheet_name=self.tank_sheet_name, engine='openpyxl', header=5)
        counts_time = pd.to_datetime(df['Date'].astype(str) + " " + df['Time'].astype(str), dayfirst=True)
        nearest_day = counts_time[0].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        decimal_days = self.convert_to_decimal_days(counts_time, nearest_day)
        return df['Count (500L)'], df['Std Dev'], decimal_days, counts_time
    
    def convert_to_decimal_days(self, dates_list, time_zero=None):
        """
        Convert datetime objects to decimal days relative to a reference time.
        
        Args:
            dates_list: List of datetime objects
            time_zero: Reference time (if None, uses first date in list)
            
        Returns:
            list: Decimal days relative to time_zero
        """
        if time_zero is None:
            time_zero = dates_list[0]
        return [(date - time_zero).total_seconds() / (60 * 60 * 24) for date in dates_list]

    def read_cslics_uuid_tank_association(self):
        """
        Read the manual counts from the Excel file and return the camera UUID and species.

        Uses the class attributes for the associations file, spawning sheet name, and tank sheet name.
        
        Returns:
            tuple: A tuple containing:
                - str: The camera UUID.
                - str: The coral species.
        """
        try:
            # Read the Excel file using the class attributes
            df = pd.read_excel(self.cslics_associations_file, sheet_name=self.spawning_sheet_name, engine='openpyxl', header=2)
            
            # Find the index of the tank sheet name
            idx = df['manual count sheet name'].tolist().index(self.tank_sheet_name)
            print(f'Index of {self.tank_sheet_name}: {idx}')
            
            # Return the camera UUID and species
            return df.at[idx, 'camera uuid'], df.at[idx, 'species']
        except ValueError:
            print(f'Error: {self.tank_sheet_name} not found in the manual count sheet.')
        except Exception as e:
            print(f'Error reading {self.cslics_associations_file}: {e}')
        
        # Return None values if an error occurs
        return None, None
    
    def plot_manual_counts(self, counts, std, days, SHOW=False):
        """
        Plot manual counts with error bars.
        
        Args:
            counts: Manual count data
            std: Standard deviation data
            days: Decimal days data
            SHOW: Whether to display the plot
        """
        scaled_counts = counts/1000  # for readability
        scaled_std = std/1000
        n = 1.0

        __, ax = plt.subplots()
        ax.plot(days, scaled_counts, marker='o', color='blue')
        ax.errorbar(days, scaled_counts, yerr=n*scaled_std, fmt='o', color='blue', alpha=0.5)
        plt.grid(True)
        plt.xlabel('Days since spawning')
        plt.ylabel('Tank count (in thousands for 500L)')
        plt.title(f'CSLICS Manual Count: {self.tank_sheet_name}')
        
        # Ensure save directory exists
        os.makedirs(self.save_manual_plot_dir, exist_ok=True)
        
        # Save plot
        plot_filename = f'Manual_counts_{self.tank_sheet_name}.png'
        plot_path = os.path.join(self.save_manual_plot_dir, plot_filename)
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        
        if SHOW:
            plt.show()
        else:
            plt.close()  # Close the figure to free memory
    
    def get_detection_directory(self) -> str:
        """Get the detection directory for current configuration."""
        return self.config_manager.get_detection_dir()
    
    def get_output_plot_path(self, filename: str) -> str:
        """
        Get full path for output plot file.
        
        Args:
            filename: Name of the plot file
            
        Returns:
            Full path to save the plot
        """
        return str(Path(self.save_manual_plot_dir) / filename)
    
    def process_all_manual_counts(self, show_plot=False):
        """
        Process all manual counts data and generate plot.
        
        Args:
            show_plot: Whether to display the plot
            
        Returns:
            dict: Dictionary containing processed data
        """
        try:
            # Read manual counts data
            counts, std, decimal_days, counts_time = self.read_manual_counts()
            
            # Read CSLICS UUID and species association
            camera_uuid, species = self.read_cslics_uuid_tank_association()
            
            # Validate that the configuration matches the data
            if camera_uuid and camera_uuid != self.config.cslics_uuid:
                print(f"Warning: Config UUID ({self.config.cslics_uuid}) doesn't match data UUID ({camera_uuid})")
            
            if species and species != self.config.coral_species:
                print(f"Warning: Config species ({self.config.coral_species}) doesn't match data species ({species})")
            
            # Generate plot
            self.plot_manual_counts(counts, std, decimal_days, SHOW=show_plot)
            
            return {
                'counts': counts,
                'std': std,
                'decimal_days': decimal_days,
                'counts_time': counts_time,
                'camera_uuid': camera_uuid,
                'species': species
            }
            
        except Exception as e:
            print(f"Error processing manual counts: {e}")
            return None
    
    def get_config_summary(self):
        """Get a summary of the current configuration."""
        return {
            'cslics_uuid': self.config.cslics_uuid,
            'coral_species': self.config.coral_species,
            'tank_sheet_name': self.config.tank_sheet_name,
            'spawning_sheet_name': self.config.spawning_sheet_name,
            'manual_counts_file': self.config.manual_counts_file,
            'detection_directory': self.get_detection_directory()
        }


# Example usage
if __name__ == "__main__":
    # Load configuration and create processor
    config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plot_config_202312_t4_alor_cslics08.json"
    
    try:
        processor = ManualCountsProcessor.from_config_file(config_path)
        
        # Print configuration summary
        print("Configuration Summary:")
        config_summary = processor.get_config_summary()
        for key, value in config_summary.items():
            print(f"  {key}: {value}")
        
        # Process manual counts
        print("\nProcessing manual counts...")
        results = processor.process_all_manual_counts(show_plot=False)
        
        if results:
            print(f"Successfully processed {len(results['counts'])} manual count records")
            print(f"Species: {results['species']}")
            print(f"Camera UUID: {results['camera_uuid']}")
        else:
            print("Failed to process manual counts")
            
    except Exception as e:
        print(f"Error: {e}")