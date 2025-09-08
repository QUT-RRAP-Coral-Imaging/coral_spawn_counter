# the purpose of this file is to plot the tank counts/estimates, based on the detection data
# and compare to the manual counts

import numpy as np
from matplotlib import pyplot as plt
import os
import json
import tqdm
from datetime import datetime
from pathlib import Path
import sys

# Add the parent directory to the path to import the config manager
sys.path.append(str(Path(__file__).parent.parent))
from config.plot_config_manager import PlotConfigManager, CoralSpawnConfig


class TankCountPlotter:
    """Class for plotting tank counts/estimates based on detection data and comparing to manual counts."""
    
    def __init__(self, config_manager: PlotConfigManager):
        """
        Initialize with a configuration manager.
        
        Args:
            config_manager: PlotConfigManager instance with loaded configuration
        """
        self.config_manager = config_manager
        self.config = config_manager.get_config()
        
        # Set up convenience attributes from config
        self.save_det_dir = self.config_manager.get_detection_dir()
        self.tank_sheet_name = self.config.tank_sheet_name
        self.cslics_uuid = self.config.cslics_uuid
        self.coral_species = self.config.coral_species
        self.skipping_frequency = self.config.skipping_frequency
        self.aggregate_size = self.config.aggregate_size
        self.confidence_threshold = self.config.confidence_threshold
        self.MAX_SAMPLE = self.config.MAX_SAMPLE
        self.calibration_window_size = self.config.calibration_window_size
        self.calibration_idx = self.config.calibration_idx
        self.calibration_window_shift = self.config.calibration_window_shift
        self.PLOT_FOCUS_VOLUME = self.config.PLOT_FOCUS_VOLUME
        self.SHOW_INVALID_POINTS = self.config.SHOW_INVALID_POINTS
        
        # Add SHOW as a class property (default to False for batch processing)
        self.SHOW = False

    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Create TankCountPlotter from configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            TankCountPlotter instance
        """
        config_manager = PlotConfigManager(config_path)
        return cls(config_manager)

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

    def read_detections(self, det_dir):
        """
        Read in all the json detection files.
        Files are sorted chronologically based on filename structure.
        
        Returns:
            list: Sorted list of detection file paths
        """
        print(f'Gathering detection files from: {det_dir}')
        sample_list = sorted(Path(det_dir).rglob('*_det.json'))
        print(f'Found {len(sample_list)} detection files.')

        if len(sample_list) < self.skipping_frequency:
            raise ValueError(f"Not enough detection files ({len(sample_list)}) for skipping frequency ({self.skipping_frequency}).")
        if len(sample_list) < self.aggregate_size:
            raise ValueError(f"Not enough detection files ({len(sample_list)}) for aggregate size ({self.aggregate_size}).")
        return sample_list
    
    def batch_detections(self, sample_list, nearest_day, invalid_indices=None):
        """
        Batch detections and handle invalid indices.

        Args:
            sample_list: List of detection files.
            nearest_day: Reference day for calculating decimal days.
            invalid_indices: List of indices corresponding to invalid detection files.

        Returns:
            Tuple containing:
                - batched_image_count: Array of batched detection counts.
                - batched_std: Array of standard deviations for batched counts.
                - decimal_capture_times: Array of decimal days for batched times.
                - batched_invalid_indices: List of invalid indices mapped to batches.
        """
        # Skip every X images
        downsampled_list = sample_list[::self.skipping_frequency]
        batched_samples = [downsampled_list[i:i + self.aggregate_size] for i in range(0, len(downsampled_list), self.aggregate_size)]

        batched_image_count, batched_std, batched_time, batched_invalid_indices = [], [], [], []

        # Iterate over all the batched samples with tqdm progress bar
        print(f'Batching {len(batched_samples)} samples...')
        for batch_idx, sample_batch in tqdm.tqdm(enumerate(batched_samples[:self.MAX_SAMPLE]), 
                                        total=min(len(batched_samples), self.MAX_SAMPLE),
                                        desc="Processing batches"):
            sample_count = []
            batch_invalid_indices = []

            # Process files within the current batch
            for i, detection_file in enumerate(sample_batch):
                try:
                    with open(detection_file, 'r') as f:
                        data = json.load(f)
                    detections = data['detections']
                    
                    # Fix: Access confidence using dictionary key, not list index
                    sample_count.append(sum(1 for d in detections if d['confidence'] >= self.confidence_threshold))

                    # Check if this file is invalid
                    if invalid_indices and sample_list.index(detection_file) in invalid_indices:
                        batch_invalid_indices.append(i)

                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f'Error reading {detection_file}: {e}')
                    # Add a count of 0 for failed files to maintain batch consistency
                    sample_count.append(0)

            # Get plot of batch
            self.plot_batch_histogram(sample_count, batch_idx)
            
            # Average stats over the batch
            batched_image_count.append(np.mean(sample_count))
            batched_std.append(np.std(sample_count))
            
            # Extract capture time from filename
            # for 2024 capture string
            # capture_time_str = Path(sample_batch[len(sample_batch) // 2]).stem[:-10]
            # for 2023 capture string
            capture_time_str = Path(sample_batch[len(sample_batch) // 2]).stem[9:-15]
            
            # Parse time string
            # 2024 capture time string format
            # batched_time.append(datetime.strptime(capture_time_str, "%Y-%m-%d_%H-%M-%S"))
            # 2023 capture time string format
            batched_time.append(datetime.strptime(capture_time_str, "%Y%m%d_%H%M%S"))
            batched_invalid_indices.append(batch_invalid_indices)

        # Convert batched_time to decimal days and zero the time since spawning
        decimal_capture_times = self.convert_to_decimal_days(batched_time, nearest_day)
        return np.array(batched_image_count), np.array(batched_std), np.array(decimal_capture_times), batched_invalid_indices

    def plot_batch_histogram(self, batch_counts, batch_idx, x_range=(0,60), y_max=50):
        """
        Plot a histogram of the counts in a single batch with consistent x and y axes.

        Args:
            batch_counts (list): List of detection counts in the batch.
            batch_idx (int): Index of the batch (used for labeling the plot).
            x_range (tuple): Tuple specifying the global x-axis range (min, max).
            y_max (int): Maximum value for the y-axis (frequency).
        """
        __, ax = plt.subplots()
        ax.hist(batch_counts, bins=10, range=x_range, color='blue', alpha=0.9, edgecolor='black')
        ax.set_xlim(x_range)  # Set consistent x-axis range
        ax.set_ylim(0, y_max)  # Set consistent y-axis range
        plt.xlabel('Detection Counts')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Batch Counts (Batch {batch_idx + 1})')
        plt.grid(True)

        # Ensure output directory exists
        os.makedirs(self.save_det_dir, exist_ok=True)
        
        # Save the plot
        output_path = os.path.join(self.save_det_dir, f'Batch_{batch_idx + 1}_Histogram.png')
        plt.savefig(output_path, dpi=600)
        print(f"Histogram for Batch {batch_idx + 1} saved to {output_path}")

        if self.SHOW:
            plt.show()
        plt.close()

    def plot_image_detections(self, counts, std, times):
        """
        Plot image-based detections with error bands.

        Args:
            counts (array-like): Array of detection counts.
            std (array-like): Array of standard deviations for the counts.
            times (array-like): Array of times (in decimal days) since spawning.
        """
        n = 1  # Multiplier for the error band
        __, ax = plt.subplots()
        ax.plot(times, counts, label='Detections')
        ax.fill_between(times, counts - n * std, counts + n * std, alpha=0.2, label='Error Band')
        plt.grid(True)
        plt.xlabel('Days since spawning')
        plt.ylabel(f'Image count (batched {self.aggregate_size} images)')
        plt.title(f'CSLICS AI Count: {self.tank_sheet_name}, {self.cslics_uuid}')
        plt.legend()
        
        # Ensure output directory exists
        os.makedirs(self.save_det_dir, exist_ok=True)
        
        output_path = os.path.join(self.save_det_dir, f'Image_counts_{self.tank_sheet_name}.png')
        plt.savefig(output_path)
        print(f"Image detections plot saved to {output_path}")
        if self.SHOW:
            plt.show()
        plt.close()
            
    def find_closest_time(self, image_time, manual_time, manual_idx=None):
        """
        Find the closest time match between image and manual times.
        
        Args:
            image_time: Array of image datetime objects
            manual_time: Array of manual datetime objects
            manual_idx: Index for manual time reference (if None, uses calibration_idx)
            
        Returns:
            tuple: (closest_index, minimum_time_difference)
        """
        if manual_idx is None:
            manual_idx = self.calibration_idx
        t_diff = abs(image_time - manual_time[manual_idx])
        return np.argmin(t_diff), np.min(t_diff)

    def scale_by_focus_volume(self):
        """
        Scale by focus volume (placeholder method - implement based on your focus volume calculation).
        
        Returns:
            float: Scale factor based on focus volume
        """
        # TODO: Implement focus volume scaling logic
        # This is a placeholder - you'll need to implement the actual focus volume calculation
        return 1.0

    def scale_by_manual_calibration_idx(self, manual_count, image_counts, closest_idx):
        """
        Determine scale factor for image_counts based on manual_counts and calibration_idx.
        
        Args:
            manual_count: Manual count value for calibration
            image_counts: Array of image counts
            closest_idx: Index of closest time match
            
        Returns:
            tuple: (scale_factor, selected_index)
        """
        # Added due to some potential calibration times lining up with "night" conditions
        idx_select = closest_idx + self.calibration_window_shift
        
        # Find the idx for the nearest time to the specified calibration manual time
        # accounting for min/max sizes of image_counts
        idx_min = int(idx_select - self.calibration_window_size/2)
        if idx_min < 0:
            idx_min = 0
            if len(image_counts) <= self.calibration_window_size:
                idx_max = len(image_counts) - 1
            else:
                idx_max = self.calibration_window_size
        else:
            idx_max = int(idx_min + self.calibration_window_size)
        if idx_max >= len(image_counts):
            idx_max = len(image_counts) - 1
            
        image_count_window = image_counts[idx_min:idx_max]
        # Compute the scale factor over specified average (based on aggregate size?)
        image_sample_average = np.mean(image_count_window)
        
        # Handle the divide by zero case 
        if image_sample_average == 0:
            scale_factor = 1
        else:
            scale_factor = manual_count / image_sample_average
        print(f'calibration scale factor = {scale_factor}')
        return scale_factor, idx_select

    def process_invalid_points(self, image_times, tank_counts, tank_std, batched_invalid_indices):
        """
        Process invalid points by either removing them or interpolating values.
        
        Args:
            image_times: Array of image times
            tank_counts: Array of tank counts
            tank_std: Array of tank standard deviations
            batched_invalid_indices: List of invalid indices per batch
            
        Returns:
            tuple: (processed_times, processed_counts, processed_std, interpolated_mask)
        """
        # Create mask for invalid batches (batches with any invalid points)
        invalid_batch_mask = np.array([len(batch_invalid) > 0 for batch_invalid in batched_invalid_indices])
        
        # Create arrays for valid points only
        valid_mask = ~invalid_batch_mask
        valid_times = image_times[valid_mask]
        valid_counts = tank_counts[valid_mask]
        valid_std = tank_std[valid_mask]
        
        # If we have enough valid points, interpolate for invalid ones
        if len(valid_times) >= 2:
            # Interpolate counts and std for invalid points
            interpolated_counts = np.interp(image_times[invalid_batch_mask], valid_times, valid_counts)
            interpolated_std = np.interp(image_times[invalid_batch_mask], valid_times, valid_std)
            
            # Combine valid and interpolated data
            processed_times = image_times.copy()
            processed_counts = tank_counts.copy()
            processed_std = tank_std.copy()
            
            processed_counts[invalid_batch_mask] = interpolated_counts
            processed_std[invalid_batch_mask] = interpolated_std
            
            return processed_times, processed_counts, processed_std, invalid_batch_mask
        else:
            # Not enough valid points for interpolation, return only valid points
            return valid_times, valid_counts, valid_std, np.zeros(len(valid_times), dtype=bool)

    def process_and_scale_counts(self, image_counts, image_std, image_times, manual_counts, manual_std, manual_times):
        """
        Process and scale image counts using focus volume and manual calibration.

        Args:
            image_counts: Array of image counts.
            image_std: Array of image standard deviations.
            image_times: Array of image times (in decimal days).
            manual_counts: Array of manual counts.
            manual_std: Array of manual standard deviations.
            manual_times: Array of manual times (in decimal days).
            
        Returns:
            tuple: ((tank_counts_def, tank_std_def), (tank_counts_cal, tank_std_cal), scaling_idx)
        """
        # Scale factor by focus volume
        scale_factor_focus = self.scale_by_focus_volume()

        # Apply scale factor
        tank_counts_def = image_counts * scale_factor_focus
        tank_std_def = image_std * scale_factor_focus

        # Find the closest time for manual calibration
        closest_idx, __ = self.find_closest_time(image_times, manual_times)

        # Scale factor by manual calibration
        scale_factor_manual, scaling_idx = self.scale_by_manual_calibration_idx(
            manual_counts[self.calibration_idx], image_counts, closest_idx
        )

        # Apply scale factor to image counts
        tank_counts_cal = image_counts * scale_factor_manual
        tank_std_cal = image_std * scale_factor_manual

        return (tank_counts_def, tank_std_def), (tank_counts_cal, tank_std_cal), scaling_idx

    def plot_detections_and_manual_counts(
        self, 
        image_times, 
        tank_counts_def, 
        tank_std_def, 
        tank_counts_cal, 
        tank_std_cal, 
        manual_counts, 
        manual_std, 
        manual_times, 
        scaling_idx, 
        batched_invalid_indices,
        plot_label):
        """
        Plot AI detections and manual counts, highlighting invalid points with red, or having them removed and interpolated.
        Interpolated points are shown in orange.

        Args:
            image_times: Array of image times (in decimal days).
            tank_counts_def: Default scaled tank counts.
            tank_std_def: Default scaled tank standard deviations.
            tank_counts_cal: Manually scaled tank counts.
            tank_std_cal: Manually scaled tank standard deviations.
            manual_counts: Array of manual counts.
            manual_std: Array of manual standard deviations.
            manual_times: Array of manual times (in decimal days).
            scaling_idx: Index of the scaling point.
            batched_invalid_indices: List of invalid indices mapped to batches.
            plot_label: A string to differentiate the plot (used in title and filename).
        """
        n = 0.5
        fig, ax = plt.subplots()
        
        # Process data based on SHOW_INVALID_POINTS setting
        if not self.SHOW_INVALID_POINTS:
            # Replace invalid points with interpolated values or remove them
            plot_times, plot_counts_cal, plot_std_cal, interpolated_mask = self.process_invalid_points(
                image_times, tank_counts_cal, tank_std_cal, batched_invalid_indices
            )
            
            if self.PLOT_FOCUS_VOLUME:
                # For focus-volume scaled counts, also get interpolated mask
                plot_counts_def, plot_std_def, _, focus_interpolated_mask = self.process_invalid_points(
                    image_times, tank_counts_def, tank_std_def, batched_invalid_indices
                )
        else:
            # Use original data when showing invalid points
            plot_times, plot_counts_cal, plot_std_cal = image_times, tank_counts_cal, tank_std_cal
            interpolated_mask = np.zeros(len(image_times), dtype=bool)
            
            if self.PLOT_FOCUS_VOLUME:
                plot_counts_def, plot_std_def = tank_counts_def, tank_std_def
                focus_interpolated_mask = np.zeros(len(image_times), dtype=bool)

        # AI counts (focus-volume scaled)
        if self.PLOT_FOCUS_VOLUME:
            # Plot regular points
            valid_mask = ~focus_interpolated_mask
            ax.plot(plot_times[valid_mask], plot_counts_def[valid_mask], label='focus-volume scaled', color='green')
            ax.fill_between(plot_times[valid_mask], 
                        plot_counts_def[valid_mask] - n * plot_std_def[valid_mask], 
                        plot_counts_def[valid_mask] + n * plot_std_def[valid_mask], 
                        alpha=0.2, color='green')
            
            # Plot interpolated points in orange
            if not self.SHOW_INVALID_POINTS and np.any(focus_interpolated_mask):
                ax.plot(plot_times[focus_interpolated_mask], plot_counts_def[focus_interpolated_mask], 
                    'o', color='orange', label='focus-volume interpolated')

        # AI counts (manually scaled)
        # Plot regular points
        valid_mask = ~interpolated_mask
        ax.plot(plot_times[valid_mask], plot_counts_cal[valid_mask], label='CSLICS Count (scaled)', color='blue')
        ax.fill_between(plot_times[valid_mask], 
                    plot_counts_cal[valid_mask] - n * plot_std_cal[valid_mask], 
                    plot_counts_cal[valid_mask] + n * plot_std_cal[valid_mask], 
                    alpha=0.2, color='blue')
        
        # Plot interpolated points in orange
        if not self.SHOW_INVALID_POINTS and np.any(interpolated_mask):
            ax.plot(plot_times[interpolated_mask], plot_counts_cal[interpolated_mask], 
                'o', color='orange', label='invalid points')

        # Highlight invalid points in red (only if SHOW_INVALID_POINTS is True)
        if self.SHOW_INVALID_POINTS:
            invalid_points_plotted = False  # Track if the legend entry for invalid points has been added
            for batch_idx, invalid_indices in enumerate(batched_invalid_indices):
                if invalid_indices:
                    invalid_times = [image_times[batch_idx]] * len(invalid_indices)
                    invalid_counts = [tank_counts_cal[batch_idx]] * len(invalid_indices)
                    if not invalid_points_plotted:
                        # Add label only for the first batch with invalid points
                        ax.scatter(invalid_times, invalid_counts, color='red', label='invalid points', zorder=5, s=5)
                        invalid_points_plotted = True
                    else:
                        # Plot without a label for subsequent batches
                        ax.scatter(invalid_times, invalid_counts, color='red', zorder=5, s=5)

        # Manual counts
        ax.plot(manual_times, manual_counts, marker='o', color='green', label='manual count')
        ax.errorbar(manual_times, manual_counts, yerr=n * manual_std, fmt='o', color='orange', alpha=0.5)

        # Highlight calibration points if they exist in the plot data
        calibration_manual_time = manual_times[self.calibration_idx]
        ax.plot(calibration_manual_time, manual_counts[self.calibration_idx], 
                marker='*', markersize=10, color='red', label='calibration')
        
        # Only show shifted calibration point if it's in the plot data
        if not self.SHOW_INVALID_POINTS and scaling_idx - 1 < len(plot_times):
            ax.plot(plot_times[scaling_idx - 1], plot_counts_cal[scaling_idx - 1], 
                    marker='*', markersize=10, color='black', label='shifted calibration')
        elif self.SHOW_INVALID_POINTS:
            ax.plot(image_times[scaling_idx - 1], tank_counts_cal[scaling_idx - 1], 
                    marker='*', markersize=10, color='black', label='shifted calibration')

        # Finalize plot
        plt.legend()
        plt.grid(True)
        plt.xlabel('Days since spawning')
        plt.ylabel(f'Tank count (batched {self.aggregate_size} images)')
        plt.title(f'CSLICS AI Count: {self.tank_sheet_name}, {self.cslics_uuid} - ({plot_label})')
        plt.tight_layout()
        
        # Ensure output directory exists
        os.makedirs(self.save_det_dir, exist_ok=True)
        
        output_path = os.path.join(self.save_det_dir, f'Combined_tank_counts_{self.tank_sheet_name}_{self.cslics_uuid}_{plot_label}.png')
        plt.savefig(output_path, dpi=600)
        print(f'Plot saved to {output_path}')
        if self.SHOW:
            plt.show()
        plt.close()

    def plot_error_between_manual_and_ai(self, image_times, tank_counts_cal, manual_times, manual_counts, batched_invalid_indices):
        """
        Compute and plot the error between manual counts and AI-calibrated tank counts.
        Only uses valid (non-excluded) time points for comparison.
        Saves the error data to a JSON file for later analysis.

        Args:
            image_times (array-like): Array of image times (in decimal days).
            tank_counts_cal (array-like): Array of AI-calibrated tank counts.
            manual_times (array-like): Array of manual times (in decimal days).
            manual_counts (array-like): Array of manual counts.
            batched_invalid_indices (list): List of lists containing invalid indices for each batch.
            
        Returns:
            tuple: A tuple containing errors and corresponding manual_times
        """
        # Process data based on SHOW_INVALID_POINTS setting
        # Even if SHOW_INVALID_POINTS is True, we still need to exclude invalid points from error calculation       
        valid_image_times, valid_tank_counts, _, _ = self.process_invalid_points(image_times, 
                                                                                tank_counts_cal, 
                                                                                np.zeros_like(tank_counts_cal), 
                                                                                batched_invalid_indices)
        
        if len(valid_image_times) == 0:
            print("Warning: No valid time points found for error calculation.")
            return [], []
        
        # Find the closest valid image time for each manual time
        closest_indices = [np.argmin(np.abs(valid_image_times - manual_time)) for manual_time in manual_times]
        closest_image_times = [valid_image_times[idx] for idx in closest_indices]
        closest_tank_counts = [valid_tank_counts[idx] for idx in closest_indices]

        # Compute the error (difference) between manual counts and AI-calibrated counts
        errors = np.array(closest_tank_counts) - np.array(manual_counts)
        
        # Calculate error statistics
        mean_abs_error = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        # Plot the error
        __, ax = plt.subplots()
        ax.plot(manual_times, errors, marker='o', color='red', label='Error (AI - Manual)')
        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Zero Error')
        plt.grid(True)
        plt.xlabel('Days since spawning')
        plt.ylabel('Error (Tank Counts)')
        plt.title(f'Error Between AI and Manual Counts: {self.tank_sheet_name}\nMAE: {mean_abs_error:.2f}, RMSE: {rmse:.2f}')
        plt.legend()

        # Adjust y-axis label formatting for better readability
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))  # Use scientific notation if needed
        plt.tight_layout()  # Ensure labels and titles fit within the figure

        # Ensure output directory exists
        os.makedirs(self.save_det_dir, exist_ok=True)
        
        # Save the plot
        output_path = os.path.join(self.save_det_dir, f'Error_plot_{self.tank_sheet_name}.png')
        plt.savefig(output_path, dpi=600)
        print(f"Error plot saved to {output_path}")
        print(f"Mean Absolute Error: {mean_abs_error:.2f}, Root Mean Square Error: {rmse:.2f}")
        
        if self.SHOW:
            plt.show()
        plt.close()
        
        # Save the error data to JSON
        self.save_error_data_to_json(manual_times, errors.tolist(), mean_abs_error, rmse)
        
        return manual_times, errors
        
    def save_error_data_to_json(self, manual_times, errors, mae, rmse):
        """
        Save error data to a JSON file.
        
        Args:
            manual_times (list): List of manual times (in decimal days).
            errors (list): List of errors between AI and manual counts.
            mae (float): Mean absolute error.
            rmse (float): Root mean square error.
        """
        # Create a dictionary to store the data
        error_data = {
            "tank_sheet_name": self.tank_sheet_name,
            "cslics_uuid": self.cslics_uuid,
            "species": self.coral_species,
            "manual_times": manual_times,
            "errors": errors,
            "statistics": {
                "mae": mae,
                "rmse": rmse
            }
        }
        
        # Create the output directory if it doesn't exist
        error_output_dir = os.path.join(self.save_det_dir, "error_data")
        os.makedirs(error_output_dir, exist_ok=True)
        
        # Save the data to a JSON file
        error_output_path = os.path.join(error_output_dir, f'error_data_{self.tank_sheet_name}_{self.cslics_uuid}.json')
        with open(error_output_path, 'w') as f:
            json.dump(error_data, f, indent=4)
        
        print(f"Error data saved to {error_output_path}")

    def run_full_analysis(self, det_dir, manual_counts, manual_std, manual_times, nearest_day, invalid_indices=None, show_plots=False):
        """
        Run the complete analysis pipeline.
        
        Args:
            det_dir: Detection directory path
            manual_counts: Array of manual counts
            manual_std: Array of manual standard deviations
            manual_times: Array of manual times
            nearest_day: Reference day for time calculations
            invalid_indices: List of invalid detection file indices
            show_plots: Whether to display plots interactively
            
        Returns:
            dict: Results dictionary containing all analysis outputs
        """
        # Set the SHOW property based on the show_plots parameter
        self.SHOW = show_plots
        
        # try:
        # Read detection files
        sample_list = self.read_detections(det_dir)
        
        # Batch detections
        image_counts, image_std, image_times, batched_invalid_indices = self.batch_detections(
            sample_list, nearest_day, invalid_indices)
        
        # Plot image detections
        self.plot_image_detections(image_counts, image_std, image_times)
        
        # Process and scale counts
        (tank_counts_def, tank_std_def), (tank_counts_cal, tank_std_cal), scaling_idx = self.process_and_scale_counts(
            image_counts, image_std, image_times, manual_counts, manual_std, manual_times
        )
        
        # Plot combined results
        self.plot_detections_and_manual_counts(
            image_times, tank_counts_def, tank_std_def, tank_counts_cal, tank_std_cal,
            manual_counts, manual_std, manual_times, scaling_idx, batched_invalid_indices,
            "combined"
        )
        
        # Plot error analysis
        manual_times_error, errors = self.plot_error_between_manual_and_ai(
            image_times, tank_counts_cal, manual_times, manual_counts, batched_invalid_indices
        )
        
        return {
            'image_counts': image_counts,
            'image_std': image_std,
            'image_times': image_times,
            'tank_counts_cal': tank_counts_cal,
            'tank_std_cal': tank_std_cal,
            'tank_counts_def': tank_counts_def,
            'tank_std_def': tank_std_def,
            'scaling_idx': scaling_idx,
            'batched_invalid_indices': batched_invalid_indices,
            'errors': errors,
            'manual_times_error': manual_times_error
        }
            
        # except Exception as e:
        #     print(f"Error in analysis pipeline: {e}")
        #     return None


if __name__ == "__main__":
    
    # config file
    config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plot_config_202312_t4_alor_cslics08.json"
    
    try:
        plotter = TankCountPlotter.from_config_file(config_path)
        print(f"Initialized TankCountPlotter for {plotter.cslics_uuid} ({plotter.coral_species})")
        
        # load manual counts data here and call run_full_analysis
        # manual_counts, manual_std, manual_times, nearest_day = load_manual_data()
        # results = plotter.run_full_analysis(manual_counts, manual_std, manual_times, nearest_day)
        
    except Exception as e:
        print(f"Error: {e}")