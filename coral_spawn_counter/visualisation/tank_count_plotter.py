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
        Initialize the tank count plotter with configuration.
        
        Args:
            config_manager: PlotConfigManager instance containing configuration
        """
        self.config_manager = config_manager
        self.config = config_manager.config
        
        # Plotting configuration
        self.cslics_uuid = self.config.cslics_uuid
        self.coral_species = self.config.coral_species
        self.tank_sheet_name = self.config.tank_sheet_name
        self.skipping_frequency = self.config.skipping_frequency
        self.aggregate_size = self.config.aggregate_size
        self.confidence_threshold = self.config.confidence_threshold
        self.MAX_SAMPLE = self.config.MAX_SAMPLE
        
        # Surface calibration parameters
        self.surface_calibration_idx = getattr(self.config, 'surface_calibration_idx', 0)
        self.surface_calibration_window_size = getattr(self.config, 'surface_calibration_window_size', 1)
        self.surface_calibration_window_shift = getattr(self.config, 'surface_calibration_window_shift', 0)
        
        # Subsurface calibration parameters  
        self.subsurface_calibration_idx = getattr(self.config, 'subsurface_calibration_idx', 0)
        self.subsurface_calibration_window_size = getattr(self.config, 'subsurface_calibration_window_size', 1)
        self.subsurface_calibration_window_shift = getattr(self.config, 'subsurface_calibration_window_shift', 0)
        
        # Detection processing parameters
        self.SHOW_INVALID_POINTS = False
        self.PLOT_FOCUS_VOLUME = False
        self.save_det_dir = os.path.join(self.config.base_detection_dir, self.config.cslics_uuid, "plots")
        os.makedirs(self.save_det_dir, exist_ok=True)
        self.batch_histogram_dir = os.path.join(self.save_det_dir, "batch_histograms")
        os.makedirs(self.batch_histogram_dir, exist_ok=True)
        self.submersion_time = getattr(self.config, 'submersion_time', None)

    def surface_calibration(self, batched_surface_counts, batched_surface_times, manual_counts, manual_times, calibration_idx=None):
        """
        Calculate calibration scale factor between manual counts and surface AI counts.
        
        Args:
            batched_surface_counts: Array of batched surface detection counts
            batched_surface_times: Array of batched surface detection times (hours since spawning)
            manual_counts: List/array of manual counts from dictionary
            manual_times: List/array of manual times (hours since spawning) from dictionary
            calibration_idx: Index of manual count to use for calibration (uses surface_calibration_idx if None)
            
        Returns:
            float: Scale factor to multiply surface counts by to match manual count
        """
        # Use surface-specific calibration index if not provided
        if calibration_idx is None:
            calibration_idx = self.surface_calibration_idx
            
        # Convert to numpy arrays for consistent processing
        batched_surface_counts = np.array(batched_surface_counts) if batched_surface_counts is not None else np.array([])
        batched_surface_times = np.array(batched_surface_times) if batched_surface_times is not None else np.array([])
        manual_counts = np.array(manual_counts) if manual_counts is not None else np.array([])
        manual_times = np.array(manual_times) if manual_times is not None else np.array([])
        
        # Check if we have valid data
        if len(batched_surface_counts) == 0 or len(manual_counts) == 0:
            print("Error: No surface counts or manual counts provided for calibration")
            return 1.0
        
        if calibration_idx >= len(manual_counts):
            print(f"Error: Surface calibration index {calibration_idx} exceeds manual counts length {len(manual_counts)}")
            return 1.0
        
        # Get the manual count and time for calibration
        target_manual_count = manual_counts[calibration_idx]
        target_manual_time = manual_times[calibration_idx]
        
        print(f"Surface calibration using manual count index {calibration_idx}")
        print(f"Surface calibration window size: {self.surface_calibration_window_size}")
        print(f"Surface calibration window shift: {self.surface_calibration_window_shift}")
        print(f"Target manual count: {target_manual_count} at time {target_manual_time:.2f} hours")
        
        # Apply window shift and size for surface calibration
        # This could be used to average over multiple surface detection points
        # Find the closest surface detection time to the manual calibration time
        time_differences = np.abs(batched_surface_times - target_manual_time)
        closest_surface_idx = np.argmin(time_differences)
        
        # Apply window around the closest point
        window_start = max(0, closest_surface_idx - self.surface_calibration_window_shift)
        window_end = min(len(batched_surface_counts), window_start + self.surface_calibration_window_size)
        
        # Calculate average over the window
        window_counts = batched_surface_counts[window_start:window_end]
        window_times = batched_surface_times[window_start:window_end]
        
        if len(window_counts) > 0:
            avg_surface_count = np.mean(window_counts)
            avg_surface_time = np.mean(window_times)
        else:
            avg_surface_count = batched_surface_counts[closest_surface_idx]
            avg_surface_time = batched_surface_times[closest_surface_idx]
        
        time_diff = abs(avg_surface_time - target_manual_time)
        print(f"Surface detection window: {len(window_counts)} points, avg count: {avg_surface_count:.2f} at time {avg_surface_time:.2f} hours")
        print(f"Time difference: {time_diff:.2f} hours")
        
        # Calculate scale factor
        if avg_surface_count > 0:
            scale_factor = target_manual_count / avg_surface_count
        else:
            print("Warning: Average surface count is 0, cannot calculate scale factor. Using 1.0")
            scale_factor = 1.0
        
        print(f"Calculated surface scale factor: {scale_factor:.4f}")
        print(f"Scaled surface count would be: {avg_surface_count * scale_factor:.2f}")
        
        return scale_factor

    def scale_by_manual_calibration_idx(self, manual_count, image_counts, closest_idx):
        """
        Scale image counts using manual calibration - updated for subsurface calibration parameters.
        
        Args:
            manual_count: Manual count value at calibration point
            image_counts: Array of image detection counts
            closest_idx: Index of closest image detection to manual count time
            
        Returns:
            float: Scale factor
        """
        print(f"Subsurface calibration using index {closest_idx}")
        print(f"Subsurface calibration window size: {self.subsurface_calibration_window_size}")
        print(f"Subsurface calibration window shift: {self.subsurface_calibration_window_shift}")
        
        # Apply window shift and size for subsurface calibration
        window_start = max(0, closest_idx - self.subsurface_calibration_window_shift)
        window_end = min(len(image_counts), window_start + self.subsurface_calibration_window_size)
        
        # Calculate average over the window
        window_counts = image_counts[window_start:window_end]
        
        if len(window_counts) > 0:
            avg_image_count = np.mean(window_counts)
            print(f"Subsurface detection window: {len(window_counts)} points, avg count: {avg_image_count:.2f}")
        else:
            avg_image_count = image_counts[closest_idx]
            print(f"Using single subsurface detection: {avg_image_count:.2f}")
        
        if avg_image_count > 0:
            scale_factor = manual_count / avg_image_count
        else:
            print("Warning: Average subsurface count is 0, cannot calculate scale factor. Using 1.0")
            scale_factor = 1.0
        
        print(f"Calculated subsurface scale factor: {scale_factor:.4f}")
        return scale_factor

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
    
    def set_batch_histogram_dir(self, batch_histogram_dir):
        """
        Set the directory for saving batch histogram plots.
        
        Args:
            batch_histogram_dir: Path to the batch histogram directory
        """
        self.batch_histogram_dir = batch_histogram_dir

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
        
        # Save the plot
        output_path = os.path.join(self.batch_histogram_dir, f'Batch_{batch_idx + 1}_Histogram.png')
        plt.savefig(output_path, dpi=600)
        print(f"Histogram for Batch {batch_idx + 1} saved to {output_path}")

        if self.SHOW:
            plt.show()
        plt.close()

    def plot_image_detections(self, counts, std, times, nearest_day=None):
        """
        Plot image-based detections with error bands. This is essentially the subsurface plot.

        Args:
            counts (array-like): Array of detection counts.
            std (array-like): Array of standard deviations for the counts.
            times (array-like): Array of times (in decimal days) since spawning.
            nearest_day (datetime): Reference day for submersion time calculations.
        """
        n = 1  # Multiplier for the error band
        __, ax = plt.subplots()
        ax.plot(times, counts, label='Detections')
        ax.fill_between(times, counts - n * std, counts + n * std, alpha=0.2, label='Error Band')
        plt.grid(True)
        plt.xlabel('Days since spawning')
        plt.ylabel(f'Image count (batched {self.aggregate_size} images)')
        
        # Get subsurface model name for the title
        try:
            subsurface_model_name = self.config_manager.get_subsurface_model_name()
            plt.title(f'CSLICS AI Count: {self.tank_sheet_name}, {self.cslics_uuid}\nSubsurface Model: {subsurface_model_name}')
        except Exception as e:
            print(f"Warning: Could not get subsurface model name: {e}")
            plt.title(f'CSLICS AI Count: {self.tank_sheet_name}, {self.cslics_uuid}')
        
        plt.legend()
        
        # Set x-axis limits from submersion time to the last time point
        if hasattr(self, 'submersion_time') and self.submersion_time and len(times) > 0 and nearest_day:
            try:
                # Parse submersion time and convert to decimal days since spawning
                submersion_datetime = datetime.strptime(self.submersion_time, '%Y-%m-%d_%H-%M-%S')
                # Convert to decimal days using the same method as other times
                submersion_days = (submersion_datetime - nearest_day).total_seconds() / (60 * 60 * 24)
                
                # Set x-axis limits from submersion time to last time point
                ax.set_xlim(submersion_days, times[-1])
                print(f"X-axis limited from submersion time ({submersion_days:.3f} days) to last time point ({times[-1]:.3f} days)")
            except Exception as e:
                print(f"Warning: Could not parse submersion_time for x-axis limits: {e}")
        elif len(times) > 0:
            # If no submersion time, just set to start from first time point
            ax.set_xlim(times[0], times[-1])
        
        output_path = os.path.join(self.save_det_dir, f'Image_counts_{self.tank_sheet_name}.png')
        plt.savefig(output_path)
        print(f"Image detections plot saved to {output_path}")
        if self.SHOW:
            plt.show()
        plt.close()
            
    def find_closest_time(self, image_time, manual_time, manual_idx=None, detection_type='surface'):
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
            if detection_type == 'surface':
                manual_idx = self.surface_calibration_idx
            else:
                manual_idx = self.subsurface_calibration_idx    
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

    def scale_by_manual_calibration_idx(self, manual_count, image_counts, closest_idx, calibration_window_size=None, calibration_window_shift=None):
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
        idx_select = closest_idx + calibration_window_shift
        
        # Find the idx for the nearest time to the specified calibration manual time
        # accounting for min/max sizes of image_counts
        idx_min = int(idx_select - calibration_window_size/2)
        if idx_min < 0:
            idx_min = 0
            if len(image_counts) <= calibration_window_size:
                idx_max = len(image_counts) - 1
            else:
                idx_max = calibration_window_size
        else:
            idx_max = int(idx_min + calibration_window_size)
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

    def process_and_scale_counts(self, image_counts, image_std, image_times, manual_counts, manual_std, manual_times, detection_type='surface'):
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
        
        if detection_type == 'surface':
            calibration_idx = self.surface_calibration_idx
            calibration_window_size = self.surface_calibration_window_size
            calibration_window_shift = self.surface_calibration_window_shift
        else:
            calibration_idx = self.subsurface_calibration_idx
            calibration_window_size = self.subsurface_calibration_window_size
            calibration_window_shift = self.subsurface_calibration_window_shift
            
        # Scale factor by focus volume
        scale_factor_focus = self.scale_by_focus_volume()

        # Apply scale factor
        tank_counts_def = image_counts * scale_factor_focus
        tank_std_def = image_std * scale_factor_focus

        # Find the closest time for manual calibration
        closest_idx, __ = self.find_closest_time(image_times, manual_times)

        # Scale factor by manual calibration
        scale_factor_manual, scaling_idx = self.scale_by_manual_calibration_idx(
            manual_counts[calibration_idx], image_counts, closest_idx, calibration_window_size, calibration_window_shift    
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
        plot_label,
        detection_type='surface'):
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
            detection_
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
        if detection_type == 'surface':
            calibration_idx = self.surface_calibration_idx
        else:
            calibration_idx = self.subsurface_calibration_idx
            
        calibration_manual_time = manual_times[calibration_idx]
        ax.plot(calibration_manual_time, manual_counts[calibration_idx], 
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
        Only calculates errors for manual times after the submersion time.
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
        
        # Filter manual times to only include those after submersion time
        # HACK in handling post-submersion time filtering, defined in the middle of coral_spawn_counter.py
        if hasattr(self, 'submersion_time_decimal') and self.submersion_time_decimal:
            try:                
                # Filter manual times and counts to only include post-submersion
                post_submersion_mask = np.array(manual_times) > self.submersion_time_decimal
                filtered_manual_times = np.array(manual_times)[post_submersion_mask]
                filtered_manual_counts = np.array(manual_counts)[post_submersion_mask]
                
                print(f"Filtering to {len(filtered_manual_times)} manual times after submersion (from {len(manual_times)} total)")
                
            except Exception as e:
                print(f"Warning: Could not parse submersion_time for error calculation: {e}")
                print("Using all manual times for error calculation")
                filtered_manual_times = np.array(manual_times)
                filtered_manual_counts = np.array(manual_counts)
        else:
            print("Warning: No submersion_time configured, using all manual times for error calculation")
            filtered_manual_times = np.array(manual_times)
            filtered_manual_counts = np.array(manual_counts)
        
        if len(filtered_manual_times) == 0:
            print("Warning: No manual times found after submersion time.")
            return [], []
        
        # Find the closest valid image time for each filtered manual time
        closest_indices = [np.argmin(np.abs(valid_image_times - manual_time)) for manual_time in filtered_manual_times]
        closest_image_times = [valid_image_times[idx] for idx in closest_indices]
        closest_tank_counts = [valid_tank_counts[idx] for idx in closest_indices]

        # Compute the error (difference) between manual counts and AI-calibrated counts
        errors = np.array(closest_tank_counts) - np.array(filtered_manual_counts)
        
        # Calculate error statistics
        mean_abs_error = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors**2))
        
        # Plot the error
        __, ax = plt.subplots()
        ax.plot(filtered_manual_times, errors, marker='o', color='red', label='Error (AI - Manual)')
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
        self.save_error_data_to_json(filtered_manual_times.tolist(), errors.tolist(), mean_abs_error, rmse)
        
        return filtered_manual_times, errors
        
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

    def run_full_analysis(self, 
                          det_dir, 
                          manual_counts, 
                          manual_std, 
                          manual_times, 
                          nearest_day, 
                          invalid_indices=None, 
                          show_plots=False, 
                          batch_histogram_dir=None, 
                          detection_type='surface'):
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
            batch_histogram_dir: Optional directory for batch histogram plots
            
        Returns:
            dict: Results dictionary containing all analysis outputs
        """
        # Set the SHOW property based on the show_plots parameter
        self.SHOW = show_plots
        
        # Set batch histogram directory if provided
        if batch_histogram_dir:
            self.set_batch_histogram_dir(batch_histogram_dir)
        
        # try:
        # Read detection files
        sample_list = self.read_detections(det_dir)
        
        # Batch detections
        image_counts, image_std, image_times, batched_invalid_indices = self.batch_detections(
            sample_list, nearest_day, invalid_indices)
        
        # Plot image detections
        self.plot_image_detections(image_counts, image_std, image_times, nearest_day)
        
        # Process and scale counts
        (tank_counts_def, tank_std_def), (tank_counts_cal, tank_std_cal), scaling_idx = self.process_and_scale_counts(
            image_counts, image_std, image_times, manual_counts, manual_std, manual_times, detection_type)
        
        # Plot error analysis
        manual_times_error, errors = self.plot_error_between_manual_and_ai(
            image_times, tank_counts_cal, manual_times, manual_counts, batched_invalid_indices
        )
        
        # Plot combined results
        self.plot_detections_and_manual_counts(
            image_times, tank_counts_def, tank_std_def, tank_counts_cal, tank_std_cal,
            manual_counts, manual_std, manual_times, scaling_idx, batched_invalid_indices,
            "combined", detection_type)
        
        
        
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


    def median_absolute_deviation(self, data):
        """
        Compute the median absolute deviation (MAD) for a list or array.
        Args:
            data: list or numpy array
        Returns:
            float: MAD value
        """
        data = np.array(data)
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        return median, mad

    def surface_tank_estimate(self, surface_df, class_names, manual_counts, manual_std, manual_times, nearest_day, show_plots=False, apply_calibration=True, calibration_idx=0):
        """
        Process surface detection DataFrame to create tank estimates by summing all classes except 'Damaged'.
        Uses batched approach similar to subsurface processing and applies calibration scaling.
        Now uses median and MAD for batch statistics.
        """
        print("Processing surface detection DataFrame for batched tank estimation...")
        
        if surface_df is None or surface_df.empty:
            print("No surface detection data provided")
            return None, None, None
        
        manual_counts = list(manual_counts) if manual_counts is not None else []
        manual_std = list(manual_std) if manual_std is not None else []
        manual_times = list(manual_times) if manual_times is not None else []
        
        damaged_class_idx = None
        damaged_class_name = None
        if 'Damaged' in class_names:
            damaged_class_idx = class_names.index('Damaged')
            damaged_class_name = str(damaged_class_idx)
            print(f"Excluding 'Damaged' class (index {damaged_class_idx}, column '{damaged_class_name}') from tank estimates")
        else:
            print("No 'Damaged' class found in class names")
        
        class_columns = [col for col in surface_df.columns if col not in ['timestamp', 'hours_since_spawning']]
        if not class_columns:
            print("No class columns found in DataFrame")
            return None, None, None
        
        print(f"Found class columns: {class_columns}")
        
        row_total_counts = []
        for _, row in surface_df.iterrows():
            total_count = 0
            for col in class_columns:
                if col != damaged_class_name:
                    total_count += row[col]
            row_total_counts.append(total_count)
        
        if 'hours_since_spawning' in surface_df.columns:
            times_hours = surface_df['hours_since_spawning'].tolist()
        else:
            print("Warning: 'hours_since_spawning' column not found, using index as time")
            times_hours = list(range(len(surface_df)))
        
        print(f"Applying batching with aggregate_size={self.aggregate_size}, skipping_frequency={self.skipping_frequency}")
        downsampled_indices = list(range(0, len(row_total_counts), self.skipping_frequency))
        downsampled_counts = [row_total_counts[i] for i in downsampled_indices]
        downsampled_times = [times_hours[i] for i in downsampled_indices]
        
        batched_counts = [downsampled_counts[i:i + self.aggregate_size] 
                         for i in range(0, len(downsampled_counts), self.aggregate_size)]
        batched_times = [downsampled_times[i:i + self.aggregate_size] 
                        for i in range(0, len(downsampled_times), self.aggregate_size)]
        
        batched_total_counts = []
        batched_std = []
        batched_times_hours = []
        
        for batch_idx, (count_batch, time_batch) in enumerate(zip(batched_counts, batched_times)):
            if len(count_batch) > 0:
                batch_median, batch_mad = self.median_absolute_deviation(count_batch)
                batch_time = np.median(time_batch)
                batched_total_counts.append(batch_median)
                batched_std.append(batch_mad)
                batched_times_hours.append(batch_time)
                print(f"Batch {batch_idx + 1}: {len(count_batch)} points, "
                      f"median={batch_median:.1f}, MAD={batch_mad:.1f}, time={batch_time:.1f}h")
    
        if hasattr(self, 'MAX_SAMPLE') and self.MAX_SAMPLE is not None:
            if len(batched_total_counts) > self.MAX_SAMPLE:
                print(f"Limiting to first {self.MAX_SAMPLE} batches (from {len(batched_total_counts)})")
                batched_total_counts = batched_total_counts[:self.MAX_SAMPLE]
                batched_std = batched_std[:self.MAX_SAMPLE]
                batched_times_hours = batched_times_hours[:self.MAX_SAMPLE]
                
            
        self.plot_density_vs_time(batched_total_counts, batched_times_hours)
        
        manual_times_hours = []
        if len(manual_times) > 0:
            if isinstance(manual_times[0], datetime):
                for dt in manual_times:
                    hours_since = (dt - nearest_day).total_seconds() / 3600
                    manual_times_hours.append(hours_since)
            else:
                for decimal_day in manual_times:
                    if isinstance(decimal_day, (int, float)):
                        nearest_day_decimal = nearest_day.timestamp() / (24 * 3600 * 1000)
                        hours_since = (decimal_day - nearest_day_decimal) * 24
                        manual_times_hours.append(hours_since)
                    else:
                        print(f"Warning: Unexpected manual time format: {type(decimal_day)}")
                        manual_times_hours.append(0)
        
        print(f"Converted {len(manual_times)} manual times to hours since spawning")
        if manual_times_hours:
            print(f"Manual time range: {min(manual_times_hours):.1f} to {max(manual_times_hours):.1f} hours")
            
        scale_factor = 1.0
        if apply_calibration and len(batched_total_counts) > 0 and len(manual_counts) > calibration_idx:
            scale_factor = self.surface_calibration(batched_total_counts, batched_times_hours, 
                                                    manual_counts, manual_times_hours, calibration_idx)
            
        batched_total_counts_scaled = [count * scale_factor for count in batched_total_counts]
        batched_std_scaled = [std * scale_factor for std in batched_std]
        
        self.plot_surface_tank_estimate(
                batched_times_hours, batched_total_counts_scaled, batched_std_scaled, 
                manual_counts, manual_std, manual_times_hours, nearest_day,
                class_names, damaged_class_idx, show_plots, scale_factor
            )
            
        print(f"Processed {len(surface_df)} surface detection data points into {len(batched_total_counts)} batches")
        if batched_times_hours:
            print(f"Batched time range: {min(batched_times_hours):.1f} to {max(batched_times_hours):.1f} hours since spawning")
        if batched_total_counts_scaled:
            print(f"Scaled count range: {min(batched_total_counts_scaled):.1f} to {max(batched_total_counts_scaled):.1f} detections")
        
        return batched_times_hours, batched_total_counts_scaled, scale_factor
        
    def plot_surface_tank_estimate(self, 
                                   times_hours, 
                                   total_counts, 
                                   total_std, 
                                   manual_counts, 
                                   manual_std, 
                                   manual_times_hours, 
                                   nearest_day, 
                                   class_names, 
                                   damaged_class_idx, 
                                   show_plots=False, scale_factor=1.0):
        """
        Plot surface tank estimates over time with manual counts up to submersion time.
        
        Args:
            times_hours: Array of batched times in hours since spawning
            total_counts: Array of batched total detection counts (scaled)
            total_std: Array of standard deviations for batched counts (scaled)
            manual_counts: List/array of manual counts from dictionary
            manual_std: List/array of manual standard deviations from dictionary 
            manual_times_hours: List/array of manual times in hours since spawning
            nearest_day: Reference datetime for submersion time calculation
            class_names: List of class names
            damaged_class_idx: Index of the damaged class (None if not present)
            show_plots: Whether to display the plot interactively
            scale_factor: Scale factor applied to counts (for display in title)
        """
        if not times_hours or not total_counts:
            print("No data to plot")
            return
            
        # Convert to numpy arrays for consistency with subsurface plotting
        times_hours = np.array(times_hours)
        total_counts = np.array(total_counts)
        total_std = np.array(total_std) if total_std else np.zeros_like(total_counts)
        
        # Convert manual data to numpy arrays (handle lists from dictionary)
        manual_counts = np.array(manual_counts) if manual_counts is not None else np.array([])
        manual_std = np.array(manual_std) if manual_std is not None else np.array([])
        manual_times_hours = np.array(manual_times_hours) if manual_times_hours is not None else np.array([])
        
        # Calculate submersion time in hours since spawning
        if hasattr(self, 'submersion_time') and self.submersion_time:
            try:
                submersion_datetime = datetime.strptime(self.submersion_time, '%Y-%m-%d_%H-%M-%S')
                submersion_hours = (submersion_datetime - nearest_day).total_seconds() / 3600
                print(f"Submersion time: {submersion_hours:.2f} hours since spawning")
            except Exception as e:
                print(f"Warning: Could not parse submersion_time: {e}, showing all manual counts")
                submersion_hours = float('inf')
        else:
            print("Warning: No submersion_time configured, showing all manual counts")
            submersion_hours = float('inf')
        
        # Filter manual counts to only show up to submersion time
        if len(manual_times_hours) > 0:
            manual_mask = manual_times_hours <= submersion_hours
            filtered_manual_times = manual_times_hours[manual_mask]
            filtered_manual_counts = manual_counts[manual_mask]
            filtered_manual_std = manual_std[manual_mask]
            
            print(f"Showing {len(filtered_manual_times)} manual counts out of {len(manual_times_hours)} (up to submersion)")
        else:
            filtered_manual_times = np.array([])
            filtered_manual_counts = np.array([])
            filtered_manual_std = np.array([])
            print("No manual counts to display")
        
        # Calculate upper and lower bounds for surface detections
        upper_bound = total_counts + total_std
        lower_bound = total_counts - total_std
        
        # Create figure with same size as subsurface plots
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot surface detections
        ax.plot(times_hours, total_counts, 'o-', linewidth=1.5, markersize=4,
                color='red', alpha=0.8, label='Surface Detections (scaled)')
        
        # Plot upper and lower bound lines for surface detections
        ax.plot(times_hours, upper_bound, '--', linewidth=1, color='red', alpha=0.6)
        ax.plot(times_hours, lower_bound, '--', linewidth=1, color='red', alpha=0.6)
        
        # Fill between upper and lower bounds for surface detections
        ax.fill_between(times_hours, lower_bound, upper_bound, 
                       alpha=0.2, color='red')
        
        # Plot manual counts (up to submersion time)
        if len(filtered_manual_times) > 0:
            ax.plot(filtered_manual_times, filtered_manual_counts, 'o-', 
                   linewidth=1.5, markersize=6, color='green', alpha=0.8, label='Manual Counts')
            
            # Add error bars for manual counts
            n = 1  # Same as in subsurface plots
            ax.fill_between(filtered_manual_times, 
                           filtered_manual_counts - n * filtered_manual_std, 
                           filtered_manual_counts + n * filtered_manual_std, 
                           alpha=0.2, color='green')
            
            # Highlight surface calibration point if it's visible
            if self.surface_calibration_idx < len(filtered_manual_counts):
                ax.plot(filtered_manual_times[self.surface_calibration_idx], filtered_manual_counts[self.surface_calibration_idx], 
                       marker='*', markersize=10, color='orange', label='Surface Calibration')
        
        # Add vertical line at submersion time if it's within the plot range
        if submersion_hours != float('inf') and len(times_hours) > 0 and submersion_hours <= max(times_hours):
            ax.axvline(x=submersion_hours, color='black', linestyle='--', alpha=0.7, 
                      label=f'Submersion ({submersion_hours:.1f}h)')
        
        # Match subsurface plot styling
        ax.set_xlabel('Hours since spawning')
        ax.set_ylabel('Detection counts')
        
        # Create title matching subsurface format
        excluded_info = " (excluding Damaged)" if damaged_class_idx is not None else ""
        scale_info = f" (scale: {scale_factor:.3f})" if scale_factor != 1.0 else ""
        title = f'{self.cslics_uuid} - Surface Detections{excluded_info}{scale_info} - {self.coral_species} - Tank: {self.tank_sheet_name}'
        ax.set_title(title, fontweight='bold')
        
        # Apply same grid style as subsurface
        ax.grid(True, alpha=0.3)
        
        # Set y-axis to start from 0 (common in subsurface plots)
        ax.set_ylim(bottom=0)
        
        # Add legend with same style
        ax.legend(loc='upper right')
        
        # Apply tight layout
        plt.tight_layout()
                
        # Save with filename format matching subsurface plots
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_filename = f'surface_detections_with_manual_{self.tank_sheet_name}_{self.cslics_uuid}_{current_time}.png'
        output_path = os.path.join(self.save_det_dir, plot_filename)

        # Save with same DPI and format as subsurface plots
        plt.savefig(output_path, dpi=600, bbox_inches='tight')
        print(f"Surface tank estimate plot with manual counts saved to: {output_path}")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
            
    def plot_density_vs_time(self, image_counts, image_times, fov_mm2=2.0, tank_diameter_m=1.0):
        """
        Plot coral density (corals per square meter) vs time.
        Density is calculated as (image_counts / focus_volume) / tank_area.

        Args:
            image_counts (array-like): Array of coral counts per image batch.
            image_times (array-like): Array of times (in decimal days) since spawning.
            focus_volume_ml (float): Camera focus volume in mL (default 2.0 mL).
            tank_diameter_m (float): Tank diameter in meters (default 1.0 m).
        """
        # Convert focus volume from mL to m^3
        fov_m2 = fov_mm2 * 1e-6
        # Calculate tank area (πr^2)
        tank_radius_m = tank_diameter_m / 2
        tank_area_m2 = np.pi * tank_radius_m ** 2

        # Calculate density: (count / focus_volume) / tank_area
        density_per_m2 = (np.array(image_counts) / fov_m2) / tank_area_m2

        # Plot density vs time
        plt.figure(figsize=(10, 6))
        plt.plot(image_times, density_per_m2, marker='o', color='purple', label='Coral Density (per m²)')
        plt.xlabel('Days since spawning')
        plt.ylabel('Coral density (count/m²)')
        plt.title(f'Coral Density vs Time\nTank: {self.tank_sheet_name}, {self.cslics_uuid}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save plot
        output_path = os.path.join(self.save_det_dir, f'Density_vs_Time_{self.tank_sheet_name}_{self.cslics_uuid}.png')
        plt.savefig(output_path, dpi=600)
        print(f'Density vs Time plot saved to {output_path}')

        plt.close()