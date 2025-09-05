#!/usr/bin/env python3

"""
CSLICS Data Processor
- Encapsulates functionality for processing and plotting tank estimates with manual counts.
"""

import os
import json
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm 

class CSLICSDataProcessor:
    def __init__(self, config_file):
        """
        Initialize the CSLICS Data Processor using a JSON configuration file.

        Args:
            config_file (str): Path to the JSON configuration file.
        """
        # Load configuration from the JSON file
        self.config = self.load_config_from_json(config_file)

        self.manual_counts_file = self.config['manual_counts_file']
        self.spawning_sheet_name = self.config['spawning_sheet_name']
        self.tank_sheet_name = self.config['tank_sheet_name']
        self.cslics_associations_file = self.config['cslics_associations_file']
        self.cslics_invalid_times_file = self.config['invalid_ranges_file']
        self.cslics_uuid = self.config['cslics_uuid']
        self.coral_species = self.config['coral_species']
        # self.cslics_uuid, self.coral_species = self.read_cslics_uuid_tank_association()
        # print(f'CSLICS UUID: {self.cslics_uuid}, Coral Species: {self.coral_species}')
        
        self.model_name = self.config['model_name']
        self.base_det_dir = self.config['base_detection_dir']
        self.save_det_dir = self._determine_detection_directory()
        
        self.save_manual_plot_dir = self.config['save_manual_plot_dir']

        # Additional configuration parameters
        self.skipping_frequency = self.config.get('skipping_frequency', 1)
        self.aggregate_size = self.config.get('aggregate_size', 100)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.5)
        self.MAX_SAMPLE = self.config.get('MAX_SAMPLE', 1000)
        self.calibration_window_size = self.config.get('calibration_window_size', 1)
        self.calibration_idx = self.config.get('calibration_idx', 1)
        self.calibration_window_shift = self.config.get('calibration_window_shift', 0)
        self.PLOT_FOCUS_VOLUME = self.config.get('PLOT_FOCUS_VOLUME', False)
        self.SHOW_INVALID_POINTS = self.config.get('SHOW_INVALID_POINTS', True)  
        
        os.makedirs(self.save_manual_plot_dir, exist_ok=True)
        os.makedirs(self.save_det_dir, exist_ok=True)
        print(f'CSLICS UUID: {self.cslics_uuid}, Coral Species: {self.coral_species}')
    
    def _determine_detection_directory(self):
        return f'{self.base_det_dir}/{self.cslics_uuid}/{self.model_name}'    
        
        
    @staticmethod
    def load_config_from_json(config_file):
        """
        Load configuration from a JSON file.

        Args:
            config_file (str): Path to the JSON configuration file.

        Returns:
            dict: Configuration dictionary.
        """
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
            print(f"Configuration loaded successfully from {config_file}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Configuration file {config_file} not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error: Failed to parse JSON file {config_file}. Details: {e}")


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
        scaled_counts = counts/1000 # for readability
        scaled_std = std/1000
        n = 1.0

        __, ax = plt.subplots()
        ax.plot(days, scaled_counts, marker='o', color='blue')
        ax.errorbar(days, scaled_counts, yerr= n*scaled_std, fmt='o', color='blue', alpha=0.5)
        plt.grid(True)
        plt.xlabel('Days since spawning')
        plt.ylabel('Tank count (in thousands for 500L)')
        plt.title(f'CSLICS Manual Count: {self.tank_sheet_name}')
        plt.savefig(os.path.join(self.save_manual_plot_dir, f'Manual_counts_{self.tank_sheet_name}.png'))
        if SHOW:
            plt.show()
        
        
    def read_invalid_times(self):
        """
        Reads a JSON file containing image exclusion ranges and returns it as a dictionary.
        NOTE: the image names should actually the the detection json files for compatibility with self.read_detections()

        :param json_path: Path to the JSON file.
        :return: Dictionary with the JSON contents.
        """
        try:
            with open(self.cslics_invalid_times_file, "r") as file:
                data = json.load(file)
            return data
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            return None  # Return None if there's an error
        
    
    def filter_invalid_times(self, detection_names, invalid_ranges):
        """
        Filter out invalid image names from the list of image detection names (json files).

        Args:
            detection_names: List of image filenames, chronologically ordered.
            invalid_ranges: List of dictionaries with 'start' and 'end' keys specifying invalid ranges.

        Returns:
            Tuple containing two lists:
                - invalid_names: List of invalid image filenames.
                - invalid_indices: List of indices of invalid image filenames.
                - valid_names: List of valid image filenames.
                - valid_indices: List of indices of valid image filenames.
        """
        # Handle the case where invalid_ranges is empty or None
        if not invalid_ranges:
            print("No invalid ranges provided. All detection names are considered valid.")
            return [], [], detection_names, list(range(len(detection_names)))

        invalid_names = set()

        # Iterate through each invalid range and collect invalid image names
        for range_ in invalid_ranges:
            start = range_['start'].replace(".jpg", "_det")
            end = range_['end'].replace(".jpg", "_det")
            # Add all image names within the range (inclusive) to the invalid set
            invalid_names.update(
                name for name in detection_names if start <= name <= end
            )

        # Convert invalid_names to a list and sort it

        invalid_names = sorted(invalid_names)

        # Determine valid names by subtracting invalid names from the full list
        valid_names = [name for name in detection_names if name not in invalid_names]

        invalid_indices = [detection_names.index(name) for name in invalid_names if name in detection_names]
        valid_indices = [detection_names.index(name) for name in valid_names if name in detection_names]
        return invalid_names, invalid_indices, valid_names, valid_indices
    
    
    def read_detections(self):
        # read in all the json files
        # it is assumed that because of the file naming structure, sorting the files by their filename sorts them chronologically
        print(f'Gathering detection files from: {self.save_det_dir}')
        sample_list = sorted(Path(self.save_det_dir).rglob('*_det.json'))
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
        for batch_idx, sample_batch in tqdm(enumerate(batched_samples[:self.MAX_SAMPLE]), 
                                        total=min(len(batched_samples), self.MAX_SAMPLE),
                                        desc="Processing batches"):
            sample_count = []
            batch_invalid_indices = []

            # Process files within the current batch
            for i, detection_file in enumerate(sample_batch):
                try:
                    with open(detection_file, 'r') as f:
                        data = json.load(f)
                    detections = data['detections [xn1, yn1, xn2, yn2, conf, cls]']
                    sample_count.append(sum(1 for d in detections if d[4] >= self.confidence_threshold))

                    # Check if this file is invalid
                    if invalid_indices and sample_list.index(detection_file) in invalid_indices:
                        batch_invalid_indices.append(i)

                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f'Error reading {detection_file}: {e}')

            # get plot of batch:
            self.plot_batch_histogram(sample_count, batch_idx, SHOW=False)
            
            # Average stats over the batch
            batched_image_count.append(np.mean(sample_count))
            batched_std.append(np.std(sample_count))
            # for 2024 capture string
            # capture_time_str = Path(sample_batch[len(sample_batch) // 2]).stem[:-10]
            # for 2023 capture string
            capture_time_str = Path(sample_batch[len(sample_batch) // 2]).stem[9:-15]
            # 2024 capture time string format
            # batched_time.append(datetime.strptime(capture_time_str, "%Y-%m-%d_%H-%M-%S"))
            # 2023 capture time string format
            # import code
            # code.interact(local=dict(globals(), **locals()))
            batched_time.append(datetime.strptime(capture_time_str, "%Y%m%d_%H%M%S"))
            batched_invalid_indices.append(batch_invalid_indices)

        # Convert batched_time to decimal days and zero the time since spawning
        decimal_capture_times = self.convert_to_decimal_days(batched_time, nearest_day)
        return np.array(batched_image_count), np.array(batched_std), np.array(decimal_capture_times), batched_invalid_indices


    def plot_batch_histogram(self, batch_counts, batch_idx, x_range=(0,60), y_max=50, SHOW=False):
        """
        Plot a histogram of the counts in a single batch with consistent x and y axes.

        Args:
            batch_counts (list): List of detection counts in the batch.
            batch_idx (int): Index of the batch (used for labeling the plot).
            x_range (tuple): Tuple specifying the global x-axis range (min, max).
            y_max (int): Maximum value for the y-axis (frequency).
            SHOW (bool): Whether to display the plot interactively. Default is False.
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
        output_path = os.path.join(self.save_det_dir, f'Batch_{batch_idx + 1}_Histogram.png')
        plt.savefig(output_path, dpi=600)
        print(f"Histogram for Batch {batch_idx + 1} saved to {output_path}")

        if SHOW:
            plt.show()
        plt.close()
            
            
    def plot_image_detections(self, counts, std, times, SHOW=False):
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
        output_path = os.path.join(self.save_det_dir, f'Image_counts_{self.tank_sheet_name}.png')
        plt.savefig(output_path)
        print(f"Image detections plot saved to {output_path}")
        if SHOW:
            plt.show()
        

    def get_hyper_focal_dist(self, f, c, n):
        """Calculate the hyper-focal distance."""
        return f + f**2 / (c * n)


    def scale_by_focus_volume(self):
        """
        A physics-based approach to solving the calibration problem.
        Calculates the scale factor based on the focus volume.
        """
        # Camera and sensor parameters
        width_pix = 4056  # pixels
        height_pix = 3040  # pixels
        pix_size = 1.55 / 1000  # um -> mm, pixel size
        sensor_width = width_pix * pix_size  # mm
        sensor_height = height_pix * pix_size  # mm
        f = 12  # mm, focal length
        aperture = 2.8  # f-stop number of the lens
        c = 0.1  # mm, circle of confusion
        focus_dist = 75  # mm, focusing distance (working distance of the camera)

        # Calculate hyper-focal distance
        hyp_dist = self.get_hyper_focal_dist(f, c, aperture)

        # Calculate depth of field
        dof_far = (hyp_dist * focus_dist) / (hyp_dist - (focus_dist - f))
        dof_near = (hyp_dist * focus_dist) / (hyp_dist + (focus_dist - f))
        dof_diff = abs(dof_far - dof_near)  # mm
        print(f'DoF diff = {dof_diff} mm')

        # Calculate field of view
        work_dist = focus_dist  # mm
        hfov = work_dist * sensor_height / (1.33 * f)  # mm, horizontal field-of-view
        vfov = work_dist * sensor_width / (1.33 * f)  # mm, vertical field-of-view
        print(f'horizontal FOV = {hfov}')
        print(f'vertical FOV = {vfov}')

        # Calculate focus volume
        area_cslics = hfov * vfov  # mm^2
        print(f'area_cslics = {area_cslics} mm^2')
        focus_volume = area_cslics * dof_diff  # mm^3
        print(f'focus volume = {focus_volume} mm^3')
        print(f'focus volume = {focus_volume / 1000} mL')

        # Calculate scale factor
        volume_image = focus_volume / 1000  # mL
        volume_tank = 475 * 1000  # 500 L = 500000 mL
        scale_factor = volume_tank / volume_image
        print(f'default scale factor = {scale_factor}')

        return scale_factor
    
    
    # manual scaling based on calibration index (see config)
    def find_closest_time(self, image_time, manual_time, manual_idx=None):
        """ assuming both image_time, and manual_time are datetime objects"""
        if manual_idx is None:
            manual_idx = self.calibration_idx
        t_diff = abs(image_time - manual_time[manual_idx])
        return np.argmin(t_diff), np.min(t_diff)


    # manual_counts, manual_std, manual_times
    def scale_by_manual_calibration_idx(self, manual_count, image_counts, closest_idx):
        """ determine scale factor for image_counts based on manual_counts and calibration_idx """
        
        # added due to some potential calibration times lining up with "night" conditions
        idx_select = closest_idx + self.calibration_window_shift
        
        # find the idx for the nearest time to the specified calibration manual time
        # accounting for min/max sizes of image_counts
        idx_min = []
        idx_max = []
        idx_min = int(idx_select - self.calibration_window_size/2)
        if idx_min < 0:
            idx_min = int(0)
            if len(image_counts) <= self.calibration_window_size:
                idx_max = int(len(image_counts)-1)
            else:
                idx_max = int(self.calibration_window_size)
        else:
            idx_max = int(idx_min + self.calibration_window_size)
        if idx_max >= len(image_counts):
            idx_max = int(len(image_counts)-1)
        image_count_window = image_counts[idx_min:idx_max]
        # compute the scale factor over specified average (based on aggregate size?)
        image_sample_average = np.mean(image_count_window)
        # TODO handle the divide by zero case 
        if image_sample_average == 0:
            scale_factor = 1
        else:
            scale_factor = manual_count / image_sample_average
        print(f'calibration scale factor = {scale_factor}')
        return scale_factor, idx_select


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
        SHOW=False):
        """
        Plot AI detections and manual counts, highlighting invalid points with red, or having them removed and interpolated with SHOW_INVALID_POINTS.
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
        output_path = os.path.join(self.save_det_dir, f'Combined_tank_counts_{self.tank_sheet_name}_{self.cslics_uuid}_{plot_label}.png')
        plt.savefig(output_path, dpi=600)
        print(f'Plot saved to {output_path}')
        if SHOW:
            plt.show()


    def process_and_scale_counts(self, image_counts, image_std, image_times, manual_counts, manual_std, manual_times):
        """
        Process and scale image counts using focus volume and manual calibration.

        :param image_counts: Array of image counts.
        :param image_std: Array of image standard deviations.
        :param image_times: Array of image times (in decimal days).
        :param manual_counts: Array of manual counts.
        :param manual_std: Array of manual standard deviations.
        :param manual_times: Array of manual times (in decimal days).
        :return: Tuple containing scaled counts and standard deviations for both focus volume and manual calibration.
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

        # Save the plot
        output_path = os.path.join(self.save_det_dir, f'Error_plot_{self.tank_sheet_name}.png')
        plt.savefig(output_path, dpi=600)
        plt.show()
        print(f"Error plot saved to {output_path}")
        print(f"Mean Absolute Error: {mean_abs_error:.2f}, Root Mean Square Error: {rmse:.2f}")
        
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
        
    def run(self):
        """
        Run the full processing and plotting pipeline.
        """
        # Read manual counts
        print(f'Reading manual counts from {self.manual_counts_file}...')
        manual_counts, manual_std, manual_times, manual_times_dt = self.read_manual_counts()
        # Plot manual counts
        self.plot_manual_counts(manual_counts, manual_std, manual_times)

        # First, read model detections from all JSON files
        print(f'Reading detections from {self.save_det_dir}...')
        nearest_day = manual_times_dt[0].replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        samples = self.read_detections()
        
        # read valid/invalid times from JSON, then re-run batch_detections to get a new set of image_counts
        print(f'Filtering invalid times...')
        image_exclusion_data = self.read_invalid_times()
        image_names = [Path(sample).stem for sample in sorted(Path(self.save_det_dir).rglob('*_det.json'))]
        if not image_exclusion_data or 'invalid_ranges' not in image_exclusion_data:
            print("No invalid ranges provided. Proceeding with all detection names as valid.")
            invalid_names, invalid_indices, valid_names, valid_indices = [], [], [
                Path(sample).stem for sample in samples
            ], list(range(len(samples)))
        else:
            image_names = [Path(sample).stem for sample in sorted(Path(self.save_det_dir).rglob('*_det.json'))]
            invalid_names, invalid_indices, valid_names, valid_indices = self.filter_invalid_times(
                image_names, image_exclusion_data['invalid_ranges']
            )
        # Filter samples to include only valid ones
        samples_valid = [sample for sample in samples if Path(sample).stem in valid_names]
        
        # Batch detections with invalid indices
        print(f'Batching detections...')
        image_counts, image_std, image_times, batched_invalid_indices = self.batch_detections(samples, nearest_day, invalid_indices)

        # Process and scale counts for all images
        print(f'Processing and scaling counts for all images...')
        (tank_counts_def, tank_std_def), (tank_counts_cal, tank_std_cal), scaling_idx = self.process_and_scale_counts(
            image_counts, image_std, image_times, manual_counts, manual_std, manual_times
        )

        # Plot detections and manual counts for all images
        print(f'Plotting detections and manual counts for all images...')
        self.plot_detections_and_manual_counts(
            image_times=image_times,
            tank_counts_def=tank_counts_def,
            tank_std_def=tank_std_def,
            tank_counts_cal=tank_counts_cal,
            tank_std_cal=tank_std_cal,
            manual_times=manual_times,
            manual_counts=manual_counts,
            manual_std=manual_std,
            scaling_idx=scaling_idx,
            batched_invalid_indices=batched_invalid_indices,
            plot_label="All_Images"
        )


        # plot error:
        print(f'Plotting error between manual and AI counts...')
        manual_times, errors = self.plot_error_between_manual_and_ai(
        image_times=image_times,
        tank_counts_cal=tank_counts_cal,
        manual_times=manual_times,
        manual_counts=manual_counts,
        batched_invalid_indices=batched_invalid_indices
        )
        
        return (
            (tank_counts_def, tank_std_def),
            (tank_counts_cal, tank_std_cal),
            (manual_counts, manual_std),
            manual_times_dt,
        )

    def process_invalid_points(self, image_times, tank_counts, tank_std, batched_invalid_indices):
        """
        Process invalid points by either interpolating or removing them.

        Args:
            image_times: Array of image times (in decimal days).
            tank_counts: Array of tank counts.
            tank_std: Array of standard deviations.
            batched_invalid_indices: List of lists containing invalid indices for each batch.

        Returns:
            Tuple containing processed (image_times, tank_counts, tank_std, interpolated_mask).
            The interpolated_mask is a boolean array where True indicates an interpolated point.
        """
        # Create a mask for valid time points (all True initially)
        valid_mask = np.ones(len(image_times), dtype=bool)
        
        # Get all batch indices that contain invalid points
        invalid_batch_indices = [batch_idx for batch_idx, invalid_idx_list in enumerate(batched_invalid_indices) 
                                if invalid_idx_list]
        
        # If no invalid points, return original data with all False interpolation mask
        if not invalid_batch_indices:
            return image_times, tank_counts, tank_std, np.zeros(len(image_times), dtype=bool)
        
        # Mark invalid points in the mask
        valid_mask[invalid_batch_indices] = False
        
        # Create a mask to track interpolated points (initially all False)
        interpolated_mask = np.zeros(len(image_times), dtype=bool)
        
        # If there are no valid points, return empty arrays
        if not np.any(valid_mask):
            return np.array([]), np.array([]), np.array([]), np.array([])
        
        # Get arrays with only valid points
        valid_times = image_times[valid_mask]
        valid_counts = tank_counts[valid_mask]
        valid_std = tank_std[valid_mask]
        
        # If we have enough valid points to interpolate (at least 2)
        if len(valid_times) >= 2:
            # Interpolate tank counts and standard deviations
            interpolated_counts = np.interp(image_times, valid_times, valid_counts)
            interpolated_std = np.interp(image_times, valid_times, valid_std)
            
            # Only use interpolated values for invalid points
            processed_counts = np.where(valid_mask, tank_counts, interpolated_counts)
            processed_std = np.where(valid_mask, tank_std, interpolated_std)
            
            # Mark which points were interpolated
            interpolated_mask = ~valid_mask
            
            return image_times, processed_counts, processed_std, interpolated_mask
        else:
            # Not enough points to interpolate, just return valid points
            return valid_times, valid_counts, valid_std, np.zeros(len(valid_times), dtype=bool)


# Example usage:
if __name__ == "__main__":
    
      

    # config = {
    #     'manual_counts_file': '/home/dtsai/Data/cslics_datasets/manual_counts/cslics_2024_manual_counts.xlsx',
    #     'spawning_sheet_name': '2024 oct',
    #     'tank_sheet_name': 'OCT24 T2 Amag',
    #     'cslics_associations_file': '/home/dtsai/Data/cslics_datasets/manual_counts/cslics_2024_spawning_setup.xlsx',
    #     'model_name': 'cslics_subsurface_20250205_640p_yolov8n',
    #     'base_detection_dir': '/media/dtsai/CSLICSOct24/cslics_october_2024/detections',
    #     'save_manual_plot_dir': '/home/dtsai/Data/cslics_datasets/manual_counts/plots',
    #     'invalid_ranges_file': '/home/dtsai/Data/cslics_datasets/manual_counts/invalid_image_times/cslics_2024_oct_100000009c23b5af.json',
    #     'skipping_frequency': 1,
    #     'aggregate_size': 100,
    #     'confidence_threshold': 0.5,
    #     'MAX_SAMPLE': 1000,
    #     'calibration_window_size': 1,
    #     'calibration_idx': 1,
    #     'calibration_window_shift': 10,
    #     'PLOT_FOCUS_VOLUME': False
    # }
    
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202410_t1_amag_100000000029da9b.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202410_t2_amag_100000009c23b5af.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202410_t3_amag_10000000f620da42.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202410_t4_maeq_100000001ab0438d.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202410_t5_maeq_100000000846a7ff.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202410_t6_aant_10000000570f9d9c.json"
    
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202411_t1_amil_100000000029da9b.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202411_t2_amil_100000009c23b5af.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202411_t3_amil_10000000f620da42.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202411_t4_pdae_100000001ab0438d.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202411_t5_pdae_100000000846a7ff.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202411_t6_lcor_10000000570f9d9c.json"
    
    config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202312_t4_alor_cslics01.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202312_t4_alor_cslics08.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/config_202312_t4_alor_cslics09.json"
    
    processor = CSLICSDataProcessor(config_file)
    # (tank_counts_def, tank_std_def), (tank_counts_cal, tank_std_cal), (manual_counts, manual_std), manual_times_dt
    results = processor.run()
    print(f'cslics uuid: {processor.cslics_uuid}, coral species: {processor.coral_species}')
    
    import code
    code.interact(local=dict(globals(), **locals()))

   
    print('Done.')