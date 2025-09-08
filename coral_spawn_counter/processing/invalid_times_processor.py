import json
import numpy as np
from pathlib import Path
import sys

# Add the parent directory to the path to import the config manager
sys.path.append(str(Path(__file__).parent.parent))
from config.plot_config_manager import PlotConfigManager, CoralSpawnConfig


class InvalidTimesProcessor:
    """Class for handling invalid time ranges and filtering detection files."""
    
    def __init__(self, config_manager: PlotConfigManager = None, invalid_times_file: str = None):
        """
        Initialize the InvalidTimesProcessor.
        
        Args:
            config_manager: PlotConfigManager instance with loaded configuration
            invalid_times_file: Direct path to invalid times file (optional, overrides config)
        """
        if config_manager:
            self.config_manager = config_manager
            self.config = config_manager.get_config()
            self.cslics_invalid_times_file = invalid_times_file or self.config.invalid_ranges_file
        else:
            self.config_manager = None
            self.config = None
            self.cslics_invalid_times_file = invalid_times_file
            
        if not self.cslics_invalid_times_file:
            raise ValueError("Either config_manager or invalid_times_file must be provided")
            
        self.invalid_ranges = None
        
    @classmethod
    def from_config_file(cls, config_path: str):
        """
        Create InvalidTimesProcessor from configuration file.
        
        Args:
            config_path: Path to JSON configuration file
            
        Returns:
            InvalidTimesProcessor instance
        """
        config_manager = PlotConfigManager(config_path)
        return cls(config_manager)
    
    @classmethod
    def from_invalid_times_file(cls, invalid_times_file: str):
        """
        Create InvalidTimesProcessor from invalid times file path only.
        
        Args:
            invalid_times_file: Path to invalid times JSON file
            
        Returns:
            InvalidTimesProcessor instance
        """
        return cls(invalid_times_file=invalid_times_file)

    def read_invalid_times(self):
        """
        Reads a JSON file containing image exclusion ranges and returns it as a dictionary.
        NOTE: the image names should actually be the detection json files for compatibility with read_detections()

        Returns:
            dict: Dictionary with the JSON contents, or None if error
        """
        try:
            with open(self.cslics_invalid_times_file, "r") as file:
                data = json.load(file)
            self.invalid_ranges = data
            print(f"Loaded {len(data) if isinstance(data, list) else 'invalid'} time ranges from {self.cslics_invalid_times_file}")
            return data
        except FileNotFoundError:
            print(f"Invalid times file not found: {self.cslics_invalid_times_file}")
            self.invalid_ranges = []
            return []
        except Exception as e:
            print(f"Error reading JSON file: {e}")
            self.invalid_ranges = None
            return None

    def filter_invalid_times(self, detection_names, invalid_ranges=None):
        """
        Filter out invalid image names from the list of image detection names (json files).

        Args:
            detection_names: List of image filenames, chronologically ordered.
            invalid_ranges: List of dictionaries with 'start' and 'end' keys specifying invalid ranges.
                          If None, will use ranges from read_invalid_times().

        Returns:
            Tuple containing four lists:
                - invalid_names: List of invalid image filenames.
                - invalid_indices: List of indices of invalid image filenames.
                - valid_names: List of valid image filenames.
                - valid_indices: List of indices of valid image filenames.
        """
        # Use provided invalid_ranges or load from file
        if invalid_ranges is None:
            if self.invalid_ranges is None:
                invalid_ranges = self.read_invalid_times()
            else:
                invalid_ranges = self.invalid_ranges
        
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

        # Get indices for invalid and valid names
        invalid_indices = [detection_names.index(name) for name in invalid_names if name in detection_names]
        valid_indices = [detection_names.index(name) for name in valid_names if name in detection_names]
        
        print(f"Filtered {len(invalid_names)} invalid detection files out of {len(detection_names)} total")
        
        return invalid_names, invalid_indices, valid_names, valid_indices

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
            print("Warning: No valid points found for processing")
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
            
            print(f"Interpolated {np.sum(interpolated_mask)} invalid points")
            
            return image_times, processed_counts, processed_std, interpolated_mask
        else:
            # Not enough points to interpolate, just return valid points
            print(f"Not enough valid points for interpolation. Returning {len(valid_times)} valid points.")
            return valid_times, valid_counts, valid_std, np.zeros(len(valid_times), dtype=bool)
    
    def get_invalid_indices_from_detection_files(self, detection_files):
        """
        Get invalid indices from a list of detection file paths.
        
        Args:
            detection_files: List of Path objects or strings representing detection files
            
        Returns:
            list: List of indices corresponding to invalid files
        """
        # Convert to list of filenames (stems without extensions)
        detection_names = []
        for file_path in detection_files:
            if isinstance(file_path, Path):
                name = file_path.stem
            else:
                name = Path(file_path).stem
            detection_names.append(name)
        
        # Filter invalid times
        invalid_names, invalid_indices, valid_names, valid_indices = self.filter_invalid_times(detection_names)
        
        return invalid_indices
    
    def create_batched_invalid_indices(self, detection_files, batch_size, skipping_frequency=1):
        """
        Create batched invalid indices for use with detection batching.
        
        Args:
            detection_files: List of detection file paths
            batch_size: Size of each batch (aggregate_size)
            skipping_frequency: How many files to skip between samples
            
        Returns:
            list: List of lists containing invalid indices for each batch
        """
        # Get invalid indices for the full file list
        invalid_indices = self.get_invalid_indices_from_detection_files(detection_files)
        
        # Apply skipping frequency to match the downsampling
        downsampled_invalid_indices = []
        for idx in invalid_indices:
            downsampled_idx = idx // skipping_frequency
            if idx % skipping_frequency == 0:  # Only include if it would be in the downsampled list
                downsampled_invalid_indices.append(downsampled_idx)
        
        # Calculate number of batches
        downsampled_length = len(detection_files) // skipping_frequency
        num_batches = (downsampled_length + batch_size - 1) // batch_size  # Ceiling division
        
        # Create batched invalid indices
        batched_invalid_indices = []
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, downsampled_length)
            
            # Find invalid indices within this batch
            batch_invalid = []
            for invalid_idx in downsampled_invalid_indices:
                if batch_start <= invalid_idx < batch_end:
                    # Convert to within-batch index
                    within_batch_idx = invalid_idx - batch_start
                    batch_invalid.append(within_batch_idx)
            
            batched_invalid_indices.append(batch_invalid)
        
        return batched_invalid_indices
    
    def save_invalid_ranges(self, invalid_ranges, output_path=None):
        """
        Save invalid time ranges to JSON file.
        
        Args:
            invalid_ranges: List of invalid time range dictionaries
            output_path: Optional custom output path (uses config path if None)
        """
        if output_path is None:
            output_path = self.cslics_invalid_times_file
            
        try:
            with open(output_path, 'w') as f:
                json.dump(invalid_ranges, f, indent=4)
            print(f"Saved {len(invalid_ranges)} invalid time ranges to {output_path}")
            self.invalid_ranges = invalid_ranges
        except Exception as e:
            print(f"Error saving invalid ranges: {e}")
    
    def add_invalid_range(self, start_name, end_name, reason="", description=""):
        """
        Add a new invalid time range.
        
        Args:
            start_name: Start image name
            end_name: End image name  
            reason: Reason for invalidity
            description: Additional description
        """
        if self.invalid_ranges is None:
            self.read_invalid_times()
        
        new_range = {
            "start": start_name,
            "end": end_name,
            "reason": reason,
            "description": description
        }
        
        if self.invalid_ranges is None:
            self.invalid_ranges = []
            
        self.invalid_ranges.append(new_range)
        print(f"Added invalid range: {start_name} to {end_name}")
    
    def get_summary(self):
        """
        Get a summary of the invalid times configuration.
        
        Returns:
            str: Summary string
        """
        if self.invalid_ranges is None:
            self.read_invalid_times()
            
        summary = f"Invalid Times Processor Summary:\n"
        summary += f"  File: {self.cslics_invalid_times_file}\n"
        
        if self.invalid_ranges:
            summary += f"  Number of invalid ranges: {len(self.invalid_ranges)}\n"
            for i, range_info in enumerate(self.invalid_ranges):
                summary += f"    {i+1}. {range_info.get('start', 'N/A')} to {range_info.get('end', 'N/A')}"
                if range_info.get('reason'):
                    summary += f" (Reason: {range_info['reason']})"
                summary += "\n"
        else:
            summary += "  No invalid ranges defined.\n"
            
        return summary


if __name__ == "__main__":
    # Example usage
    config_path = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plot_config_202312_t4_alor_cslics08.json"
    
    try:
        # Initialize from config file
        processor = InvalidTimesProcessor.from_config_file(config_path)
        
        # Print summary
        print(processor.get_summary())
        
        # Example: process some detection files
        detection_files = [f"detection_{i:04d}_det.json" for i in range(100)]
        invalid_indices = processor.get_invalid_indices_from_detection_files(detection_files)
        
        print(f"Found {len(invalid_indices)} invalid files")
        
    except Exception as e:
        print(f"Error: {e}")

