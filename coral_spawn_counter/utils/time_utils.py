# utils/time_utils.py
from datetime import datetime
from pathlib import Path


class TimeUtils:
    """Utilities for time-related operations."""
    
    def __init__(self, submersion_datetime):
        self.submersion_datetime = submersion_datetime
        self._timestamp_cache = {}
    
    def is_surface_image(self, img_path):
        """
        Determine if an image was captured before submersion time with caching.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            bool: True if the image is a surface image, False otherwise
        """
        if img_path in self._timestamp_cache:
            return self._timestamp_cache[img_path]
            
        # Extract datetime from filename
        filename = Path(img_path).stem
        try:
            timestamp_str = filename[9:-11]  # Assumes consistent filename format
            image_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            result = image_datetime < self.submersion_datetime
            self._timestamp_cache[img_path] = result
            return result
        except (ValueError, IndexError):
            # If datetime parsing fails, assume it's a surface image
            print(f"Warning: Could not parse datetime from filename: {filename}")
            self._timestamp_cache[img_path] = True
            return True
    
    def convert_to_decimal_days(self, dates_list, time_zero=None):
        """
        Convert datetime objects to decimal days since time_zero.
        
        Args:
            dates_list: List of datetime objects
            time_zero: Reference time (if None, use submersion time)
            
        Returns:
            list: Decimal days since time_zero
        """
        if time_zero is None:
            time_zero = self.submersion_datetime
        
        decimal_days = []
        for dt in dates_list:
            delta = dt - time_zero
            days = delta.total_seconds() / (24 * 3600)
            decimal_days.append(days)
            
        return decimal_days
    
    def extract_timestamp_from_filename(self, img_path):
        """
        Extract timestamp from image filename.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            datetime or None: Extracted timestamp or None if parsing fails
        """
        filename = Path(img_path).stem
        try:
            timestamp_str = filename[9:-11]
            return datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
        except (ValueError, IndexError):
            return None