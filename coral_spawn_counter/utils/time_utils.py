# utils/time_utils.py
import re
from datetime import datetime
from pathlib import Path


class TimeUtils:
    """Utilities for time-related operations."""
    
    def __init__(self, submersion_datetime):
        self.submersion_datetime = submersion_datetime
        self._timestamp_cache = {}

    def _parse_datetime_from_stem(self, filename_stem):
        """Parse datetime from different known filename formats."""
        candidates = []

        # Legacy format slice used in historical datasets
        if len(filename_stem) > 20:
            candidates.append((filename_stem[9:-11], "%Y%m%d_%H%M%S"))

        # 2025+ format e.g. 2025-12-17_07-20-42+475141_clean
        candidates.append((filename_stem.split('+')[0], "%Y-%m-%d_%H-%M-%S"))

        # Regex fallback for compact timestamp format
        compact_match = re.search(r"(\d{8}_\d{6})", filename_stem)
        if compact_match:
            candidates.append((compact_match.group(1), "%Y%m%d_%H%M%S"))

        # Regex fallback for dashed timestamp format
        dashed_match = re.search(r"(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})", filename_stem)
        if dashed_match:
            candidates.append((dashed_match.group(1), "%Y-%m-%d_%H-%M-%S"))

        for timestamp_str, date_format in candidates:
            try:
                return datetime.strptime(timestamp_str, date_format)
            except (ValueError, TypeError):
                continue

        return None
    
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
        image_datetime = self._parse_datetime_from_stem(filename)

        if image_datetime is None:
            # If datetime parsing fails, assume it's a surface image
            print(f"Warning: Could not parse datetime from filename: {filename}")
            self._timestamp_cache[img_path] = True
            return True

        result = image_datetime < self.submersion_datetime
        self._timestamp_cache[img_path] = result
        return result
    
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
        return self._parse_datetime_from_stem(filename)