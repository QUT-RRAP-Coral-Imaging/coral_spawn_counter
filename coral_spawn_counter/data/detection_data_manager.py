# data/detection_data_manager.py
import os
import json
import time
import fcntl
import psutil  # You'll need to add this dependency
from pathlib import Path
from datetime import datetime


class DetectionDataManager:
    """Manages detection data storage and retrieval for plotting and analysis."""
    
    def __init__(self, config):
        """
        Initialize the detection data manager.
        
        Args:
            config: ConfigManager instance with configuration parameters
        """
        self.config = config
        self.cslics_uuid = config.cslics_uuid
        self.save_dir = config.save_dir
        self.mode = config.mode
        self.submersion_time = config.submersion_time
        self.surface_model_id = config.surface_model_id
        self.subsurface_model_id = config.subsurface_model_id
        
        # Initialize detection data file paths
        self.stats_dir, self.surface_data_path, self.subsurface_data_path = self._get_detection_data_paths()
        
        # Initialize detection data files if they don't exist
        self._initialize_detection_files()
    
    def _get_detection_data_paths(self):
        """Get paths for detection data JSON files."""
        # Create a directory for stats/metadata
        stats_dir = os.path.join(self.save_dir, self.cslics_uuid, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        # Define paths for surface and subsurface detection data
        surface_data_path = os.path.join(stats_dir, f"{self.surface_model_id}_detection_data.json")
        subsurface_data_path = os.path.join(stats_dir, f"{self.subsurface_model_id}_detection_data.json")
        
        return stats_dir, surface_data_path, subsurface_data_path
    
    def _initialize_detection_files(self):
        """Initialize JSON files for storing detection data."""
        # Initialize surface data file if needed
        if self.mode in ["surface", "both"]:
            if not os.path.exists(self.surface_data_path):
                surface_data = {
                    "cslics_uuid": self.cslics_uuid,
                    "model_id": self.surface_model_id,
                    "detection_type": "surface",
                    "submersion_time": self.submersion_time,
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_detections": 0,
                    "total_files_processed": 0,
                    "detections_by_timestamp": {}
                }
                self._write_detection_file(self.surface_data_path, surface_data)
                print(f"Initialized surface detection data file: {self.surface_data_path}")
        
        # Initialize subsurface data file if needed
        if self.mode in ["subsurface", "both"]:
            if not os.path.exists(self.subsurface_data_path):
                subsurface_data = {
                    "cslics_uuid": self.cslics_uuid,
                    "model_id": self.subsurface_model_id,
                    "detection_type": "subsurface",
                    "submersion_time": self.submersion_time,
                    "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "last_updated": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    "total_detections": 0,
                    "total_files_processed": 0,
                    "detections_by_timestamp": {}
                }
                self._write_detection_file(self.subsurface_data_path, subsurface_data)
                print(f"Initialized subsurface detection data file: {self.subsurface_data_path}")
    
    def _write_detection_file(self, file_path, data):
        """Write detection data to file with proper formatting."""
        try:
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2, separators=(',', ': '))
        except Exception as e:
            print(f"Error writing detection file {file_path}: {e}")
            raise
    
    def _acquire_lock_with_cleanup(self, lock_path, timeout=30):
        """
        Acquire lock with automatic cleanup of stale locks.
        
        Args:
            lock_path: Path to the lock file
            timeout: Maximum time to wait for lock (seconds)
            
        Returns:
            file object or None if lock couldn't be acquired
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if lock file exists and if it's stale
                if os.path.exists(lock_path):
                    if self._is_stale_lock(lock_path):
                        print(f"Removing stale lock file: {lock_path}")
                        try:
                            os.remove(lock_path)
                        except OSError:
                            pass  # File might have been removed by another process
                
                # Try to acquire lock
                lock_file = open(lock_path, 'w')
                try:
                    # Try non-blocking lock first
                    fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    # Write current process ID to lock file
                    lock_file.write(str(os.getpid()))
                    lock_file.flush()
                    return lock_file
                except BlockingIOError:
                    # Lock is held by another process
                    lock_file.close()
                    time.sleep(0.1)  # Wait a bit before retrying
                    continue
                    
            except Exception as e:
                print(f"Error acquiring lock {lock_path}: {e}")
                time.sleep(0.1)
                continue
        
        print(f"Failed to acquire lock {lock_path} within {timeout} seconds")
        return None
    
    def _is_stale_lock(self, lock_path, max_age_seconds=300):
        """
        Check if a lock file is stale (old or from dead process).
        
        Args:
            lock_path: Path to the lock file
            max_age_seconds: Maximum age before considering lock stale
            
        Returns:
            bool: True if lock is stale
        """
        try:
            # Check age of lock file
            lock_age = time.time() - os.path.getmtime(lock_path)
            if lock_age > max_age_seconds:
                return True
            
            # Check if process that created lock is still alive
            try:
                with open(lock_path, 'r') as f:
                    pid_str = f.read().strip()
                    if pid_str.isdigit():
                        pid = int(pid_str)
                        if not psutil.pid_exists(pid):
                            return True
            except (FileNotFoundError, ValueError, OSError):
                # Can't read PID or invalid format
                return True
            
            return False
            
        except OSError:
            # Can't access lock file
            return True
    
    def update_detection_data(self, timestamp_str, count, model_type):
        """
        Update detection data in JSON file with file locking to prevent race conditions.
        
        Args:
            timestamp_str: Timestamp string (ISO format)
            count: Number of detections
            model_type: Type of detection ("surface" or "subsurface")
        """
        if count <= 0:
            return  # Skip empty detections
        
        data_path = self.surface_data_path if model_type == "surface" else self.subsurface_data_path
        
        # Skip if we're not processing this type
        if (model_type == "surface" and self.mode == "subsurface") or \
           (model_type == "subsurface" and self.mode == "surface"):
            return
        
        lock_path = f"{data_path}.lock"
        
        # Acquire lock with automatic cleanup
        lock_file = self._acquire_lock_with_cleanup(lock_path)
        if lock_file is None:
            print(f"Could not acquire lock for {data_path}, skipping update")
            return
        
        try:
            # Load current data
            if os.path.exists(data_path):
                with open(data_path, 'r') as f:
                    data = json.load(f)
            else:
                self._initialize_detection_files()
                with open(data_path, 'r') as f:
                    data = json.load(f)
            
            # Update the data
            data["total_detections"] += count
            data["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            if timestamp_str in data["detections_by_timestamp"]:
                data["detections_by_timestamp"][timestamp_str]["count"] += count
                data["detections_by_timestamp"][timestamp_str]["files"] += 1
            else:
                data["detections_by_timestamp"][timestamp_str] = {
                    "count": count,
                    "files": 1
                }
                data["total_files_processed"] += 1
            
            # Save updated data
            self._write_detection_file(data_path, data)
            
        except Exception as e:
            print(f"Error updating detection data file {data_path}: {e}")
        finally:
            # Always clean up lock
            try:
                lock_file.close()
                os.remove(lock_path)
            except OSError:
                pass
    
    def batch_update_detection_data(self, detections_dict, total_count, model_type):
        """
        Update detection data in JSON file with batched updates for better performance.
        
        Args:
            detections_dict: Dictionary of timestamp -> {count, files}
            total_count: Total detections in this batch
            model_type: Type of detection ("surface" or "subsurface")
        """
        if total_count <= 0 or not detections_dict:
            return  # Skip if no detections
        
        data_path = self.surface_data_path if model_type == "surface" else self.subsurface_data_path
        
        # Skip if we're not processing this type
        if (model_type == "surface" and self.mode == "subsurface") or \
           (model_type == "subsurface" and self.mode == "surface"):
            return
        
        # Use a lock file to prevent concurrent access
        lock_path = f"{data_path}.lock"
        
        try:
            with open(lock_path, 'w') as lock_file:
                fcntl.flock(lock_file, fcntl.LOCK_EX)
                
                try:
                    # Load current data
                    if os.path.exists(data_path):
                        with open(data_path, 'r') as f:
                            data = json.load(f)
                    else:
                        # Initialize if needed
                        self._initialize_detection_files()
                        with open(data_path, 'r') as f:
                            data = json.load(f)
                    
                    # Update with batch data
                    data["total_detections"] += total_count
                    data["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    
                    new_files_processed = 0
                    for timestamp_str, counts in detections_dict.items():
                        if timestamp_str in data["detections_by_timestamp"]:
                            data["detections_by_timestamp"][timestamp_str]["count"] += counts["count"]
                            data["detections_by_timestamp"][timestamp_str]["files"] += counts["files"]
                        else:
                            data["detections_by_timestamp"][timestamp_str] = counts
                            new_files_processed += counts["files"]
                    
                    data["total_files_processed"] += new_files_processed
                    
                    # Save updated data
                    self._write_detection_file(data_path, data)
                    
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    print(f"Error during batch update: {e}")
                    # Initialize and try again
                    self._initialize_detection_files()
                    self.batch_update_detection_data(detections_dict, total_count, model_type)
                
        except Exception as e:
            print(f"Error with file locking when batch updating {data_path}: {e}")
        finally:
            # Clean up lock file
            try:
                if os.path.exists(lock_path):
                    os.remove(lock_path)
            except OSError:
                pass  # Ignore cleanup errors
    
    def load_detection_data(self, model_type):
        """
        Load detection data from JSON file for a specific model type.
        
        Args:
            model_type: Either "surface" or "subsurface"
            
        Returns:
            dict or None: Detection data dictionary or None if file doesn't exist
        """
        data_path = self.surface_data_path if model_type == "surface" else self.subsurface_data_path
        
        if not os.path.exists(data_path):
            print(f"No {model_type} detection data file found at {data_path}")
            return None
        
        try:
            with open(data_path, 'r') as f:
                data = json.load(f)
            return data
        except json.JSONDecodeError as e:
            print(f"Error reading {model_type} detection data file: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error loading {model_type} detection data: {e}")
            return None
    
    def get_detection_summary(self, model_type=None):
        """
        Get a summary of detection data.
        
        Args:
            model_type: Either "surface", "subsurface", or None for both
            
        Returns:
            dict: Summary statistics
        """
        summary = {}
        
        # Get surface data if requested
        if model_type in [None, "surface"] and self.mode in ["surface", "both"]:
            surface_data = self.load_detection_data("surface")
            if surface_data:
                summary["surface"] = {
                    "total_detections": surface_data.get("total_detections", 0),
                    "total_files_processed": surface_data.get("total_files_processed", 0),
                    "unique_timestamps": len(surface_data.get("detections_by_timestamp", {})),
                    "last_updated": surface_data.get("last_updated", "Unknown"),
                    "model_id": surface_data.get("model_id", "Unknown")
                }
        
        # Get subsurface data if requested
        if model_type in [None, "subsurface"] and self.mode in ["subsurface", "both"]:
            subsurface_data = self.load_detection_data("subsurface")
            if subsurface_data:
                summary["subsurface"] = {
                    "total_detections": subsurface_data.get("total_detections", 0),
                    "total_files_processed": subsurface_data.get("total_files_processed", 0),
                    "unique_timestamps": len(subsurface_data.get("detections_by_timestamp", {})),
                    "last_updated": subsurface_data.get("last_updated", "Unknown"),
                    "model_id": subsurface_data.get("model_id", "Unknown")
                }
        
        return summary
    
    def print_detection_summary(self):
        """Print a summary of detection data."""
        summary = self.get_detection_summary()
        
        print("\nDetection Data Summary:")
        print(f"CSLICS UUID: {self.cslics_uuid}")
        print(f"Data directory: {self.stats_dir}")
        
        for model_type, data in summary.items():
            print(f"\n{model_type.capitalize()} detections:")
            print(f"  Model ID: {data['model_id']}")
            print(f"  Total detections: {data['total_detections']}")
            print(f"  Files processed: {data['total_files_processed']}")
            print(f"  Unique timestamps: {data['unique_timestamps']}")
            print(f"  Last updated: {data['last_updated']}")
    
    def export_detection_data(self, output_dir=None, format='json'):
        """
        Export detection data to external files.
        
        Args:
            output_dir: Directory to save exported files (default: stats directory)
            format: Export format ('json' or 'csv')
        """
        if output_dir is None:
            output_dir = self.stats_dir
        
        os.makedirs(output_dir, exist_ok=True)
        
        for model_type in ["surface", "subsurface"]:
            if self.mode in [model_type, "both"]:
                data = self.load_detection_data(model_type)
                if data is None:
                    continue
                
                if format.lower() == 'json':
                    export_path = os.path.join(output_dir, f"{self.cslics_uuid}_{model_type}_detections_export.json")
                    with open(export_path, 'w') as f:
                        json.dump(data, f, indent=2)
                    print(f"Exported {model_type} detection data to: {export_path}")
                
                elif format.lower() == 'csv':
                    export_path = os.path.join(output_dir, f"{self.cslics_uuid}_{model_type}_detections_export.csv")
                    self._export_to_csv(data, export_path)
                    print(f"Exported {model_type} detection data to: {export_path}")
    
    def _export_to_csv(self, data, output_path):
        """Export detection data to CSV format."""
        import csv
        
        with open(output_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            
            # Write header
            writer.writerow(['timestamp', 'detection_count', 'files_count'])
            
            # Write data
            for timestamp_str, counts in data.get("detections_by_timestamp", {}).items():
                writer.writerow([timestamp_str, counts["count"], counts["files"]])
    
    def validate_detection_files(self):
        """
        Validate the integrity of detection data files.
        
        Returns:
            dict: Validation results
        """
        validation_results = {}
        
        for model_type in ["surface", "subsurface"]:
            if self.mode in [model_type, "both"]:
                data_path = self.surface_data_path if model_type == "surface" else self.subsurface_data_path
                
                validation_results[model_type] = {
                    "file_exists": os.path.exists(data_path),
                    "file_readable": False,
                    "valid_json": False,
                    "has_required_fields": False,
                    "consistent_totals": False
                }
                
                if validation_results[model_type]["file_exists"]:
                    try:
                        with open(data_path, 'r') as f:
                            data = json.load(f)
                        
                        validation_results[model_type]["file_readable"] = True
                        validation_results[model_type]["valid_json"] = True
                        
                        # Check required fields
                        required_fields = ["cslics_uuid", "model_id", "total_detections", "detections_by_timestamp"]
                        validation_results[model_type]["has_required_fields"] = all(
                            field in data for field in required_fields
                        )
                        
                        # Check consistency of totals
                        calculated_total = sum(
                            counts["count"] for counts in data.get("detections_by_timestamp", {}).values()
                        )
                        validation_results[model_type]["consistent_totals"] = (
                            calculated_total == data.get("total_detections", 0)
                        )
                        
                    except json.JSONDecodeError:
                        validation_results[model_type]["file_readable"] = True
                        validation_results[model_type]["valid_json"] = False
                    except Exception:
                        validation_results[model_type]["file_readable"] = False
        
        return validation_results
    
    def repair_detection_files(self):
        """Attempt to repair corrupted detection data files."""
        validation = self.validate_detection_files()
        
        for model_type, results in validation.items():
            if not results["file_exists"]:
                print(f"Re-initializing missing {model_type} detection file")
                self._initialize_detection_files()
            
            elif not results["valid_json"]:
                print(f"Repairing corrupted {model_type} detection file")
                # Create backup and re-initialize
                data_path = self.surface_data_path if model_type == "surface" else self.subsurface_data_path
                backup_path = f"{data_path}.corrupted_{int(time.time())}"
                try:
                    os.rename(data_path, backup_path)
                    print(f"Backed up corrupted file to: {backup_path}")
                except OSError:
                    pass
                self._initialize_detection_files()
            
            elif not results["consistent_totals"]:
                print(f"Fixing inconsistent totals in {model_type} detection file")
                self._fix_inconsistent_totals(model_type)
    
    def _fix_inconsistent_totals(self, model_type):
        """Fix inconsistent total counts in detection data."""
        data = self.load_detection_data(model_type)
        if data is None:
            return
        
        # Recalculate totals
        calculated_total = sum(
            counts["count"] for counts in data.get("detections_by_timestamp", {}).values()
        )
        calculated_files = sum(
            counts["files"] for counts in data.get("detections_by_timestamp", {}).values()
        )
        
        # Update totals
        data["total_detections"] = calculated_total
        data["total_files_processed"] = calculated_files
        data["last_updated"] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Save corrected data
        data_path = self.surface_data_path if model_type == "surface" else self.subsurface_data_path
        self._write_detection_file(data_path, data)
        
        print(f"Fixed {model_type} totals: {calculated_total} detections, {calculated_files} files")
    
    def cleanup_old_lock_files(self):
        """Clean up any old lock files that may have been left behind."""
        for data_path in [self.surface_data_path, self.subsurface_data_path]:
            lock_path = f"{data_path}.lock"
            if os.path.exists(lock_path):
                try:
                    # Check if lock file is old (more than 1 hour)
                    lock_age = time.time() - os.path.getmtime(lock_path)
                    if lock_age > 3600:  # 1 hour
                        os.remove(lock_path)
                        print(f"Removed old lock file: {lock_path}")
                except OSError:
                    pass  # Ignore errors