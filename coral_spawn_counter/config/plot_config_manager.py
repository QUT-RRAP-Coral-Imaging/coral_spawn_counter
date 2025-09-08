import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CoralSpawnConfig:
    """Configuration class for coral spawn counter."""
    
    # File paths
    manual_counts_file: str
    cslics_associations_file: str
    base_detection_dir: str
    surface_detection_dir: str
    subsurface_detection_dir: str
    save_manual_plot_dir: str
    invalid_ranges_file: str
    
    # Model weights paths
    surface_weights_path: str
    subsurface_weights_path: str
    
    # Sheet names
    spawning_sheet_name: str
    tank_sheet_name: str
    
    # Identifiers
    cslics_uuid: str
    coral_species: str
    submersion_time: str
    
    # Processing parameters
    mode: str  = "both" # "surface", "subsurface", or "both"
    verbose: bool = False
    image_skip: int = 1
    skipping_frequency: int = 1
    aggregate_size: int = 100
    confidence_threshold: float = 0.3
    MAX_SAMPLE: int = 2000
    
    # Surface-specific calibration parameters
    surface_calibration_idx: int = 0
    surface_calibration_window_size: int = 1
    surface_calibration_window_shift: int = 0
    
    # Subsurface-specific calibration parameters
    subsurface_calibration_idx: int = 0
    subsurface_calibration_window_size: int = 1
    subsurface_calibration_window_shift: int = 0

class PlotConfigManager:
    """Manager for loading and validating coral spawn counter plot configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the plot configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, must call load_config later.
        """
        self.config_path = config_path
        self.config: Optional[CoralSpawnConfig] = None
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> CoralSpawnConfig:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the JSON configuration file
            
        Returns:
            CoralSpawnConfig object
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required fields are missing or invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in configuration file: {e}")
        
        # Validate required fields
        required_fields = [
            'manual_counts_file', 'spawning_sheet_name', 'tank_sheet_name',
            'cslics_uuid', 'coral_species', 'cslics_associations_file',
            'surface_weights_path', 'subsurface_weights_path', 
            'base_detection_dir', 'surface_detection_dir', 'subsurface_detection_dir',
            'save_manual_plot_dir', 'invalid_ranges_file', 'submersion_time', 'mode',
            'verbose', 'skipping_frequency', 'aggregate_size', 'confidence_threshold', 'MAX_SAMPLE'
        ]
        
        missing_fields = [field for field in required_fields if field not in config_data]
        if missing_fields:
            raise ValueError(f"Missing required configuration fields: {missing_fields}")
        
        # Create config object
        self.config = CoralSpawnConfig(**config_data)
        self.config_path = str(config_path)
        
        # Validate paths
        self._validate_paths()
        
        return self.config
    
    def _validate_paths(self) -> None:
        """Validate that required file paths exist."""
        if not self.config:
            return
        
        # Check if required files exist
        files_to_check = [
            ('manual_counts_file', self.config.manual_counts_file),
            ('cslics_associations_file', self.config.cslics_associations_file),
            ('surface_weights_path', self.config.surface_weights_path),
            ('subsurface_weights_path', self.config.subsurface_weights_path),
        ]
        
        for field_name, file_path in files_to_check:
            if not os.path.exists(file_path):
                print(f"Warning: {field_name} file not found: {file_path}")
        
        # Check if directories exist, create if needed
        dirs_to_check = [
            ('base_detection_dir', self.config.base_detection_dir),
            ('surface_detection_dir', self.config.surface_detection_dir),
            ('subsurface_detection_dir', self.config.subsurface_detection_dir),
            ('save_manual_plot_dir', self.config.save_manual_plot_dir),
        ]
        
        for field_name, dir_path in dirs_to_check:
            if not os.path.exists(dir_path):
                print(f"Warning: {field_name} directory not found: {dir_path}")
                # Optionally create the directory for save_manual_plot_dir
                if field_name == 'save_manual_plot_dir':
                    os.makedirs(dir_path, exist_ok=True)
                    print(f"Created directory: {dir_path}")
    
    def get_config(self) -> CoralSpawnConfig:
        """
        Get the loaded configuration.
        
        Returns:
            CoralSpawnConfig object
            
        Raises:
            ValueError: If no configuration has been loaded
        """
        if self.config is None:
            raise ValueError("No configuration loaded. Call load_config() first.")
        return self.config
    
    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to JSON file.
        
        Args:
            output_path: Path where to save the configuration
        """
        if self.config is None:
            raise ValueError("No configuration to save. Load a configuration first.")
        
        # Convert dataclass to dict
        config_dict = {
            field.name: getattr(self.config, field.name)
            for field in self.config.__dataclass_fields__.values()
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_dict, f, indent=4)
    
    def update_config(self, **kwargs) -> None:
        """
        Update configuration parameters.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        if self.config is None:
            raise ValueError("No configuration loaded. Call load_config() first.")
        
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def get_detection_dir(self) -> str:
        """Get the full detection directory path for the current CSLICS UUID."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return os.path.join(self.config.base_detection_dir, self.config.cslics_uuid)
    
    def get_surface_detection_dir(self) -> str:
        """Get the surface detection directory path."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return self.config.surface_detection_dir
    
    def get_subsurface_detection_dir(self) -> str:
        """Get the subsurface detection directory path."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return self.config.subsurface_detection_dir
    
    def get_invalid_ranges_path(self) -> str:
        """Get the invalid ranges file path."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return self.config.invalid_ranges_file
    
    def get_surface_model_name(self) -> str:
        """Extract model name from surface weights path."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return Path(self.config.surface_weights_path).stem
    
    def get_subsurface_model_name(self) -> str:
        """Extract model name from subsurface weights path."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return Path(self.config.subsurface_weights_path).stem
    
    # Surface calibration parameter getters
    def get_surface_calibration_idx(self) -> int:
        """Get surface calibration index."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return self.config.surface_calibration_idx
    
    def get_surface_calibration_window_size(self) -> int:
        """Get surface calibration window size."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return self.config.surface_calibration_window_size
    
    def get_surface_calibration_window_shift(self) -> int:
        """Get surface calibration window shift."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return self.config.surface_calibration_window_shift
    
    # Subsurface calibration parameter getters
    def get_subsurface_calibration_idx(self) -> int:
        """Get subsurface calibration index."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return self.config.subsurface_calibration_idx
    
    def get_subsurface_calibration_window_size(self) -> int:
        """Get subsurface calibration window size."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return self.config.subsurface_calibration_window_size
    
    def get_subsurface_calibration_window_shift(self) -> int:
        """Get subsurface calibration window shift."""
        if self.config is None:
            raise ValueError("No configuration loaded.")
        
        return self.config.subsurface_calibration_window_shift
    
    def __str__(self) -> str:
        """String representation of the configuration."""
        if self.config is None:
            return "ConfigManager: No configuration loaded"
        
        return f"ConfigManager: {self.config.cslics_uuid} ({self.config.coral_species})"


# Example usage function
def load_config_from_data_dir(config_filename: str) -> PlotConfigManager:
    """
    Load configuration from the data_yaml_files directory.
    
    Args:
        config_filename: Name of the config file (e.g., 'plot_config_202312_t4_alor_cslics08.json')
        
    Returns:
        PlotConfigManager instance
    """
    # Assuming this script is in the same directory as data_yaml_files
    script_dir = Path(__file__).parent
    config_path = script_dir / "data_yaml_files" / config_filename
    
    return PlotConfigManager(str(config_path))


if __name__ == "__main__":
    # Example usage
    try:
        # Load the example configuration
        config_manager = PlotConfigManager("/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/plot_config_202312_t4_alor_cslics08.json")
        config = config_manager.get_config()
        
        print(f"Loaded configuration for: {config.cslics_uuid}")
        print(f"Coral species: {config.coral_species}")
        print(f"Surface Model: {config_manager.get_surface_model_name()}")
        print(f"Subsurface Model: {config_manager.get_subsurface_model_name()}")
        print(f"Surface Detection Dir: {config.surface_detection_dir}")
        print(f"Subsurface Detection Dir: {config.subsurface_detection_dir}")
        print(f"Confidence threshold: {config.confidence_threshold}")
        
        # Example of updating configuration
        config_manager.update_config(confidence_threshold=0.5, MAX_SAMPLE=1500)
        print(f"Updated confidence threshold: {config.confidence_threshold}")
        
    except Exception as e:
        print(f"Error: {e}")