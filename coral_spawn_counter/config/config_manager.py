# config/config_manager.py
import json
import os
from pathlib import Path
from datetime import datetime


class ConfigManager:
    """Handles configuration loading, validation, and parameter extraction."""
    
    def __init__(self, config_file):
        self.config_file = config_file
        self.config = self.load_config_from_json(config_file)
        self._extract_config_parameters()
        self._validate_params()
    
    @staticmethod
    def load_config_from_json(config_file):
        """Load configuration from a JSON file."""
        try:
            with open(config_file, 'r') as file:
                config = json.load(file)
            print(f"Configuration loaded successfully from {config_file}")
            return config
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: Configuration file {config_file} not found.")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error: Failed to parse JSON file {config_file}. Details: {e}")
    
    def _extract_config_parameters(self):
        """Extract configuration parameters from the loaded JSON config"""
        # Required parameters
        self.surface_weights_path = self.config['surface_weights_path']
        self.subsurface_weights_path = self.config['subsurface_weights_path']
        self.img_dir = self.config['img_dir']
        self.save_dir = self.config['save_dir']
        self.cslics_uuid = self.config['cslics_uuid']
        self.submersion_time = self.config['submersion_time']

        # Optional parameters with defaults
        self.mode = self.config.get('processing_mode', 'both')
        self.plot_only = self._parse_bool(self.config.get('plot_only', False))
        self.iou_thresh = float(self.config.get('iou_thresh', 0.3))
        self.conf_thresh = float(self.config.get('conf_thresh', 0.25))
        self.max_det = int(self.config.get('max_det', 1000))
        self.save_img = self._parse_bool(self.config.get('save_img', True))
        self.save_txt = self._parse_bool(self.config.get('save_txt', True))
        self.save_txt_bb = self._parse_bool(self.config.get('save_txt_bb', False))
        
        # Check image skip parameter
        self.image_skip = self.config.get('image_skip', 1)
        
        # Parse max_images more efficiently
        max_images_val = self.config.get('max_images')
        self.max_images = int(max_images_val) if max_images_val else None
        
        # Parse bool values
        self.verbose = self._parse_bool(self.config.get('verbose', False))
        self.parallel = self._parse_bool(self.config.get('parallel', False))
        
        # Resume capabilities
        self.resume = self._parse_bool(self.config.get('resume', False))
        self.resume_from_image = self.config.get('resume_image_name')
        
        # # If resume_image_name is specified, automatically enable resume mode
        # if self.resume_from_image and not self.resume:
        #     self.resume = True
        # print(f"Resume mode automatically enabled due to resume_image_name: {self.resume_from_image}")
    
        # Parse submersion time once
        self.submersion_datetime = datetime.strptime(self.submersion_time, "%Y-%m-%d_%H-%M-%S")
        
        # Get current time once
        self.current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Extract model IDs (stem names) 
        self.surface_model_id = Path(self.surface_weights_path).stem
        self.subsurface_model_id = Path(self.subsurface_weights_path).stem
        
        

    def _validate_params(self):
        """Validate required parameters"""
        required_params = {
            'surface_weights_path': self.surface_weights_path,
            'subsurface_weights_path': self.subsurface_weights_path,
            'img_dir': self.img_dir,
            'save_dir': self.save_dir,
            'submersion_time': self.submersion_time
        }
        
        for param_name, param_value in required_params.items():
            if not param_value:
                raise ValueError(f"{param_name} not specified")

    @staticmethod
    def _parse_bool(value):
        """Parse boolean values from various formats"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 't', 'y')
        return bool(value)
    
    def print_config_summary(self):
        """Print a complete summary of the configuration from the JSON file."""
        print("\n" + "="*60)
        print("CORAL SPAWN PREDICTOR CONFIGURATION")
        print("="*60)
        
        # Print the complete JSON configuration
        print("\nComplete JSON Configuration:")
        print("-" * 40)
        for key, value in self.config.items():
            # Format the output nicely
            if isinstance(value, str):
                print(f"  {key:20s}: \"{value}\"")
            else:
                print(f"  {key:20s}: {value}")
        
        print("\n" + "-" * 40)
        print("Parsed Configuration Parameters:")
        print("-" * 40)
        
        # Core parameters
        print(f"  CSLICS UUID        : {self.cslics_uuid}")
        print(f"  Images directory   : {self.img_dir}")
        print(f"  Output directory   : {self.save_dir}")
        print(f"  Processing mode    : {self.mode}")
        print(f"  Submersion time    : {self.submersion_time}")
        
        # Model paths
        print(f"  Surface model      : {self.surface_weights_path}")
        print(f"  Surface model ID   : {self.surface_model_id}")
        print(f"  Subsurface model   : {self.subsurface_weights_path}")
        print(f"  Subsurface model ID: {self.subsurface_model_id}")
        
        # Processing parameters
        print(f"  Confidence thresh  : {self.conf_thresh}")
        print(f"  IoU threshold      : {self.iou_thresh}")
        print(f"  Max detections     : {self.max_det}")
        print(f"  Max images         : {self.max_images if self.max_images else 'No limit'}")
        
        # Processing options
        print(f"  Parallel processing: {'Enabled' if self.parallel else 'Disabled'}")
        print(f"  Verbose output     : {'Enabled' if self.verbose else 'Disabled'}")
        print(f"  Plot only mode     : {'Enabled' if self.plot_only else 'Disabled'}")
        
        # Save options
        print(f"  Save images        : {'Enabled' if self.save_img else 'Disabled'}")
        print(f"  Save text files    : {'Enabled' if self.save_txt else 'Disabled'}")
        print(f"  Save bbox text     : {'Enabled' if self.save_txt_bb else 'Disabled'}")
        
        # Resume options
        print(f"  Resume mode        : {'Enabled' if self.resume else 'Disabled'}")
        if self.resume and self.resume_from_image:
            print(f"  Resume from image  : {self.resume_from_image}")
        
        # Timestamps
        print(f"  Submersion datetime: {self.submersion_datetime}")
        print(f"  Current time       : {self.current_datetime}")
        
        print("="*60)
        print()

    def print_config_json_only(self):
        """Print only the raw JSON configuration in a formatted way."""
        print("\nJSON Configuration File Contents:")
        print("="*50)
        
        import json
        formatted_json = json.dumps(self.config, indent=2, separators=(',', ': '))
        print(formatted_json)
        
        print("="*50)
        print()

    def validate_and_print_config(self):
        """Validate configuration and print detailed summary with any warnings."""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION AND SUMMARY")
        print("="*60)
        
        # Print complete config first
        self.print_config_summary()
        
        # Validation checks
        warnings = []
        errors = []
        
        # Check file paths
        import os
        if not os.path.exists(self.surface_weights_path):
            errors.append(f"Surface model file not found: {self.surface_weights_path}")
        if not os.path.exists(self.subsurface_weights_path):
            errors.append(f"Subsurface model file not found: {self.subsurface_weights_path}")
        if not os.path.exists(self.img_dir):
            errors.append(f"Image directory not found: {self.img_dir}")
        
        # Check parameter ranges
        if not (0.0 <= self.conf_thresh <= 1.0):
            warnings.append(f"Confidence threshold {self.conf_thresh} outside recommended range [0.0, 1.0]")
        if not (0.0 <= self.iou_thresh <= 1.0):
            warnings.append(f"IoU threshold {self.iou_thresh} outside recommended range [0.0, 1.0]")
        if self.max_det <= 0:
            warnings.append(f"Max detections {self.max_det} should be positive")
        
                
        # Validate image_skip
        if not isinstance(self.image_skip, int) or self.image_skip < 1:
            raise ValueError(f"image_skip must be a positive integer, got: {self.image_skip}")
        
        # Check logical combinations
        if self.plot_only and not (self.save_img or self.save_txt):
            warnings.append("Plot-only mode enabled but no save options enabled")
        if self.resume and not os.path.exists(self.save_dir):
            warnings.append("Resume mode enabled but output directory doesn't exist")
        
        # Print validation results
        if errors:
            print("ERRORS FOUND:")
            print("-" * 20)
            for error in errors:
                print(f" {error}")
            print()
        
        if warnings:
            print("WARNINGS:")
            print("-" * 20)
            for warning in warnings:
                print(f"{warning}")
            print()
        
        if not errors and not warnings:
            print("Configuration validation passed - no issues found")
            print()
        
        return len(errors) == 0  # Return True if no errors