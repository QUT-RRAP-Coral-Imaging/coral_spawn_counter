# models/model_manager.py
import os
import torch
from pathlib import Path
from ultralytics import YOLO


class ModelManager:
    """Manages YOLO model loading, initialization, and class information."""
    
    def __init__(self, config, device, model_type):
        """
        Initialize the model manager.
        
        Args:
            config: ConfigManager instance with model configuration
            device: torch.device for model inference
            model_type: str, either 'surface' or 'subsurface'
        """
        self.config = config
        self.device = device
        self.model_type = model_type
        
        # Validate model_type
        if model_type not in ['surface', 'subsurface']:
            raise ValueError(f"model_type must be 'surface' or 'subsurface', got '{model_type}'")
        
        # Model paths based on type
        if model_type == 'surface':
            self.weights_path = config.surface_weights_path
        else:  # subsurface
            self.weights_path = config.subsurface_weights_path
        
        # Processing parameters
        self.mode = config.mode
        self.verbose = config.verbose
        
        # Initialize model and class information
        self.model = None
        self.classes = None
        self.class_colours = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize YOLO model and extract class information for the specific model type."""
        print(f"Initializing {self.model_type} model...")
        
        print(f"Loading {self.model_type} model from: {self.weights_path}")
        self._validate_model_path(self.weights_path, self.model_type)
        self.model = YOLO(self.weights_path)
        
        # Move model to appropriate device
        if torch.cuda.is_available():
            self.model.to(self.device)
        
        # Extract class information
        self.classes = self._extract_class_names(self.model)
        self.class_colours = self._generate_class_colours(len(self.classes))
        
        print(f"{self.model_type.capitalize()} model loaded successfully with {len(self.classes)} classes")
        
        print(f"{self.model_type.capitalize()} classes: {self.classes}")
        
        print(f"{self.model_type.capitalize()} model initialization complete")
    
    def _validate_model_path(self, model_path, model_type):
        """
        Validate that the model file exists and has the correct extension.
        
        Args:
            model_path: Path to the model file
            model_type: Type of model ("surface" or "subsurface") for error messages
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"{model_type.capitalize()} model file not found: {model_path}")
        
        # Check if it's a valid model file extension
        valid_extensions = ['.pt', '.pth', '.onnx', '.torchscript']
        if not any(model_path.endswith(ext) for ext in valid_extensions):
            print(f"Warning: {model_type} model file does not have a recognized extension: {model_path}")
    
    def _extract_class_names(self, model):
        """
        Extract class names from a YOLO model.
        
        Args:
            model: YOLO model instance
            
        Returns:
            list: List of class names
        """
        try:
            # Try to get class names from the model
            if hasattr(model, 'names') and model.names:
                return list(model.names.values())
            elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                return list(model.model.names.values())
            else:
                # Fallback: try to infer from a dummy prediction
                dummy_results = model.predict(source=torch.zeros((1, 3, 640, 640)), verbose=False)
                if dummy_results and hasattr(dummy_results[0], 'names'):
                    return list(dummy_results[0].names.values())
                else:
                    # Last resort: use generic class names
                    print("Warning: Could not extract class names from model, using generic names")
                    return [f"class_{i}" for i in range(80)]  # COCO has 80 classes by default
        except Exception as e:
            print(f"Error extracting class names: {e}")
            # Return generic class names as fallback
            return [f"class_{i}" for i in range(80)]
    
    def _generate_class_colours(self, num_classes):
        """
        Generate distinct colors for each class.
        
        Args:
            num_classes: Number of classes to generate colors for
            
        Returns:
            list: List of RGB color tuples
        """
        import colorsys
        
        colors = []
        for i in range(num_classes):
            # Generate evenly spaced hues
            hue = i / num_classes
            # Use high saturation and value for bright colors
            saturation = 0.8
            value = 0.9
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            # Convert to 0-255 range
            colors.append(tuple(int(c * 255) for c in rgb))
        
        return colors
    
    def get_model_for_image(self, is_surface):
        """
        Get the appropriate model for an image based on whether it's surface or subsurface.
        
        Args:
            is_surface: Boolean indicating if the image is a surface image
            
        Returns:
            tuple: (model, classes, class_colours, model_path) for the appropriate model
        """
        # Check if this model manager's type matches the requested type
        if (is_surface and self.model_type == 'surface') or (not is_surface and self.model_type == 'subsurface'):
            return (self.model, self.classes, self.class_colours, self.weights_path)
        else:
            raise ValueError(f"This model manager is for {self.model_type} but {'surface' if is_surface else 'subsurface'} model was requested")
    
    def get_model_info(self):
        """
        Get model information for this model type.
        
        Returns:
            dict: Dictionary with model information
        """
        model_id = getattr(self.config, f'{self.model_type}_model_id', f'{self.model_type}_model')
        
        return {
            "model": self.model,
            "classes": self.classes,
            "class_colours": self.class_colours,
            "model_path": self.weights_path,
            "model_id": model_id,
            "model_type": self.model_type,
            "num_classes": len(self.classes) if self.classes else 0
        }
    
    def predict(self, source, **kwargs):
        """
        Run prediction using this model.
        
        Args:
            source: Input source for prediction
            **kwargs: Additional arguments to pass to model.predict()
            
        Returns:
            Prediction results from the model
        """
        if self.model is None:
            raise ValueError(f"{self.model_type.capitalize()} model not initialized")
        
        # Set default prediction parameters
        default_params = {
            'conf': getattr(self.config, 'confidence_threshold', 0.3),
            'verbose': self.verbose
        }
        
        # Add additional parameters if they exist in config
        if hasattr(self.config, 'iou_thresh'):
            default_params['iou'] = self.config.iou_thresh
        if hasattr(self.config, 'max_det'):
            default_params['max_det'] = self.config.max_det
        if hasattr(self.config, 'agnostic_nms'):
            default_params['agnostic_nms'] = self.config.agnostic_nms
        
        # Update with any provided kwargs
        default_params.update(kwargs)
        
        return self.model.predict(source=source, **default_params)
    
    def get_class_names(self):
        """
        Get class names for this model.
        
        Returns:
            list: List of class names
        """
        return self.classes if self.classes else []
    
    def get_model_name(self):
        """
        Extract model name from weights path.
        
        Returns:
            str: Model name
        """
        return Path(self.weights_path).stem
    
    def print_model_summary(self):
        """Print a summary of the loaded model and its information."""
        print(f"\nModel Manager Summary ({self.model_type.capitalize()}):")
        print(f"Device: {self.device}")
        print(f"Model Type: {self.model_type}")
        print(f"Model Path: {self.weights_path}")
        print(f"Model Name: {self.get_model_name()}")
        print(f"Number of classes: {len(self.classes) if self.classes else 0}")
        if self.classes:
            print(f"Classes: {self.classes}")
    
    def cleanup(self):
        """Clean up model resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print(f"{self.model_type.capitalize()} model resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup

    # Legacy compatibility methods for backward compatibility
    @property
    def surface_classes(self):
        """Legacy property for surface classes."""
        if self.model_type == 'surface':
            return self.classes
        else:
            return None
    
    @property
    def subsurface_classes(self):
        """Legacy property for subsurface classes."""
        if self.model_type == 'subsurface':
            return self.classes
        else:
            return None
    
    @property
    def surface_class_colours(self):
        """Legacy property for surface class colours."""
        if self.model_type == 'surface':
            return self.class_colours
        else:
            return None
    
    @property
    def subsurface_class_colours(self):
        """Legacy property for subsurface class colours."""
        if self.model_type == 'subsurface':
            return self.class_colours
        else:
            return None