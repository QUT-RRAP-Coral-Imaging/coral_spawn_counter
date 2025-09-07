# models/model_manager.py
import os
import torch
from pathlib import Path
from ultralytics import YOLO


class ModelManager:
    """Manages YOLO model loading, initialization, and class information."""
    
    def __init__(self, config, device):
        """
        Initialize the model manager.
        
        Args:
            config: ConfigManager instance with model configuration
            device: torch.device for model inference
        """
        self.config = config
        self.device = device
        
        # Model paths
        self.surface_weights_path = config.surface_weights_path
        self.subsurface_weights_path = config.subsurface_weights_path
        
        # Processing mode
        self.mode = config.mode
        self.verbose = config.verbose
        
        # Initialize models and class information
        self.surface_model = None
        self.subsurface_model = None
        self.surface_classes = None
        self.subsurface_classes = None
        self.surface_class_colours = None
        self.subsurface_class_colours = None
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize YOLO models and extract class information based on processing mode."""
        print("Initializing models...")
        
        # Initialize surface model if needed
        if self.mode in ["surface", "both"]:
            print(f"Loading surface model from: {self.surface_weights_path}")
            self._validate_model_path(self.surface_weights_path, "surface")
            self.surface_model = YOLO(self.surface_weights_path)
            
            # Move model to appropriate device
            if torch.cuda.is_available():
                self.surface_model.to(self.device)
            
            # Extract class information
            self.surface_classes = self._extract_class_names(self.surface_model)
            self.surface_class_colours = self._generate_class_colours(len(self.surface_classes))
            
            print(f"Surface model loaded successfully with {len(self.surface_classes)} classes")
            if self.verbose:
                print(f"Surface classes: {self.surface_classes}")
        
        # Initialize subsurface model if needed
        if self.mode in ["subsurface", "both"]:
            print(f"Loading subsurface model from: {self.subsurface_weights_path}")
            self._validate_model_path(self.subsurface_weights_path, "subsurface")
            self.subsurface_model = YOLO(self.subsurface_weights_path)
            
            # Move model to appropriate device
            if torch.cuda.is_available():
                self.subsurface_model.to(self.device)
            
            # Extract class information
            self.subsurface_classes = self._extract_class_names(self.subsurface_model)
            self.subsurface_class_colours = self._generate_class_colours(len(self.subsurface_classes))
            
            print(f"Subsurface model loaded successfully with {len(self.subsurface_classes)} classes")
            if self.verbose:
                print(f"Subsurface classes: {self.subsurface_classes}")
        
        print("Model initialization complete")
    
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
        if is_surface:
            if self.surface_model is None:
                raise ValueError("Surface model not initialized but surface image encountered")
            return (
                self.surface_model,
                self.surface_classes,
                self.surface_class_colours,
                self.surface_weights_path
            )
        else:
            if self.subsurface_model is None:
                raise ValueError("Subsurface model not initialized but subsurface image encountered")
            return (
                self.subsurface_model,
                self.subsurface_classes,
                self.subsurface_class_colours,
                self.subsurface_weights_path
            )
    
    def get_model_info(self, model_type):
        """
        Get model information for a specific model type.
        
        Args:
            model_type: Either "surface" or "subsurface"
            
        Returns:
            dict: Dictionary with model information
        """
        if model_type == "surface":
            return {
                "model": self.surface_model,
                "classes": self.surface_classes,
                "class_colours": self.surface_class_colours,
                "model_path": self.surface_weights_path,
                "model_id": self.config.surface_model_id,
                "num_classes": len(self.surface_classes) if self.surface_classes else 0
            }
        elif model_type == "subsurface":
            return {
                "model": self.subsurface_model,
                "classes": self.subsurface_classes,
                "class_colours": self.subsurface_class_colours,
                "model_path": self.subsurface_weights_path,
                "model_id": self.config.subsurface_model_id,
                "num_classes": len(self.subsurface_classes) if self.subsurface_classes else 0
            }
        else:
            raise ValueError(f"Invalid model type: {model_type}. Must be 'surface' or 'subsurface'")
    
    def predict(self, model_type, source, **kwargs):
        """
        Run prediction using the specified model.
        
        Args:
            model_type: Either "surface" or "subsurface"
            source: Input source for prediction
            **kwargs: Additional arguments to pass to model.predict()
            
        Returns:
            Prediction results from the model
        """
        model_info = self.get_model_info(model_type)
        model = model_info["model"]
        
        if model is None:
            raise ValueError(f"{model_type.capitalize()} model not initialized")
        
        # Set default prediction parameters
        default_params = {
            'iou': self.config.iou_thresh,
            'conf': self.config.conf_thresh,
            'agnostic_nms': True,
            'max_det': self.config.max_det,
            'verbose': self.verbose
        }
        
        # Update with any provided kwargs
        default_params.update(kwargs)
        
        return model.predict(source=source, **default_params)
    
    def get_available_models(self):
        """
        Get a list of available models based on the processing mode.
        
        Returns:
            list: List of available model types
        """
        available = []
        if self.mode in ["surface", "both"] and self.surface_model is not None:
            available.append("surface")
        if self.mode in ["subsurface", "both"] and self.subsurface_model is not None:
            available.append("subsurface")
        return available
    
    def print_model_summary(self):
        """Print a summary of loaded models and their information."""
        print("\nModel Manager Summary:")
        print(f"Device: {self.device}")
        print(f"Processing mode: {self.mode}")
        
        available_models = self.get_available_models()
        for model_type in available_models:
            info = self.get_model_info(model_type)
            print(f"\n{model_type.capitalize()} Model:")
            print(f"  Path: {info['model_path']}")
            print(f"  Model ID: {info['model_id']}")
            print(f"  Number of classes: {info['num_classes']}")
            if self.verbose and info['classes']:
                print(f"  Classes: {info['classes']}")
    
    def cleanup(self):
        """Clean up model resources."""
        if self.surface_model is not None:
            del self.surface_model
            self.surface_model = None
        
        if self.subsurface_model is not None:
            del self.subsurface_model
            self.subsurface_model = None
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("Model resources cleaned up")
    
    def __del__(self):
        """Destructor to ensure proper cleanup."""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup