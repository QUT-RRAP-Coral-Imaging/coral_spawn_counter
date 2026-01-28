# data/file_manager.py
import os
import json
import cv2
import torch
from pathlib import Path
from datetime import datetime


class FileManager:
    """Manages file operations including directory creation and saving outputs."""
    
    def __init__(self, config):
        """
        Initialize the file manager.
        
        Args:
            config: ConfigManager instance with configuration parameters
        """
        self.config = config
        self.cslics_uuid = config.cslics_uuid
        self.save_dir = config.save_dir
        self.img_dir = config.img_dir
        self.mode = config.mode
        self.save_img = config.save_img
        self.save_txt = config.save_txt
        self.save_txt_bb = config.save_txt_bb
        
        # Model information
        self.surface_model_id = config.surface_model_id
        self.subsurface_model_id = config.subsurface_model_id
        
        # Create output directories
        self._prepare_output_directories()
    
    def _prepare_output_directories(self):
        """Prepare output directories for saving results."""
        # Create base output directory
        self.output_base_dir = os.path.join(self.save_dir, self.cslics_uuid)
        os.makedirs(self.output_base_dir, exist_ok=True)
        
        # Initialize directory paths
        self.surface_imgsave_dir = None
        self.surface_txtsave_dir = None
        self.subsurface_imgsave_dir = None
        self.subsurface_txtsave_dir = None
        
        # Create surface model directories if needed
        if self.mode in ["surface", "both"]:
            self.surface_imgsave_dir = os.path.join(self.output_base_dir, self.surface_model_id, "detections_images")
            self.surface_txtsave_dir = os.path.join(self.output_base_dir, self.surface_model_id, "detections_text")
            
            if self.save_img:
                os.makedirs(self.surface_imgsave_dir, exist_ok=True)
            if self.save_txt or self.save_txt_bb:
                os.makedirs(self.surface_txtsave_dir, exist_ok=True)
        
        # Create subsurface model directories if needed
        if self.mode in ["subsurface", "both"]:
            self.subsurface_imgsave_dir = os.path.join(self.output_base_dir, self.subsurface_model_id, "detections_images")
            self.subsurface_txtsave_dir = os.path.join(self.output_base_dir, self.subsurface_model_id, "detections_text")
            
            if self.save_img:
                os.makedirs(self.subsurface_imgsave_dir, exist_ok=True)
            if self.save_txt or self.save_txt_bb:
                os.makedirs(self.subsurface_txtsave_dir, exist_ok=True)
        
        print(f"Output directories prepared in: {self.output_base_dir}")
    
    def save_image_predictions_bb(self, predictions, img_path, img_save_dir, classes, class_colours):
        """
        Save an image with bounding box annotations.
        
        Args:
            predictions: Tensor of predictions [x1, y1, x2, y2, conf, cls]
            img_path: Path to the original image
            img_save_dir: Directory to save the annotated image
            classes: List/dict of class names
            class_colours: Dict/list of class colors
        """
        if not self.save_img or len(predictions) == 0:
            return
        
        try:
            # Ensure output directory exists
            os.makedirs(img_save_dir, exist_ok=True)
            
            # Load the original image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"Warning: Could not load image {img_path}")
                return
            
            # Get image dimensions
            img_height, img_width = img.shape[:2]
            
            # Draw bounding boxes
            for pred in predictions:
                x1, y1, x2, y2, conf, cls = pred[:6]
                
                # Convert normalized coordinates to pixel coordinates
                if torch.is_tensor(x1):
                    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                    conf = conf.item()
                    cls = int(cls.item())
                else:
                    cls = int(cls)
                
                # Handle normalized coordinates (0-1 range)
                if x1 <= 1.0 and y1 <= 1.0 and x2 <= 1.0 and y2 <= 1.0:
                    x1 = int(x1 * img_width)
                    y1 = int(y1 * img_height)
                    x2 = int(x2 * img_width)
                    y2 = int(y2 * img_height)
                else:
                    # Already in pixel coordinates
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Get class name and color
                class_name = self._get_class_name(cls, classes)
                color = self._get_class_color(cls, class_colours)
                
                # Draw bounding box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                
                # Draw label background
                label = f"{class_name}: {conf:.2f}"
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                cv2.rectangle(img, (x1, y1 - label_size[1] - 10), 
                             (x1 + label_size[0], y1), color, -1)
                
                # Draw label text
                cv2.putText(img, label, (x1, y1 - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save the annotated image
            img_filename = Path(img_path).stem + '_det.jpg'
            img_save_path = os.path.join(img_save_dir, img_filename)
            
            success = cv2.imwrite(img_save_path, img)
            if not success:
                print(f"Warning: Failed to save image {img_save_path}")
                
        except Exception as e:
            print(f"Error saving image predictions for {img_path}: {e}")
    
    def save_txt_predictions_bb(self, predictions, img_path, txt_save_dir):
        """
        Save predictions in YOLO text format (normalized coordinates).
        
        Args:
            predictions: Tensor of predictions [x1, y1, x2, y2, conf, cls]
            img_path: Path to the original image
            txt_save_dir: Directory to save the text file
        """
        if not self.save_txt:
            return
        
        try:
            # Ensure output directory exists
            os.makedirs(txt_save_dir, exist_ok=True)
            
            # Create text file path
            txt_filename = Path(img_path).stem + '_det.txt'
            txt_save_path = os.path.join(txt_save_dir, txt_filename)
            
            with open(txt_save_path, 'w') as f:
                for pred in predictions:
                    x1, y1, x2, y2, conf, cls = pred[:6]
                    
                    # Convert to CPU and extract values if tensors
                    if torch.is_tensor(x1):
                        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                        conf = conf.item()
                        cls = int(cls.item())
                    else:
                        cls = int(cls)
                    
                    # Convert to center coordinates and width/height (YOLO format)
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    # Write in YOLO format: class center_x center_y width height confidence
                    f.write(f"{cls} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f} {conf:.6f}\n")
                    
        except Exception as e:
            print(f"Error saving text predictions for {img_path}: {e}")
    
    def save_json_predictions_bb(self, predictions, img_path, json_save_dir, model_path, classes):
        """
        Save predictions in JSON format with metadata.
        
        Args:
            predictions: Tensor of predictions [x1, y1, x2, y2, conf, cls]
            img_path: Path to the original image
            json_save_dir: Directory to save the JSON file
            model_path: Path to the model used for prediction
            classes: List/dict of class names
        """
        if not self.save_txt:
            return
        
        try:
            # Ensure output directory exists
            os.makedirs(json_save_dir, exist_ok=True)
            
            # Create JSON file path
            json_filename = Path(img_path).stem + '_det.json'
            json_save_path = os.path.join(json_save_dir, json_filename)
            
            # Prepare JSON data
            json_data = {
                "image_path": str(img_path),
                "image_filename": Path(img_path).name,
                "model_path": str(model_path),
                "model_id": Path(model_path).stem,
                "timestamp": datetime.now().isoformat(),
                "cslics_uuid": self.cslics_uuid,
                "detection_count": len(predictions),
                "detections": []
            }
            
            # Add detection data
            for i, pred in enumerate(predictions):
                x1, y1, x2, y2, conf, cls = pred[:6]
                
                # Convert to CPU and extract values if tensors
                if torch.is_tensor(x1):
                    x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                    conf = conf.item()
                    cls = int(cls.item())
                else:
                    cls = int(cls)
                
                # Get class name
                class_name = self._get_class_name(cls, classes)
                
                # Convert to center coordinates and width/height
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                width = x2 - x1
                height = y2 - y1
                
                detection = {
                    "detection_id": i,
                    "class_id": cls,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_normalized": {
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2
                    },
                    "bbox_yolo_format": {
                        "center_x": center_x,
                        "center_y": center_y,
                        "width": width,
                        "height": height
                    }
                }
                
                json_data["detections"].append(detection)
            
            # Save JSON file
            with open(json_save_path, 'w') as f:
                json.dump(json_data, f, indent=2, separators=(',', ': '))
                
        except Exception as e:
            print(f"Error saving JSON predictions for {img_path}: {e}")
    
    def save_bounding_box_txt(self, predictions, img_path, txt_save_dir):
        """
        Save predictions in bounding box text format (x1 y1 x2 y2 confidence).
        
        Args:
            predictions: Tensor of predictions [x1, y1, x2, y2, conf, cls]
            img_path: Path to the original image
            txt_save_dir: Directory to save the text file
        """
        if not self.save_txt_bb:
            return
        
        try:
            # Ensure output directory exists
            os.makedirs(txt_save_dir, exist_ok=True)
            
            # Create text file path
            txt_filename = Path(img_path).stem + '_det_bb.txt'
            txt_save_path = os.path.join(txt_save_dir, txt_filename)
            
            with open(txt_save_path, 'w') as f:
                for pred in predictions:
                    x1, y1, x2, y2, conf, cls = pred[:6]
                    
                    # Convert to CPU and extract values if tensors
                    if torch.is_tensor(x1):
                        x1, y1, x2, y2 = x1.item(), y1.item(), x2.item(), y2.item()
                        conf = conf.item()
                    
                    # Write in bounding box format: x1 y1 x2 y2 confidence
                    f.write(f"{x1:.6f} {y1:.6f} {x2:.6f} {y2:.6f} {conf:.6f}\n")
                    
        except Exception as e:
            print(f"Error saving bounding box text for {img_path}: {e}")
    
    def _get_class_name(self, cls_id, classes):
        """
        Get class name from class ID.
        
        Args:
            cls_id: Class ID
            classes: List/dict of class names
            
        Returns:
            str: Class name
        """
        try:
            if isinstance(classes, dict):
                return classes.get(cls_id, f"class_{cls_id}")
            elif isinstance(classes, list):
                if 0 <= cls_id < len(classes):
                    return classes[cls_id]
                else:
                    return f"class_{cls_id}"
            else:
                return f"class_{cls_id}"
        except Exception:
            return f"class_{cls_id}"
    
    def _get_class_color(self, cls_id, class_colours):
        """
        Get color for a class ID.
        
        Args:
            cls_id: Class ID
            class_colours: Dict/list of class colors
            
        Returns:
            tuple: BGR color tuple for OpenCV
        """
        try:
            if isinstance(class_colours, dict):
                color = class_colours.get(cls_id, (0, 255, 0))  # Default green
            elif isinstance(class_colours, list):
                if 0 <= cls_id < len(class_colours):
                    color = class_colours[cls_id]
                else:
                    color = (0, 255, 0)  # Default green
            else:
                color = (0, 255, 0)  # Default green
            
            # Ensure color is in BGR format for OpenCV
            if len(color) == 3:
                # Convert RGB to BGR if needed
                return (color[2], color[1], color[0])
            else:
                return (0, 255, 0)  # Default green
                
        except Exception:
            return (0, 255, 0)  # Default green
    
    def create_subdirectories(self, img_path, base_img_dir, base_txt_dir):
        """
        Create subdirectories based on the relative path of the image.
        
        Args:
            img_path: Path to the image file
            base_img_dir: Base directory for saving images
            base_txt_dir: Base directory for saving text files
            
        Returns:
            tuple: (img_subdir, txt_subdir) paths
        """
        # Get relative path from the original image directory
        rel_path = os.path.relpath(os.path.dirname(img_path), self.img_dir)
        
        # Create subdirectories
        img_subdir = os.path.join(base_img_dir, rel_path) if base_img_dir else None
        txt_subdir = os.path.join(base_txt_dir, rel_path) if base_txt_dir else None
        
        # Create directories if they don't exist
        if img_subdir and self.save_img:
            os.makedirs(img_subdir, exist_ok=True)
        if txt_subdir and (self.save_txt or self.save_txt_bb):
            os.makedirs(txt_subdir, exist_ok=True)
        
        return img_subdir, txt_subdir
    
    def get_output_paths(self, img_path, model_type):
        """
        Get the appropriate output directories for an image based on model type.
        
        Args:
            img_path: Path to the image file
            model_type: Either "surface" or "subsurface"
            
        Returns:
            tuple: (img_save_dir, txt_save_dir)
        """
        if model_type == "surface":
            base_img_dir = self.surface_imgsave_dir
            base_txt_dir = self.surface_txtsave_dir
        elif model_type == "subsurface":
            base_img_dir = self.subsurface_imgsave_dir
            base_txt_dir = self.subsurface_txtsave_dir
        else:
            raise ValueError(f"Invalid model type: {model_type}")
        
        return self.create_subdirectories(img_path, base_img_dir, base_txt_dir)
    
    def cleanup_empty_directories(self):
        """Remove empty directories from the output structure."""
        def remove_empty_dirs(path):
            """Recursively remove empty directories."""
            if not os.path.isdir(path):
                return
            
            # Remove empty subdirectories first
            for subdir in os.listdir(path):
                subdir_path = os.path.join(path, subdir)
                if os.path.isdir(subdir_path):
                    remove_empty_dirs(subdir_path)
            
            # Remove this directory if it's empty
            try:
                if not os.listdir(path):
                    os.rmdir(path)
                    print(f"Removed empty directory: {path}")
            except OSError:
                pass  # Directory not empty or other error
        
        # Clean up all output directories
        for directory in [self.surface_imgsave_dir, self.surface_txtsave_dir,
                         self.subsurface_imgsave_dir, self.subsurface_txtsave_dir]:
            if directory and os.path.exists(directory):
                remove_empty_dirs(directory)
    
    def get_directory_info(self):
        """
        Get information about the created directories.
        
        Returns:
            dict: Directory information
        """
        info = {
            "base_output_dir": self.output_base_dir,
            "cslics_uuid": self.cslics_uuid,
            "directories_created": []
        }
        
        if self.mode in ["surface", "both"]:
            info["surface_img_dir"] = self.surface_imgsave_dir
            info["surface_txt_dir"] = self.surface_txtsave_dir
            if self.save_img and self.surface_imgsave_dir:
                info["directories_created"].append(self.surface_imgsave_dir)
            if (self.save_txt or self.save_txt_bb) and self.surface_txtsave_dir:
                info["directories_created"].append(self.surface_txtsave_dir)
        
        if self.mode in ["subsurface", "both"]:
            info["subsurface_img_dir"] = self.subsurface_imgsave_dir
            info["subsurface_txt_dir"] = self.subsurface_txtsave_dir
            if self.save_img and self.subsurface_imgsave_dir:
                info["directories_created"].append(self.subsurface_imgsave_dir)
            if (self.save_txt or self.save_txt_bb) and self.subsurface_txtsave_dir:
                info["directories_created"].append(self.subsurface_txtsave_dir)
        
        return info
    
    def print_directory_summary(self):
        """Print a summary of the file manager setup."""
        info = self.get_directory_info()
        
        print("\nFile Manager Summary:")
        print(f"CSLICS UUID: {info['cslics_uuid']}")
        print(f"Base output directory: {info['base_output_dir']}")
        print(f"Processing mode: {self.mode}")
        print(f"Save images: {self.save_img}")
        print(f"Save text files: {self.save_txt}")
        print(f"Save bounding box text: {self.save_txt_bb}")
        
        if info["directories_created"]:
            print("Directories created:")
            for directory in info["directories_created"]:
                print(f"  {directory}")
        else:
            print("No output directories created (saving disabled)")
    
    def validate_output_directories(self):
        """
        Validate that output directories exist and are writable.
        
        Returns:
            dict: Validation results
        """
        validation = {
            "all_valid": True,
            "directory_checks": {}
        }
        
        directories_to_check = []
        
        if self.mode in ["surface", "both"]:
            if self.save_img and self.surface_imgsave_dir:
                directories_to_check.append(("surface_img", self.surface_imgsave_dir))
            if (self.save_txt or self.save_txt_bb) and self.surface_txtsave_dir:
                directories_to_check.append(("surface_txt", self.surface_txtsave_dir))
        
        if self.mode in ["subsurface", "both"]:
            if self.save_img and self.subsurface_imgsave_dir:
                directories_to_check.append(("subsurface_img", self.subsurface_imgsave_dir))
            if (self.save_txt or self.save_txt_bb) and self.subsurface_txtsave_dir:
                directories_to_check.append(("subsurface_txt", self.subsurface_txtsave_dir))
        
        for name, directory in directories_to_check:
            check_result = {
                "exists": os.path.exists(directory),
                "is_directory": os.path.isdir(directory) if os.path.exists(directory) else False,
                "writable": os.access(directory, os.W_OK) if os.path.exists(directory) else False
            }
            
            validation["directory_checks"][name] = check_result
            
            if not all(check_result.values()):
                validation["all_valid"] = False
        
        return validation
    
    def filter_images_for_resume(self, img_list, resume_image_name):
        """
        Filter image list to start from the resume image.
        
        Args:
            img_list: List of image paths
            resume_image_name: Name of the image to resume from
            
        Returns:
            tuple: (filtered_images, resume_index, skipped_count)
        """
        if not resume_image_name:
            return img_list, 0, 0
        
        # Find the resume image in the list
        resume_index = None
        for i, img_path in enumerate(img_list):
            if os.path.basename(img_path) == resume_image_name:
                resume_index = i
                break
        
        if resume_index is None:
            print(f"Warning: Resume image '{resume_image_name}' not found in image list")
            print("Processing will start from the beginning")
            return img_list, 0, 0
        
        # Return images starting from the resume point
        filtered_images = img_list[resume_index:]
        skipped_count = resume_index
        
        print(f"Resume mode: Starting from image {resume_index + 1} of {len(img_list)}")
        print(f"Skipping {skipped_count} images")
        print(f"Processing {len(filtered_images)} remaining images")
        
        return filtered_images, resume_index, skipped_count

    def check_resume_image_processed(self, resume_image_name, model_type):
        """
        Check if the resume image has already been processed.
        
        Args:
            resume_image_name: Name of the resume image
            model_type: Either "surface" or "subsurface"
            
        Returns:
            bool: True if already processed, False otherwise
        """
        if not resume_image_name:
            return False
        
        # Get the appropriate output directory
        if model_type == "surface":
            txt_dir = self.surface_txtsave_dir
        else:
            txt_dir = self.subsurface_txtsave_dir
        
        if not txt_dir:
            return False
        
        # Check if detection files exist for this image
        base_name = os.path.splitext(resume_image_name)[0]
        
        # Look for detection files in the directory structure
        for root, dirs, files in os.walk(txt_dir):
            for file in files:
                if file.startswith(base_name) and ('_det.txt' in file or '_det.json' in file):
                    return True
        
        return False


