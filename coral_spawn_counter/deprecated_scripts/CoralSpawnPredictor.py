#!/usr/bin/env python3

"""
CSLICS Data Processor
- Encapsulates functionality for processing and plotting tank estimates with manual counts.
"""
import os
import json
import time
from pathlib import Path
from datetime import datetime
import torch
import cv2 as cv
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import fcntl  # For file locking on Linux/Unix


class CoralSpawnPredictor:
    def __init__(self, config_file):
        """
        Initialize the coral spawn predictor with a config file.
        
        Args:
            config_file: Path to the JSON configuration file
        """
        # Load configuration from the JSON file
        self.config = self.load_config_from_json(config_file)
        
        # Extract all parameters at once rather than repeatedly accessing the dictionary
        self._extract_config_parameters()
        
        # Validate parameters before proceeding with expensive operations
        self._validate_params()
        
        # Initialize device once
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Initialize models and load class information
        self._initialize_models()
        
        # Prepare output directories
        self._prepare_output_directories()
        
        # Initialize detection data files
        self._initialize_detection_files()
        
        # Print configuration summary
        self._print_config_summary()
    
        # Add tracking for cumulative detection data
        self.detection_data = {
            'surface': {},    # Will store timestamp -> count mappings for surface detections
            'subsurface': {}  # Will store timestamp -> count mappings for subsurface detections
        }
        self.total_detections = {
            'surface': 0,
            'subsurface': 0
        }
    
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
    
    def _validate_params(self):
        """Validate required parameters"""
        if not self.surface_weights_path:
            raise ValueError("surface_weights_path not specified")
        if not self.subsurface_weights_path:
            raise ValueError("subsurface_weights_path not specified")
        if not self.img_dir:
            raise ValueError("img_dir not specified")
        if not self.save_dir:
            raise ValueError("save_dir not specified")
        if not self.submersion_time:
            raise ValueError("submersion_time not specified")
    
    def _extract_config_parameters(self):
        """Extract configuration parameters from the loaded JSON config"""
        # Extract required parameters
        self.surface_weights_path = self.config['surface_weights_path']
        self.subsurface_weights_path = self.config['subsurface_weights_path']
        self.img_dir = self.config['img_dir']
        self.save_dir = self.config['save_dir']
        self.cslics_uuid = self.config['cslics_uuid']
        self.submersion_time = self.config['submersion_time']

        # Extract optional parameters with defaults
        self.mode = self.config.get('processing_mode', 'both')
        self.plot_only = self.config.get('plot_only', False)
        self.iou_thresh = float(self.config.get('iou_thresh', 0.3))
        self.conf_thresh = float(self.config.get('conf_thresh', 0.25))
        self.max_det = int(self.config.get('max_det', 1000))
        self.save_img = self.config.get('save_img', True)
        self.save_txt = self.config.get('save_txt', True)
        self.save_txt_bb = self.config.get('save_txt_bb', False)
        
        # Parse max_images more efficiently
        max_images_val = self.config.get('max_images')
        self.max_images = int(max_images_val) if max_images_val else None
        
        # Parse bool values
        self.verbose = self._parse_bool(self.config.get('verbose', False))
        self.parallel = self._parse_bool(self.config.get('parallel', False))
        
        # Parse submersion time once
        self.submersion_datetime = datetime.strptime(self.submersion_time, "%Y-%m-%d_%H-%M-%S")
        
        # Get current time once
        self.current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Extract model IDs (stem names) 
        self.surface_model_id = Path(self.surface_weights_path).stem
        self.subsurface_model_id = Path(self.subsurface_weights_path).stem

    def _initialize_models(self):
        """Initialize models based on mode - only load what's needed"""
        # Only load needed models based on mode
        self.surface_model = None
        self.subsurface_model = None
        self.surface_classes = None
        self.surface_class_colours = None
        self.subsurface_classes = None
        self.subsurface_class_colours = None
        
        if self.mode in ["surface", "both"]:
            print(f'Loading surface model: {self.surface_weights_path}')
            self.surface_model = YOLO(self.surface_weights_path, verbose=self.verbose).to(self.device)
            self.surface_classes = self._extract_classes_from_model(self.surface_model)
            self.surface_class_colours = self._generate_class_colors(self.surface_classes)
            print(f'Surface model classes: {self.surface_classes}')
            
        if self.mode in ["subsurface", "both"]:
            print(f'Loading subsurface model: {self.subsurface_weights_path}')
            self.subsurface_model = YOLO(self.subsurface_weights_path, verbose=self.verbose).to(self.device)
            self.subsurface_classes = self._extract_classes_from_model(self.subsurface_model)
            self.subsurface_class_colours = self._generate_class_colors(self.subsurface_classes)
            print(f'Subsurface model classes: {self.subsurface_classes}')

    @staticmethod
    def _parse_bool(value):
        """Parse boolean values from various formats"""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.lower() in ('true', 'yes', '1', 't', 'y')
        return bool(value)
    
    def _print_config_summary(self):
        """Print a summary of the current configuration"""
        print("\nCoralSpawnPredictor Configuration:")
        print(f"  CSLICS UUID: {self.cslics_uuid}")
        print(f"  Images directory: {self.img_dir}")
        print(f"  Output directory: {self.save_dir}")
        print(f"  Processing mode: {self.mode}")
        print(f"  Submersion time: {self.submersion_time}")
        print(f"  Parallel processing: {'Enabled' if self.parallel else 'Disabled'}")
        print(f"  Confidence threshold: {self.conf_thresh}")
        print(f"  IoU threshold: {self.iou_thresh}")
        if self.max_images:
            print(f"  Max images: {self.max_images}")
        print(f"  Device: {self.device}\n")
    
    def _extract_classes_from_model(self, model):
        """
        Extract class names from a YOLO model.
        
        Args:
            model: YOLO model instance
            
        Returns:
            list: List of class names
        """
        return model.names if hasattr(model, 'names') else []
    
    def _generate_class_colors(self, classes):
        """
        Generate colors for classes.
        
        Args:
            classes: List or dictionary of class names
            
        Returns:
            dict: Dictionary mapping class names to BGR colors
        """
        colors = {}
        
        # Define some distinct colors (BGR format)
        predefined_colors = [
            [0, 0, 255],    # Red
            [0, 255, 0],    # Green
            [255, 0, 0],    # Blue
            [0, 255, 255],  # Yellow
            [255, 0, 255],  # Magenta
            [255, 255, 0],  # Cyan
            [128, 0, 0],    # Dark blue
            [0, 128, 0],    # Dark green
            [0, 0, 128],    # Dark red
            [0, 128, 128]   # Dark yellow
        ]
        
        # If classes is a dictionary, use its keys
        class_names = classes.keys() if isinstance(classes, dict) else classes
        
        for i, name in enumerate(class_names):
            # Use predefined colors for the first few classes, then generate random colors
            if i < len(predefined_colors):
                colors[name] = predefined_colors[i]
            else:
                # Generate a random color (BGR)
                colors[name] = [
                    np.random.randint(0, 255),
                    np.random.randint(0, 255),
                    np.random.randint(0, 255)
                ]
        
        return colors
    
    def _prepare_output_directories(self):
        """Create output directories for both models."""
        
        # Create separate directories for surface and subsurface detections using UUID and model ID
        self.surface_save_dir = os.path.join(self.save_dir,  self.cslics_uuid, 'detections_surface', self.surface_model_id)
        self.subsurface_save_dir = os.path.join(self.save_dir, self.cslics_uuid, 'detections_subsurface', self.subsurface_model_id)

        # Surface directories
        self.surface_imgsave_dir = os.path.join(self.surface_save_dir, 'detections_images')
        self.surface_txtsave_dir = os.path.join(self.surface_save_dir, 'detections_txt')
        os.makedirs(self.surface_imgsave_dir, exist_ok=True)
        os.makedirs(self.surface_txtsave_dir, exist_ok=True)
        
        # Subsurface directories
        self.subsurface_imgsave_dir = os.path.join(self.subsurface_save_dir, 'detections_images')
        self.subsurface_txtsave_dir = os.path.join(self.subsurface_save_dir, 'detections_txt')
        os.makedirs(self.subsurface_imgsave_dir, exist_ok=True)
        os.makedirs(self.subsurface_txtsave_dir, exist_ok=True)
        
        # Print output directories
        print(f"Surface detection outputs will be saved to: {self.surface_save_dir}")
        print(f"Subsurface detection outputs will be saved to: {self.subsurface_save_dir}")
        
    
    def convert_to_decimal_days(self, dates_list, time_zero=None):
        """
        Convert datetime objects to decimal days since time_zero.
        
        Args:
            dates_list: List of datetime objects
            time_zero: Reference time (if None, use configured time_zero)
            
        Returns:
            list: Decimal days since time_zero
        """
        if time_zero is None:
            # Use configured time_zero or spawning_start_time as default
            if self.time_zero:
                time_zero = datetime.strptime(self.time_zero, "%Y-%m-%d_%H-%M-%S")
            elif self.spawning_start_time:
                time_zero = datetime.strptime(self.spawning_start_time, "%Y-%m-%d_%H-%M-%S")
            else:
                # Use the first date in the list
                time_zero = dates_list[0] if dates_list else datetime.now()
                print(f"No time_zero specified, using first date: {time_zero}")
        
        # Convert each datetime to days since time_zero
        decimal_days = []
        for dt in dates_list:
            delta = dt - time_zero
            days = delta.total_seconds() / (24 * 3600)  # Convert seconds to days
            decimal_days.append(days)
            
        return decimal_days

    def is_surface_image(self, img_path):
        """
        Determine if an image was captured before submersion time with caching.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            bool: True if the image is a surface image, False otherwise
        """
        # Use a class-level cache to avoid repeated timestamp parsing
        if not hasattr(self, '_timestamp_cache'):
            self._timestamp_cache = {}
        
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
    
    def process_image(self, img_name):
        """
        Process a single image and update detection data files.
        
        Args:
            img_name: Path to the image file
            
        Returns:
            tuple: (detection_count, model_type) or None if skipped
        """
        is_surface = self.is_surface_image(img_name)
        
        # Skip processing based on mode
        if (is_surface and self.mode == "subsurface") or (not is_surface and self.mode == "surface"):
            return 0, "skipped"
        
        # Determine model type for this image
        model_type = "surface" if is_surface else "subsurface"
        
        # Select appropriate model, classes, colors, and save directories
        if is_surface:
            model = self.surface_model
            classes = self.surface_classes
            class_colours = self.surface_class_colours
            imgsave_dir = self.surface_imgsave_dir
            txtsave_dir = self.surface_txtsave_dir
            model_path = self.surface_weights_path
        else:
            model = self.subsurface_model
            classes = self.subsurface_classes
            class_colours = self.subsurface_class_colours
            imgsave_dir = self.subsurface_imgsave_dir
            txtsave_dir = self.subsurface_txtsave_dir
            model_path = self.subsurface_weights_path

        # Run inference
        results = model.predict(
            source=img_name, 
            iou=self.iou_thresh, 
            conf=self.conf_thresh, 
            agnostic_nms=True, 
            max_det=self.max_det,
            verbose=self.verbose
        )
        
        boxes = results[0].boxes
        pred = []
        for b in boxes:
            if torch.cuda.is_available():
                xyxyn = b.xyxyn[0]
                pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
        predictions = torch.tensor(pred)
        
        # Extract timestamp from filename for detection tracking
        filename = Path(img_name).stem
        print(f'Attempting to update detection data for {model_type} image with {len(pred)} detections')
        try:
            timestamp_str = filename[9:-11]  # Assumes consistent filename format
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            
            # Update detection data file with ISO format timestamp
            iso_timestamp = timestamp.isoformat()
            print(f'Updating detection data for {model_type} image at {iso_timestamp} with {len(pred)} detections')
            self._update_detection_data(iso_timestamp, len(pred), model_type)
        except (ValueError, IndexError):
            # If timestamp parsing fails, just skip tracking this for plotting
            pass

        # Determine relative path for saving
        rel_path = os.path.relpath(os.path.dirname(img_name), self.img_dir)

        # Save image predictions
        if self.save_img:
            os.makedirs(os.path.join(imgsave_dir, rel_path), exist_ok=True)
            self.save_image_predictions_bb(predictions, img_name, os.path.join(imgsave_dir, rel_path), classes, class_colours)

        # Save text and JSON predictions
        if self.save_txt:
            os.makedirs(os.path.join(txtsave_dir, rel_path), exist_ok=True)
            self.save_txt_predictions_bb(predictions, img_name, os.path.join(txtsave_dir, rel_path))
            self.save_json_predictions_bb(predictions, img_name, os.path.join(txtsave_dir, rel_path), model_path, classes)
        
        # Save bounding box text format if enabled
        if self.save_txt_bb:
            bb_txt_save_path = os.path.join(txtsave_dir, rel_path, os.path.basename(img_name)[:-4] + '_det_bb.txt')
            with open(bb_txt_save_path, "w") as file:
                for p in predictions:
                    x1, y1, x2, y2 = p[0:4].tolist()
                    conf = p[4]
                    cls = int(p[5])
                    line = f"{x1} {y1} {x2} {y2} {conf}\n"
                    file.write(line)
    
        return len(pred), model_type
    
    def save_image_predictions_bb(self, predictions, imgname, imgsavedir, classes, class_colours):
        """
        Save predictions/detections on image as bounding box.
        
        Args:
            predictions: Detection predictions
            imgname: Path to the original image
            imgsavedir: Directory to save the annotated image
            classes: List or dictionary of class names
            class_colours: Dictionary mapping class names to BGR colors
        """
        FONT_SIZE = 2
        FONT_THICK = 2
        BOX_THICK = 2
        quality = 25

        img = cv.imread(imgname)  # BGR
        if img is None:
            print(f"Warning: Could not read image {imgname}")
            return
            
        imgw, imgh = img.shape[1], img.shape[0]
        for p in predictions:
            x1, y1, x2, y2 = p[0:4].tolist()
            conf = p[4]
            cls = int(p[5])
            
            # Get class name and color
            class_name = classes[cls]
            color = class_colours[cls]
            
            x1, x2 = x1 * imgw, x2 * imgw
            y1, y2 = y1 * imgh, y2 * imgh
            cv.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, BOX_THICK)
            cv.putText(img, f"{class_name}: {conf:.2f}", (int(x1), int(y1 - 5)), cv.FONT_HERSHEY_SIMPLEX, FONT_SIZE, color, FONT_THICK)
        
        imgsavename = os.path.basename(imgname)
        imgsave_path = os.path.join(imgsavedir, imgsavename[:-4] + '_det.jpg')
        encode_param = [int(cv.IMWRITE_JPEG_QUALITY), quality]
        cv.imwrite(imgsave_path, img, encode_param)

    def save_txt_predictions_bb(self, predictions, imgname, txtsavedir):
        """
        Save predictions/detections as bounding box in text format.
        
        Args:
            predictions: Detection predictions
            imgname: Path to the original image
            txtsavedir: Directory to save the text file
        """
        imgsavename = os.path.basename(imgname)
        txt_save_path = os.path.join(txtsavedir, imgsavename[:-4] + '_det.txt')
        with open(txt_save_path, "w") as file:
            for p in predictions:
                x1, y1, x2, y2 = p[0:4].tolist()
                conf = p[4]
                cls = int(p[5])
                line = f"{x1} {y1} {x2} {y2} {cls} {conf}\n"
                file.write(line)

    def save_json_predictions_bb(self, predictions, imgname, txtsavedir, model_path, classes):
        """
        Save predictions as bounding box in JSON format.
        
        Args:
            predictions: Detection predictions
            imgname: Path to the original image
            txtsavedir: Directory to save the JSON file
            model_path: Path to the model used for predictions
            classes: List or dictionary of class names
        """
        imgsavename = os.path.basename(imgname)
        json_save_path = os.path.join(txtsavedir, imgsavename[:-4] + '_det.json')
        
        # Convert class indices to names for better readability
        processed_detections = []
        for det in predictions.tolist():
            processed_det = det.copy()
            cls_idx = int(processed_det[5])
            processed_det.append(classes[cls_idx])
            processed_detections.append(processed_det)
        
        predictions_dict = {
            "model_name": Path(model_path).stem,
            "date run": self.current_datetime,
            "classes": classes,
            "detections [xn1, yn1, xn2, yn2, conf, cls, cls_name]": processed_detections
        }
        with open(json_save_path, 'w') as f:
            json.dump(predictions_dict, f, indent=4)
    
    def run(self):
        """
        Run prediction on all images with improved efficiency.
        """
        print(f'Fetching image list in all subfolders from: {self.img_dir}')
        print(f'Processing mode: {self.mode}')
        print(f'Parallel processing: {"Enabled" if self.parallel else "Disabled"}')
        
        # Use faster method to gather images
        start_time = time.time()
        img_list = sorted(Path(self.img_dir).rglob('cslics*_img.jpg'))
        print(f'Image list gathered in {time.time() - start_time:.2f} seconds')
        
        # Apply max_images limit if specified
        if self.max_images is not None and self.max_images > 0:
            img_list = img_list[:self.max_images]
            print(f'Limited to first {self.max_images} images')
        
        print(f'Number of images found: {len(img_list)}')
        print(f'Submersion time: {self.submersion_time}')

        # Pre-filter images based on mode
        start_time = time.time()
        surface_img_list = []
        subsurface_img_list = []
        
        if self.mode in ["surface", "both"]:
            surface_img_list = [img for img in img_list if self.is_surface_image(img)]
            
        if self.mode in ["subsurface", "both"]:
            subsurface_img_list = [img for img in img_list if not self.is_surface_image(img)]
        
        print(f'Images filtered by mode in {time.time() - start_time:.2f} seconds')
        print(f'Found {len(surface_img_list)} surface images and {len(subsurface_img_list)} subsurface images')
        
        # Initialize results list
        results = []
        start_time = time.time()
        
        # Select processing method based on parallel setting and device
        if self.parallel and torch.cuda.is_available():
            print("Using parallel GPU processing")
            # Process everything in one go with GPU batching
            if self.mode == "both":
                combined_list = surface_img_list + subsurface_img_list
                results = self._process_images_parallel_gpu(combined_list, "Processing images")
            else:
                # Process only the required images
                img_subset = surface_img_list if self.mode == "surface" else subsurface_img_list
                results = self._process_images_parallel_gpu(img_subset, "Processing images")
        elif self.parallel:
            print("Using parallel CPU processing")
            # Process surface images if needed
            if self.mode in ["surface", "both"] and surface_img_list:
                print("\nProcessing surface images:")
                surface_results = self._process_images_parallel(surface_img_list, "Surface images")
                results.extend(surface_results)
            
            # Process subsurface images if needed
            if self.mode in ["subsurface", "both"] and subsurface_img_list:
                print("\nProcessing subsurface images:")
                subsurface_results = self._process_images_parallel(subsurface_img_list, "Subsurface images")
                results.extend(subsurface_results)
        else:
            print("Using sequential processing")
            # Process surface images if needed
            if self.mode in ["surface", "both"] and surface_img_list:
                print("\nProcessing surface images:")
                surface_results = self._process_images_with_progress(surface_img_list, "Surface images")
                results.extend(surface_results)
            
            # Process subsurface images if needed
            if self.mode in ["subsurface", "both"] and subsurface_img_list:
                print("\nProcessing subsurface images:")
                subsurface_results = self._process_images_with_progress(subsurface_img_list, "Subsurface images")
                results.extend(subsurface_results)
        
        # Calculate statistics
        end_time = time.time()
        duration = end_time - start_time
        
        # Count detections more efficiently
        surface_detections = sum(count for count, model_type in results if model_type == "surface")
        subsurface_detections = sum(count for count, model_type in results if model_type == "subsurface")
        
        # Print statistics
        print('\nProcessing complete:')
        processed_count = len(surface_img_list) if self.mode == "surface" else (
            len(subsurface_img_list) if self.mode == "subsurface" else 
            len(surface_img_list) + len(subsurface_img_list)
        )
        print(f'Total images processed: {processed_count}')
        
        if self.mode in ["surface", "both"]:
            surface_count = sum(1 for _, model_type in results if model_type == "surface")
            print(f'Surface model detections: {surface_detections} in {surface_count} images')
        
        if self.mode in ["subsurface", "both"]:
            subsurface_count = sum(1 for _, model_type in results if model_type == "subsurface")
            print(f'Subsurface model detections: {subsurface_detections} in {subsurface_count} images')
        
        print(f'Run time: {duration:.2f} sec ({duration / 60.0:.2f} min, {duration / 3600.0:.2f} hrs)')
        if processed_count > 0:
            print(f'Time per image: {duration / processed_count:.2f} sec')
    
    def _process_images_with_progress(self, img_list, desc):
        """
        Process a list of images with a progress bar.
        
        Args:
            img_list: List of image paths to process
            desc: Description for the progress bar
            
        Returns:
            list: List of (detection_count, model_type) tuples
        """
        results = []
        try:
            for img in tqdm(img_list, desc=desc, unit="img"):
                result = self.process_image(img)
                results.append(result)
        except ImportError:
            # Fallback if tqdm is not installed
            total = len(img_list)
            for i, img in enumerate(img_list):
                if i % 10 == 0 or i == total - 1:
                    progress = (i + 1) / total * 100
                    print(f"{desc}: {progress:.1f}% ({i+1}/{total})", end="\r")
                result = self.process_image(img)
                results.append(result)
            print()  # Add a newline after progress updates
        
        return results

    def _process_images_parallel(self, img_list, desc):
        """
        Process a list of images in parallel with a progress bar.
        
        Args:
            img_list: List of image paths to process
            desc: Description for the progress bar
            
        Returns:
            list: List of (detection_count, model_type) tuples
        """
        results = []
        total = len(img_list)
        
        # Use GPU batching if CUDA is available
        if torch.cuda.is_available():
            print("GPU detected: Using batch processing for parallel GPU inference")
            return self._process_images_parallel_gpu(img_list, desc)
        else:
            # For CPU-only, we can use multiple workers
            max_workers = os.cpu_count()
            print(f"Using {max_workers} parallel workers (CPU-only mode)")
            
            # Multi-worker mode (CPU-only)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Process with separate model instances per thread
                def process_with_new_model(img_path):
                    is_surface = self.is_surface_image(img_path)
                    
                    # Skip processing based on mode
                    if (is_surface and self.mode == "subsurface") or (not is_surface and self.mode == "surface"):
                        return 0, "skipped"
                    
                    # Create a fresh model instance to avoid thread conflicts
                    if is_surface:
                        model = YOLO(self.surface_weights_path, verbose=self.verbose).to('cpu')
                        classes = self.surface_classes
                        class_colours = self.surface_class_colours
                        imgsave_dir = self.surface_imgsave_dir
                        txtsave_dir = self.surface_txtsave_dir
                        model_path = self.surface_weights_path
                        model_type = "surface"
                    else:
                        model = YOLO(self.subsurface_weights_path, verbose=self.verbose).to('cpu')
                        classes = self.subsurface_classes
                        class_colours = self.subsurface_class_colours
                        imgsave_dir = self.subsurface_imgsave_dir
                        txtsave_dir = self.subsurface_txtsave_dir
                        model_path = self.subsurface_weights_path
                        model_type = "subsurface"
                    
                    # Run inference with thread-local model
                    results = model.predict(
                        source=img_path, 
                        iou=self.iou_thresh, 
                        conf=self.conf_thresh, 
                        agnostic_nms=True, 
                        max_det=self.max_det,
                        verbose=self.verbose
                    )
                    
                    # Process results and save outputs (similar to process_image)
                    boxes = results[0].boxes
                    pred = []
                    for b in boxes:
                        xyxyn = b.xyxyn[0] if torch.cuda.is_available() else b.xyxyn
                        pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
                    predictions = torch.tensor(pred)

                    # Save outputs (same as in process_image)
                    rel_path = os.path.relpath(os.path.dirname(img_path), self.img_dir)
                    
                    # Save image predictions
                    if self.save_img:
                        os.makedirs(os.path.join(imgsave_dir, rel_path), exist_ok=True)
                        self.save_image_predictions_bb(predictions, img_path, os.path.join(imgsave_dir, rel_path), classes, class_colours)

                    # Save text and JSON predictions
                    if self.save_txt:
                        os.makedirs(os.path.join(txtsave_dir, rel_path), exist_ok=True)
                        self.save_txt_predictions_bb(predictions, img_path, os.path.join(txtsave_dir, rel_path))
                        self.save_json_predictions_bb(predictions, img_path, os.path.join(txtsave_dir, rel_path), model_path, classes)
                    
                    return len(pred), model_type
                
                # Submit all tasks
                futures = [executor.submit(process_with_new_model, img) for img in img_list]
                
                # Process results as they complete with a progress bar
                try:
                    for future in tqdm(futures, desc=desc, unit="img", total=len(futures)):
                        result = future.result()
                        results.append(result)
                except ImportError:
                    # Fallback if tqdm is not installed
                    completed = 0
                    for future in futures:
                        result = future.result()
                        results.append(result)
                        completed += 1
                        if completed % 10 == 0 or completed == total:
                            progress = completed / total * 100
                            print(f"{desc}: {progress:.1f}% ({completed}/{total})", end="\r")
                    print()  # Add a newline after progress updates
            
            return results  
    
    def _process_images_parallel_gpu(self, img_list, desc):
        """
        Process images in parallel using GPU batching with optimized memory usage.
        """
        results = []
        
        # Pre-filter images by type (more efficient than checking in loop)
        surface_images = []
        subsurface_images = []
        
        # Use list comprehensions for faster filtering
        if self.mode in ["surface", "both"]:
            surface_images = [img for img in img_list if self.is_surface_image(img)]
        
        if self.mode in ["subsurface", "both"]:
            subsurface_images = [img for img in img_list if not self.is_surface_image(img)]
        
        # Calculate optimal batch size based on VRAM
        gpu_mem = torch.cuda.get_device_properties(0).total_memory if torch.cuda.is_available() else 0
        batch_size = min(16, max(1, int(gpu_mem / (1.5 * 10**9))))  # Adjust based on VRAM (1.5GB per batch as estimate)
        print(f"Using batch size of {batch_size} based on available GPU memory")
        
        # Pre-create all required directories to avoid repeated checks
        self._precreate_output_directories(surface_images, self.surface_imgsave_dir, self.surface_txtsave_dir)
        self._precreate_output_directories(subsurface_images, self.subsurface_imgsave_dir, self.subsurface_txtsave_dir)
        
        # Process surface images in batches
        if surface_images:
            print(f"Processing {len(surface_images)} surface images with GPU batching")
            surface_results = self._process_batch(
                surface_images, 
                self.surface_model, 
                self.surface_classes, 
                self.surface_class_colours,
                self.surface_imgsave_dir, 
                self.surface_txtsave_dir,
                self.surface_weights_path,
                "surface",
                batch_size,
                f"{desc} (surface)"
            )
            results.extend(surface_results)
        
        # Process subsurface images in batches
        if subsurface_images:
            print(f"Processing {len(subsurface_images)} subsurface images with GPU batching")
            subsurface_results = self._process_batch(
                subsurface_images, 
                self.subsurface_model, 
                self.subsurface_classes, 
                self.subsurface_class_colours,
                self.subsurface_imgsave_dir, 
                self.subsurface_txtsave_dir,
                self.subsurface_weights_path,
                "subsurface",
                batch_size,
                f"{desc} (subsurface)"
            )
            results.extend(subsurface_results)
        
        return results

    def _precreate_output_directories(self, images, imgsave_dir, txtsave_dir):
        """Create all necessary output directories in advance"""
        needed_dirs = set()
        for img_path in images:
            rel_path = os.path.relpath(os.path.dirname(img_path), self.img_dir)
            if self.save_img:
                needed_dirs.add(os.path.join(imgsave_dir, rel_path))
            if self.save_txt:
                needed_dirs.add(os.path.join(txtsave_dir, rel_path))
        
        # Create all needed directories at once
        for directory in needed_dirs:
            os.makedirs(directory, exist_ok=True)

    def _process_batch(self, images, model, classes, class_colours, imgsave_dir, 
                      txtsave_dir, model_path, model_type, batch_size, desc):
        """Helper method to process batches of images with the same model"""
        results = []
        
        # For batch-level aggregation of detections
        batch_detections = {}
        batch_total = 0
        
        for i in tqdm(range(0, len(images), batch_size), desc=desc, unit="batch"):
            batch = images[i:i+batch_size]
            
            # Run prediction on batch with torch.no_grad() for memory efficiency
            with torch.no_grad():
                batch_results = model.predict(
                    source=batch, 
                    iou=self.iou_thresh, 
                    conf=self.conf_thresh, 
                    agnostic_nms=True, 
                    max_det=self.max_det,
                    verbose=self.verbose
                )
            
            # Process each result in the batch
            for idx, r in enumerate(batch_results):
                img_path = batch[idx]
                boxes = r.boxes
                pred_count = len(boxes)
                
                # Convert predictions to tensor format - IMPORTANT: Missing this
                pred = []
                for b in boxes:
                    xyxyn = b.xyxyn[0]
                    pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
                
                predictions = torch.tensor(pred) if pred else torch.zeros((0, 6))
                
                # Extract timestamp from filename for detection tracking
                filename = Path(img_path).stem
                try:
                    timestamp_str = filename[9:-11]  # Assumes consistent filename format
                    timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                    
                    # Collect detections for batch update
                    iso_timestamp = timestamp.isoformat()
                    if iso_timestamp in batch_detections:
                        batch_detections[iso_timestamp]["count"] += pred_count
                        batch_detections[iso_timestamp]["files"] += 1
                    else:
                        batch_detections[iso_timestamp] = {
                            "count": pred_count, 
                            "files": 1
                        }
                    batch_total += pred_count
                    
                except (ValueError, IndexError):
                    # If timestamp parsing fails, just skip tracking this for plotting
                    pass
                
                # MISSING CODE: Save the actual files
                rel_path = os.path.relpath(os.path.dirname(img_path), self.img_dir)
                
                # Skip empty predictions to avoid unnecessary file operations
                if pred_count == 0 and not self.save_img:
                    results.append((0, model_type))
                    continue
                    
                # Save image predictions
                if self.save_img:
                    img_save_subdir = os.path.join(imgsave_dir, rel_path)
                    self.save_image_predictions_bb(
                        predictions, img_path, img_save_subdir, classes, class_colours
                    )
                
                # Save text and JSON predictions
                if self.save_txt:
                    txt_save_subdir = os.path.join(txtsave_dir, rel_path)
                    self.save_txt_predictions_bb(predictions, img_path, txt_save_subdir)
                    self.save_json_predictions_bb(
                        predictions, img_path, txt_save_subdir, model_path, classes
                    )
                
                results.append((pred_count, model_type))
        
        # Update JSON file once per batch instead of per image
        self._batch_update_detection_data(batch_detections, batch_total, model_type)
        batch_detections = {}  # Reset for next batch
        batch_total = 0
    
        return results

    def _batch_update_detection_data(self, detections_dict, total_count, model_type):
        """
        Update detection data in JSON file with batched updates.
        
        Args:
            detections_dict: Dictionary of timestamp -> {count, files}
            total_count: Total detections in this batch
            model_type: Type of detection ("surface" or "subsurface")
        """
        if total_count <= 0:
            return  # Skip if no detections
            
        _, surface_data_path, subsurface_data_path = self._get_detection_data_paths()
        data_path = surface_data_path if model_type == "surface" else subsurface_data_path
        
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
                    with open(data_path, 'r') as f:
                        data = json.load(f)
                    
                    # Update with batch data
                    data["total_detections"] += total_count
                    
                    for timestamp_str, counts in detections_dict.items():
                        if timestamp_str in data["detections_by_timestamp"]:
                            data["detections_by_timestamp"][timestamp_str]["count"] += counts["count"]
                            data["detections_by_timestamp"][timestamp_str]["files"] += counts["files"]
                        else:
                            data["detections_by_timestamp"][timestamp_str] = counts
                    
                    # Save updated data
                    with open(data_path, 'w') as f:
                        json.dump(data, f, indent=2)
                        
                except (FileNotFoundError, json.JSONDecodeError):
                    # Initialize and try again
                    self._initialize_detection_files()
                    self._batch_update_detection_data(detections_dict, total_count, model_type)
                
        except Exception as e:
            print(f"Error with file locking when batch updating {data_path}: {e}")

    def plot_surface_detections(self, save_path=None):
        """
        Plot time history of surface detections using data from JSON file.
        
        Args:
            save_path: Path to save the plot (if None, will use default path)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        _, surface_data_path, _ = self._get_detection_data_paths()
        
        # Check if data file exists
        if not os.path.exists(surface_data_path):
            print(f"No surface detection data file found at {surface_data_path}")
            return None
        
        # Load detection data
        try:
            with open(surface_data_path, 'r') as f:
                surface_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading surface detection data file: {surface_data_path}")
            return None
        
        # Extract data for plotting
        timestamps = []
        counts = []
        for timestamp_str, data in surface_data["detections_by_timestamp"].items():
            timestamps.append(datetime.fromisoformat(timestamp_str))
            counts.append(data["count"])
        
        if not timestamps:
            print("No surface detection data available for plotting")
            return None
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, counts))
        timestamps = [item[0] for item in sorted_data]
        counts = [item[1] for item in sorted_data]
        
        # Convert to decimal days since submersion time
        days = self.convert_to_decimal_days(timestamps, self.submersion_datetime)
        
        # Create the plot
        return self._create_detection_plot(
            days, counts, "Surface", "blue", 
            surface_data["total_detections"], save_path
        )

    def plot_subsurface_detections(self, save_path=None):
        """
        Plot time history of subsurface detections using data from JSON file.
        
        Args:
            save_path: Path to save the plot (if None, will use default path)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        _, _, subsurface_data_path = self._get_detection_data_paths()
        
        # Check if data file exists
        if not os.path.exists(subsurface_data_path):
            print(f"No subsurface detection data file found at {subsurface_data_path}")
            return None
        
        # Load detection data
        try:
            with open(subsurface_data_path, 'r') as f:
                subsurface_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Error reading subsurface detection data file: {subsurface_data_path}")
            return None
        
        # Extract data for plotting
        timestamps = []
        counts = []
        for timestamp_str, data in subsurface_data["detections_by_timestamp"].items():
            timestamps.append(datetime.fromisoformat(timestamp_str))
            counts.append(data["count"])
        
        if not timestamps:
            print("No subsurface detection data available for plotting")
            return None
        
        # Sort by timestamp
        sorted_data = sorted(zip(timestamps, counts))
        timestamps = [item[0] for item in sorted_data]
        counts = [item[1] for item in sorted_data]
        
        # Convert to decimal days since submersion time
        days = self.convert_to_decimal_days(timestamps, self.submersion_datetime)
        
        # Create the plot
        return self._create_detection_plot(
            days, counts, "Subsurface", "green", 
            subsurface_data["total_detections"], save_path
        )

    def _create_detection_plot(self, days, counts, title_prefix, color, total_objects, save_path=None):
        """Helper method to create detection plots"""
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 7))
        
        # Plot total detections
        ax.plot(days, counts, f'{color}', alpha=0.5, label='Detections per Second')
        
        # Add zero marker for submersion time
        ax.axvline(x=0, color='red', linestyle='--', label='Submersion Time')
        
        # Format axes
        ax.set_xlabel('Days Since CSLICS Submersion')
        ax.set_ylabel('Detections per Second')
        ax.set_title(f'{title_prefix} Detections - {self.cslics_uuid}\nTotal Objects Detected: {total_objects}')
        
        # Add legend
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
        
        # Add text annotation for reference time
        ax.text(0.01, 0.97, f'Submersion time: {self.submersion_time}', 
                transform=ax.transAxes, fontsize=9, va='top')
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Format the plot
        plt.tight_layout()
        
        # Save if path provided or use default
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            # If save_path not specified, use a default path
            default_save_dir = os.path.join(self.save_dir, self.cslics_uuid)
            os.makedirs(default_save_dir, exist_ok=True)
            detection_type = title_prefix.lower()
            default_save_path = os.path.join(
                default_save_dir,
                f"{self.cslics_uuid}_{detection_type}_detections_plot.png"
            )
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {default_save_path}")
        
        return fig

    def plot_all_detections(self, save_path=None):
        """
        Plot time history of both surface and subsurface detections on the same graph.
        
        Args:
            save_path: Path to save the plot (if None, will use default path)
            
        Returns:
            matplotlib.figure.Figure: The generated figure
        """
        _, surface_data_path, subsurface_data_path = self._get_detection_data_paths()
        surface_data = None
        subsurface_data = None
        
        # Load surface data if available
        if os.path.exists(surface_data_path):
            try:
                with open(surface_data_path, 'r') as f:
                    surface_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading surface detection data file: {surface_data_path}")
        
        # Load subsurface data if available
        if os.path.exists(subsurface_data_path):
            try:
                with open(subsurface_data_path, 'r') as f:
                    subsurface_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Error reading subsurface detection data file: {subsurface_data_path}")
        
        # Check if we have any data
        if not surface_data and not subsurface_data:
            print("No detection data files found for plotting")
            return None
        
        # Create plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Plot surface data
        if surface_data and surface_data["detections_by_timestamp"]:
            timestamps = []
            counts = []
            for timestamp_str, data in surface_data["detections_by_timestamp"].items():
                timestamps.append(datetime.fromisoformat(timestamp_str))
                counts.append(data["count"])
            
            # Sort by timestamp
            sorted_data = sorted(zip(timestamps, counts))
            timestamps = [item[0] for item in sorted_data]
            counts = [item[1] for item in sorted_data]
            
            # Convert to decimal days since submersion time
            days = self.convert_to_decimal_days(timestamps, self.submersion_datetime)
            
            # Plot surface data
            ax.plot(days, counts, 'b-', alpha=0.5, 
                    label=f'Surface ({surface_data["total_detections"]} total)')
        
        # Plot subsurface data
        if subsurface_data and subsurface_data["detections_by_timestamp"]:
            timestamps = []
            counts = []
            for timestamp_str, data in subsurface_data["detections_by_timestamp"].items():
                timestamps.append(datetime.fromisoformat(timestamp_str))
                counts.append(data["count"])
            
            # Sort by timestamp
            sorted_data = sorted(zip(timestamps, counts))
            timestamps = [item[0] for item in sorted_data]
            counts = [item[1] for item in sorted_data]
            
            # Convert to decimal days since submersion time
            days = self.convert_to_decimal_days(timestamps, self.submersion_datetime)
            
            # Plot subsurface data
            ax.plot(days, counts, 'g-', alpha=0.5, 
                    label=f'Subsurface ({subsurface_data["total_detections"]} total)')
        
        # Add zero marker for submersion time
        ax.axvline(x=0, color='red', linestyle='--', label='Submersion Time')
        
        # Format axes
        ax.set_xlabel('Days Since Submersion')
        ax.set_ylabel('Detection Count')
        ax.set_title(f'All Detections - {self.cslics_uuid}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add text annotation for reference time
        ax.text(0.01, 0.97, f'Submersion time: {self.submersion_time}', 
                transform=ax.transAxes, fontsize=9, va='top')
        
        # Set y-axis to start at 0
        ax.set_ylim(bottom=0)
        
        # Format the plot
        plt.tight_layout()
        
        # Save if path provided or use default
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")
        else:
            # If save_path not specified, use a default path
            default_save_dir = os.path.join(self.save_dir, self.cslics_uuid)
            os.makedirs(default_save_dir, exist_ok=True)
            default_save_path = os.path.join(
                default_save_dir,
                f"{self.cslics_uuid}_all_detections_plot.png"
            )
            plt.savefig(default_save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {default_save_path}")
        
        return fig
        
    def _get_detection_data_paths(self):
        """Get paths for detection data JSON files"""
        # Create a directory for stats/metadata
        stats_dir = os.path.join(self.save_dir, self.cslics_uuid, "stats")
        os.makedirs(stats_dir, exist_ok=True)
        
        # Define paths for surface and subsurface detection data
        surface_data_path = os.path.join(stats_dir, f"{self.surface_model_id}_detection_data.json")
        subsurface_data_path = os.path.join(stats_dir, f"{self.subsurface_model_id}_detection_data.json")
        
        return stats_dir, surface_data_path, subsurface_data_path

    def _initialize_detection_files(self):
        """Initialize JSON files for storing detection data"""
        _, surface_data_path, subsurface_data_path = self._get_detection_data_paths()
        
        # Initialize surface data file if needed
        if self.mode in ["surface", "both"]:
            surface_data = {
                "cslics_uuid": self.cslics_uuid,
                "model_id": self.surface_model_id,
                "detection_type": "surface",
                "submersion_time": self.submersion_time,
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_detections": 0,
                "detections_by_timestamp": {}
            }
            with open(surface_data_path, 'w') as f:
                json.dump(surface_data, f, indent=2)
        
        # Initialize subsurface data file if needed
        if self.mode in ["subsurface", "both"]:
            subsurface_data = {
                "cslics_uuid": self.cslics_uuid,
                "model_id": self.subsurface_model_id,
                "detection_type": "subsurface",
                "submersion_time": self.submersion_time,
                "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "total_detections": 0,
                "detections_by_timestamp": {}
            }
            with open(subsurface_data_path, 'w') as f:
                json.dump(subsurface_data, f, indent=2)

    def _update_detection_data(self, timestamp_str, count, model_type):
        """
        Update detection data in JSON file.
        
        Args:
            timestamp_str: Timestamp string (ISO format)
            count: Number of detections
            model_type: Type of detection ("surface" or "subsurface")
        """
        if count <= 0:
            return  # Skip empty detections
            
        _, surface_data_path, subsurface_data_path = self._get_detection_data_paths()
        data_path = surface_data_path if model_type == "surface" else subsurface_data_path
        
        # Skip if we're not processing this type
        if (model_type == "surface" and self.mode == "subsurface") or \
           (model_type == "subsurface" and self.mode == "surface"):
            return
            
        try:
            # Load current data
            with open(data_path, 'r') as f:
                data = json.load(f)
                
            # Update the data
            data["total_detections"] += count
            
            if timestamp_str in data["detections_by_timestamp"]:
                data["detections_by_timestamp"][timestamp_str]["count"] += count
                data["detections_by_timestamp"][timestamp_str]["files"] += 1
            else:
                data["detections_by_timestamp"][timestamp_str] = {
                    "count": count,
                    "files": 1
                }
            
            print(f'writing to json file {data_path}')
            # Save updated data
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error updating detection data file {data_path}: {e}")
            # Create a new file if it doesn't exist
            if not os.path.exists(data_path):
                if model_type == "surface":
                    self._initialize_detection_files()
                    self._update_detection_data(timestamp_str, count, model_type)  # Try again
                else:
                    self._initialize_detection_files()
                    self._update_detection_data(timestamp_str, count, model_type)  # Try again

if __name__ == "__main__":
    
    config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/spawn_predictor_20231205_t4_alor_cslics01.json"
    # config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/spawn_predictor_20231205_t4_alor_cslics08_test.json"
    # Initialize predictor with the config file
    predictor = CoralSpawnPredictor(config_file)
    
    # Run the prediction process
    predictor.run()
    
    # Generate and save plots
    if predictor.mode in ["surface", "both"]:
        predictor.plot_surface_detections()
    if predictor.mode in ["subsurface", "both"]:
        predictor.plot_subsurface_detections()
    if predictor.mode == "both":
        predictor.plot_all_detections()
        
    print("Processing complete:")
    print(f'cslics_uuid: {predictor.cslics_uuid}')
    print(f'Surface model: {predictor.surface_model_id}')
    print(f'Subsurface model: {predictor.subsurface_model_id}')
    print(f'config_file: {Path(config_file).stem}')