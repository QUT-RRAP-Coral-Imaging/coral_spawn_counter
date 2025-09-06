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

class CoralSpawnPredictor:
    def __init__(self, config_file):
        """
        Initialize the coral spawn predictor with either a config file or direct parameters.
        
        Args:
            config_file: Path to the JSON configuration file
            **kwargs: Direct parameters (override config file if both provided)
        """
        # Load configuration from the JSON file
        self.config = self.load_config_from_json(config_file)
        
        # Extract required parameters
        self.surface_weights_path = self.config['surface_weights_path']
        self.subsurface_weights_path = self.config['subsurface_weights_path']
        self.img_dir = self.config['img_dir']
        self.save_dir = self.config['save_dir']
        self.cslics_uuid = self.config['cslics_uuid']
        self.submersion_time = self.config['submersion_time']

        # Extract optional parameters with defaults
        self.mode = self.config.get('processing_mode', 'both')
        self.iou_thresh = float(self.config.get('iou_thresh', 0.3))
        self.conf_thresh = float(self.config.get('conf_thresh', 0.25))
        self.max_det = int(self.config.get('max_det', 1000))
        self.save_img = self.config.get('save_img', True)
        self.save_txt = self.config.get('save_txt', True)
        self.save_txt_bb = self.config.get('save_txt_bb', False)
        self.max_images = int(self.config.get('max_images', 0)) if self.config.get('max_images') else None
        self.verbose = self._parse_bool(self.config.get('verbose', False))
        self.parallel = self._parse_bool(self.config.get('parallel', False))
        
        # Validate required parameters
        self._validate_params()
        
        # Validate processing mode
        valid_modes = ["surface", "subsurface", "both"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {self.mode}. Must be one of {valid_modes}")
        
        # Extract model IDs (stem names)
        self.surface_model_id = Path(self.surface_weights_path).stem
        self.subsurface_model_id = Path(self.subsurface_weights_path).stem

        # Parse submersion time
        self.submersion_datetime = datetime.strptime(self.submersion_time, "%Y-%m-%d_%H-%M-%S")
        
        # Timestamp for detection runs
        self.current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Initialize models and load class information
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Only load needed models based on mode
        if self.mode in ["surface", "both"]:
            print(f'Loading surface model: {self.surface_weights_path}')
            self.surface_model = YOLO(self.surface_weights_path, verbose=self.verbose).to(self.device)
            self.surface_classes = self._extract_classes_from_model(self.surface_model)
            self.surface_class_colours = self._generate_class_colors(self.surface_classes)
            print(f'Surface model classes: {self.surface_classes}')
        else:
            self.surface_model = None
            
        if self.mode in ["subsurface", "both"]:
            print(f'Loading subsurface model: {self.subsurface_weights_path}')
            self.subsurface_model = YOLO(self.subsurface_weights_path, verbose=self.verbose).to(self.device)
            self.subsurface_classes = self._extract_classes_from_model(self.subsurface_model)
            self.subsurface_class_colours = self._generate_class_colors(self.subsurface_classes)
            print(f'Subsurface model classes: {self.subsurface_classes}')
        else:
            self.subsurface_model = None

        # Prepare output directories
        self._prepare_output_directories()
        
        # Print configuration summary
        self._print_config_summary()
    
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
    
    def _parse_bool(self, value):
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
        Determine if an image was captured before submersion time.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            bool: True if the image is a surface image, False otherwise
        """
        # Extract datetime from filename
        filename = Path(img_path).stem
        # Assuming the first part of the filename is a timestamp in format YYYY-MM-DD_HH-MM-SS
        try:
            timestamp_str = filename[9:-11]
            # print(f"Extracted timestamp: {timestamp_str} from filename: {filename}")
            image_datetime = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            return image_datetime < self.submersion_datetime
        except (ValueError, IndexError):
            # If datetime parsing fails, assume it's a surface image
            print(f"Warning: Could not parse datetime from filename: {filename}")
            return True
    
    def process_image(self, img_name):
        """
        Process a single image: determine if it's surface or subsurface,
        run inference with the appropriate model, save predictions.
        
        Args:
            img_name: Path to the image file
            
        Returns:
            tuple: (detection_count, model_type) or None if skipped
        """
        is_surface = self.is_surface_image(img_name)
        
        # Skip processing based on mode
        if (is_surface and self.mode == "subsurface") or (not is_surface and self.mode == "surface"):
            return 0, "skipped"
        
        # Select appropriate model, classes, colors, and save directories
        if is_surface:
            model = self.surface_model
            classes = self.surface_classes
            class_colours = self.surface_class_colours
            imgsave_dir = self.surface_imgsave_dir
            txtsave_dir = self.surface_txtsave_dir
            model_path = self.surface_weights_path
            model_type = "surface"
        else:
            model = self.subsurface_model
            classes = self.subsurface_classes
            class_colours = self.subsurface_class_colours
            imgsave_dir = self.subsurface_imgsave_dir
            txtsave_dir = self.subsurface_txtsave_dir
            model_path = self.subsurface_weights_path
            model_type = "subsurface"
        
        # Run inference
        results = model.predict(
            source=img_name, 
            iou=self.iou_thresh, 
            conf=self.conf_thresh, 
            agnostic_nms=True, 
            max_det=self.max_det,
            verbose=self.verbose  # Add this parameter
        )
        
        boxes = results[0].boxes
        pred = []
        for b in boxes:
            if torch.cuda.is_available():
                xyxyn = b.xyxyn[0]
                pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
        predictions = torch.tensor(pred)

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
        Run prediction on all images in the directory with separate progress bars
        for surface and subsurface images.
        """
        print(f'Fetching image list in all subfolders from: {self.img_dir}')
        print(f'Processing mode: {self.mode}')
        print(f'Parallel processing: {"Enabled" if self.parallel else "Disabled"}')
        img_list = sorted(Path(self.img_dir).rglob('cslics*_img.jpg'))
        
        # Apply max_images limit if specified
        if self.max_images is not None and self.max_images > 0:
            img_list = img_list[:self.max_images]
            print(f'Limited to first {self.max_images} images')
        
        print(f'Number of images found: {len(img_list)}')
        print(f'Submersion time: {self.submersion_time}')

        # Separate images into surface and subsurface lists
        surface_img_list = [img for img in img_list if self.is_surface_image(img)]
        subsurface_img_list = [img for img in img_list if not self.is_surface_image(img)]
        
        # Report counts
        print(f'Found {len(surface_img_list)} surface images and {len(subsurface_img_list)} subsurface images')
        
        # Initialize results list
        results = []
        start_time = time.time()
        
        # Select processing method based on parallel setting
        process_method = self._process_images_parallel if self.parallel else self._process_images_with_progress
        
        # Process surface images if needed
        if self.mode in ["surface", "both"] and surface_img_list:
            print("\nProcessing surface images:")
            surface_results = process_method(surface_img_list, "Surface images")
            results.extend(surface_results)
        
        # Process subsurface images if needed
        if self.mode in ["subsurface", "both"] and subsurface_img_list:
            print("\nProcessing subsurface images:")
            subsurface_results = process_method(subsurface_img_list, "Subsurface images")
            results.extend(subsurface_results)
        
        end_time = time.time()
        duration = end_time - start_time

        # Count detections
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
        Process a list of images in parallel using GPU batching.
        
        Args:
            img_list: List of image paths to process
            desc: Description for the progress bar
            
        Returns:
            list: List of (detection_count, model_type) tuples
        """
        results = []
        
        # Group images by type (surface/subsurface)
        surface_images = []
        subsurface_images = []
        
        for img_path in img_list:
            is_surface = self.is_surface_image(img_path)
            if is_surface and self.mode in ["surface", "both"]:
                surface_images.append(img_path)
            elif not is_surface and self.mode in ["subsurface", "both"]:
                subsurface_images.append(img_path)
        
        # Process surface images in batches
        if surface_images:
            print(f"Processing {len(surface_images)} surface images with GPU batching")
            batch_size = 8  # You can adjust this based on your GPU memory
            
            for i in tqdm(range(0, len(surface_images), batch_size), desc=f"{desc} (surface)", unit="batch"):
                batch = surface_images[i:i+batch_size]
                
                # Run prediction on batch
                batch_results = self.surface_model.predict(
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
                    pred = []
                    
                    for b in boxes:
                        xyxyn = b.xyxyn[0]
                        pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
                    
                    predictions = torch.tensor(pred)
                    
                    # Save outputs
                    rel_path = os.path.relpath(os.path.dirname(img_path), self.img_dir)
                    
                    # Save image predictions
                    if self.save_img:
                        os.makedirs(os.path.join(self.surface_imgsave_dir, rel_path), exist_ok=True)
                        self.save_image_predictions_bb(
                            predictions, 
                            img_path, 
                            os.path.join(self.surface_imgsave_dir, rel_path),
                            self.surface_classes, 
                            self.surface_class_colours
                        )
                    
                    # Save text and JSON predictions
                    if self.save_txt:
                        os.makedirs(os.path.join(self.surface_txtsave_dir, rel_path), exist_ok=True)
                        self.save_txt_predictions_bb(predictions, img_path, os.path.join(self.surface_txtsave_dir, rel_path))
                        self.save_json_predictions_bb(
                            predictions, 
                            img_path, 
                            os.path.join(self.surface_txtsave_dir, rel_path),
                            self.surface_weights_path, 
                            self.surface_classes
                        )
                
                results.append((len(pred), "surface"))
        
        # Process subsurface images in batches
        if subsurface_images:
            print(f"Processing {len(subsurface_images)} subsurface images with GPU batching")
            batch_size = 8  # You can adjust this based on your GPU memory
            
            for i in tqdm(range(0, len(subsurface_images), batch_size), desc=f"{desc} (subsurface)", unit="batch"):
                batch = subsurface_images[i:i+batch_size]
                
                # Run prediction on batch
                batch_results = self.subsurface_model.predict(
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
                    pred = []
                    
                    for b in boxes:
                        xyxyn = b.xyxyn[0]
                        pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
                    
                    predictions = torch.tensor(pred)
                    
                    # Save outputs
                    rel_path = os.path.relpath(os.path.dirname(img_path), self.img_dir)
                    
                    # Save image predictions
                    if self.save_img:
                        os.makedirs(os.path.join(self.subsurface_imgsave_dir, rel_path), exist_ok=True)
                        self.save_image_predictions_bb(
                            predictions, 
                            img_path, 
                            os.path.join(self.subsurface_imgsave_dir, rel_path),
                            self.subsurface_classes, 
                            self.subsurface_class_colours
                        )
                    
                    # Save text and JSON predictions
                    if self.save_txt:
                        os.makedirs(os.path.join(self.subsurface_txtsave_dir, rel_path), exist_ok=True)
                        self.save_txt_predictions_bb(predictions, img_path, os.path.join(self.subsurface_txtsave_dir, rel_path))
                        self.save_json_predictions_bb(
                            predictions, 
                            img_path, 
                            os.path.join(self.subsurface_txtsave_dir, rel_path),
                            self.subsurface_weights_path, 
                            self.subsurface_classes
                        )
                
                results.append((len(pred), "subsurface"))
        
        return results

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Coral Spawn Predictor')
    parser.add_argument('--config', type=str, help='Path to config file')
    args = parser.parse_args()
    
    # Default config file path if not specified
    if not args.config:
        args.config = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/spawn_predictor_20231205_t4_alor_cslics08.json"

    predictor = CoralSpawnPredictor(config_file=args.config)    
    predictor.run()
    
    print("Processing complete")