#!/usr/bin/env python3

import os
import json
import time
import numpy as np
from pathlib import Path
from datetime import datetime
import torch
import cv2 as cv
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


class CoralSpawnPredictor:
    def __init__(
        self, 
        surface_weights_path, 
        subsurface_weights_path,
        img_dir, 
        save_dir, 
        submersion_time,  # Format: "YYYY-MM-DD_HH-MM-SS"
        mode="both",      # "surface", "subsurface", or "both"
        iou_thresh=0.3, 
        conf_thresh=0.25, 
        max_det=1000, 
        save_img=True, 
        save_txt=True,
        save_txt_bb=False,
        max_images=None,
        verbose=False     # Add this parameter to control verbosity
    ):
        """
        Initialize the coral spawn predictor with two models: one for surface and one for subsurface.
        
        Args:
            surface_weights_path: Path to the surface detection model weights
            subsurface_weights_path: Path to the subsurface detection model weights
            img_dir: Directory containing input images
            save_dir: Base directory to save detection results
            submersion_time: Time string in format "YYYY-MM-DD_HH-MM-SS" dividing surface/subsurface
            mode: Which images to process - "surface", "subsurface", or "both" (default)
            iou_thresh: IoU threshold for non-maximum suppression
            conf_thresh: Confidence threshold for detections
            max_det: Maximum number of detections per image
            save_img: Whether to save images with bounding boxes
            save_txt: Whether to save detection results as text/json
            save_txt_bb: Whether to save detection results as bounding box text format
            max_images: Maximum number of images to process
        """
        # Validate processing mode
        valid_modes = ["surface", "subsurface", "both"]
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        
        self.mode = mode
        
        self.verbose = verbose  # Store verbosity setting
        
        # Model paths
        self.surface_weights_path = surface_weights_path
        self.subsurface_weights_path = subsurface_weights_path
        
        # Directory paths
        self.img_dir = img_dir
        self.save_dir = save_dir
        
        # Create separate directories for surface and subsurface detections
        self.surface_save_dir = os.path.join(save_dir, 'detections_surface')
        self.subsurface_save_dir = os.path.join(save_dir, 'detections_subsurface')
        
        # Submersion time as a dividing point
        self.submersion_time = submersion_time
        self.submersion_datetime = datetime.strptime(submersion_time, "%Y-%m-%d_%H-%M-%S")
        
        # Detection parameters
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.max_det = max_det
        self.save_img = save_img
        self.save_txt = save_txt
        self.save_txt_bb = save_txt_bb
        self.max_images = max_images
        self.current_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Initialize models and load class information
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Only load needed models based on mode
        if self.mode in ["surface", "both"]:
            print(f'Loading surface model: {surface_weights_path}')
            self.surface_model = YOLO(surface_weights_path, verbose=self.verbose).to(self.device)
            self.surface_classes = self._extract_classes_from_model(self.surface_model)
            self.surface_class_colours = self._generate_class_colors(self.surface_classes)
            print(f'Surface model classes: {self.surface_classes}')
        else:
            self.surface_model = None
            
        if self.mode in ["subsurface", "both"]:
            print(f'Loading subsurface model: {subsurface_weights_path}')
            self.subsurface_model = YOLO(subsurface_weights_path, verbose=self.verbose).to(self.device)
            self.subsurface_classes = self._extract_classes_from_model(self.subsurface_model)
            self.subsurface_class_colours = self._generate_class_colors(self.subsurface_classes)
            print(f'Subsurface model classes: {self.subsurface_classes}')
        else:
            self.subsurface_model = None

        
        
        # Prepare output directories
        self._prepare_output_directories()

    def _prepare_output_directories(self):
        """Create output directories for both models."""
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

    def run(self):
        """
        Run prediction on all images in the directory with separate progress bars
        for surface and subsurface images.
        """
        print(f'Fetching image list in all subfolders from: {self.img_dir}')
        print(f'Processing mode: {self.mode}')
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
        

# Example usage
if __name__ == "__main__":
    # Configuration
    surface_weights_path = '/home/dtsai/Data/cslics_datasets/models/cslics_20230905_yolov8n_640p_amtenuis1000.pt'
    subsurface_weights_path = '/home/dtsai/Data/cslics_datasets/models/cslics_subsurface_20250205_640p_yolov8n.pt'
    img_dir = '/home/dtsai/Data/cslics_datasets/icra2025/feature_development/images'
    save_dir = '/home/dtsai/Data/cslics_datasets/icra2025/feature_development/output'
    
    # Define submersion time (when camera was submerged)
    submersion_time = "2023-12-05_23-45-00"  # Format: YYYY-MM-DD_HH-MM-SS
    
    # Choose processing mode: "surface", "subsurface", or "both"
    processing_mode = "both"
    
    # Initialize predictor with the specified mode
    predictor = CoralSpawnPredictor(
        surface_weights_path=surface_weights_path,
        subsurface_weights_path=subsurface_weights_path,
        img_dir=img_dir,
        save_dir=save_dir,
        submersion_time=submersion_time,
        mode=processing_mode,  # Specify which images to process
        iou_thresh=0.3,
        conf_thresh=0.25,
        max_images=200000,
        verbose=False  # Set to True for detailed model output
    )
    
    # Run prediction
    predictor.run()