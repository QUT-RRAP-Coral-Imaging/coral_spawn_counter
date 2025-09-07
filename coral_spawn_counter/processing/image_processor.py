# processing/image_processor.py
import os
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm


class ImageProcessor:
    """Handles sequential processing of individual images."""
    
    def __init__(self, config, model_manager, file_manager, detection_data_manager, time_utils):
        """
        Initialize the image processor.
        
        Args:
            config: ConfigManager instance
            model_manager: ModelManager instance
            file_manager: FileManager instance
            detection_data_manager: DetectionDataManager instance
            time_utils: TimeUtils instance
        """
        self.config = config
        self.model_manager = model_manager
        self.file_manager = file_manager
        self.detection_data_manager = detection_data_manager
        self.time_utils = time_utils
        
        # Cache frequently used config values
        self.mode = config.mode
        self.iou_thresh = config.iou_thresh
        self.conf_thresh = config.conf_thresh
        self.max_det = config.max_det
        self.verbose = config.verbose
        self.save_img = config.save_img
        self.save_txt = config.save_txt
        self.save_txt_bb = config.save_txt_bb
        self.img_dir = config.img_dir
    
    def process_image(self, img_path):
        """
        Process a single image and return detection results.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            tuple: (detection_count, model_type) or (0, "skipped") if skipped
        """
        # Determine if this is a surface or subsurface image
        is_surface = self.time_utils.is_surface_image(img_path)
        
        # Skip processing based on mode
        if (is_surface and self.mode == "subsurface") or (not is_surface and self.mode == "surface"):
            return 0, "skipped"
        
        # Determine model type for this image
        model_type = "surface" if is_surface else "subsurface"
        
        # Get the appropriate model and related information
        try:
            model, classes, class_colours, model_path = self.model_manager.get_model_for_image(is_surface)
        except ValueError as e:
            print(f"Error getting model for image {img_path}: {e}")
            return 0, "error"
        
        # Get output directories
        if is_surface:
            imgsave_dir = self.file_manager.surface_imgsave_dir
            txtsave_dir = self.file_manager.surface_txtsave_dir
        else:
            imgsave_dir = self.file_manager.subsurface_imgsave_dir
            txtsave_dir = self.file_manager.subsurface_txtsave_dir
        
        # Run inference
        try:
            results = model.predict(
                source=img_path,
                iou=self.iou_thresh,
                conf=self.conf_thresh,
                agnostic_nms=True,
                max_det=self.max_det,
                verbose=self.verbose
            )
        except Exception as e:
            print(f"Error during inference on {img_path}: {e}")
            return 0, "error"
        
        # Extract predictions
        boxes = results[0].boxes
        pred = []
        for b in boxes:
            if torch.cuda.is_available():
                xyxyn = b.xyxyn[0]
            else:
                xyxyn = b.xyxyn[0] if hasattr(b, 'xyxyn') else b.xyxy[0]
            pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
        
        predictions = torch.tensor(pred) if pred else torch.zeros((0, 6))
        detection_count = len(pred)
        
        # Update detection data for plotting
        self._update_detection_tracking(img_path, detection_count, model_type)
        
        # Save outputs if requested
        self._save_outputs(predictions, img_path, imgsave_dir, txtsave_dir, 
                          model_path, classes, class_colours)
        
        return detection_count, model_type
    
    def _update_detection_tracking(self, img_path, detection_count, model_type):
        """Update detection data for plotting purposes."""
        if detection_count <= 0:
            return
        
        # Extract timestamp from filename
        timestamp = self.time_utils.extract_timestamp_from_filename(img_path)
        if timestamp is None:
            return
        
        # Update detection data manager
        iso_timestamp = timestamp.isoformat()
        self.detection_data_manager.update_detection_data(iso_timestamp, detection_count, model_type)
    
    def _save_outputs(self, predictions, img_path, imgsave_dir, txtsave_dir, 
                     model_path, classes, class_colours):
        """Save image and text outputs if requested."""
        # Determine relative path for saving
        rel_path = os.path.relpath(os.path.dirname(img_path), self.img_dir)
        
        # Skip empty predictions to avoid unnecessary file operations
        if len(predictions) == 0 and not self.save_img:
            return
        
        # Save image predictions
        if self.save_img:
            img_save_dir = os.path.join(imgsave_dir, rel_path)
            os.makedirs(img_save_dir, exist_ok=True)
            self.file_manager.save_image_predictions_bb(
                predictions, img_path, img_save_dir, classes, class_colours
            )
        
        # Save text and JSON predictions
        if self.save_txt:
            txt_save_dir = os.path.join(txtsave_dir, rel_path)
            os.makedirs(txt_save_dir, exist_ok=True)
            self.file_manager.save_txt_predictions_bb(predictions, img_path, txt_save_dir)
            self.file_manager.save_json_predictions_bb(
                predictions, img_path, txt_save_dir, model_path, classes
            )
        
        # Save bounding box text format if enabled
        if self.save_txt_bb:
            txt_save_dir = os.path.join(txtsave_dir, rel_path)
            os.makedirs(txt_save_dir, exist_ok=True)
            bb_txt_save_path = os.path.join(txt_save_dir, os.path.basename(img_path)[:-4] + '_det_bb.txt')
            with open(bb_txt_save_path, "w") as file:
                for p in predictions:
                    x1, y1, x2, y2 = p[0:4].tolist()
                    conf = p[4]
                    cls = int(p[5])
                    line = f"{x1} {y1} {x2} {y2} {conf}\n"
                    file.write(line)
    
    def process_images_with_progress(self, img_list, desc):
        """
        Process a list of images sequentially with a progress bar.
        
        Args:
            img_list: List of image paths to process
            desc: Description for the progress bar
            
        Returns:
            list: List of (detection_count, model_type) tuples
        """
        results = []
        
        try:
            # Use tqdm for progress bar if available
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
    
    def process_images_by_type(self, img_list, force_type=None):
        """
        Process images, optionally forcing them to be treated as a specific type.
        
        Args:
            img_list: List of image paths to process
            force_type: If specified, treat all images as this type ("surface" or "subsurface")
            
        Returns:
            list: List of (detection_count, model_type) tuples
        """
        results = []
        
        for img_path in img_list:
            if force_type:
                # Override the automatic detection
                is_surface = (force_type == "surface")
            else:
                # Use automatic detection based on timestamp
                is_surface = self.time_utils.is_surface_image(img_path)
            
            # Skip processing based on mode
            if (is_surface and self.mode == "subsurface") or (not is_surface and self.mode == "surface"):
                results.append((0, "skipped"))
                continue
            
            result = self.process_image(img_path)
            results.append(result)
        
        return results
    
    def validate_image(self, img_path):
        """
        Validate that an image file exists and is readable.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            bool: True if the image is valid, False otherwise
        """
        try:
            # Check if file exists
            if not os.path.exists(img_path):
                return False
            
            # Check if it has a valid image extension
            valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
            if not any(str(img_path).lower().endswith(ext) for ext in valid_extensions):
                return False
            
            # Try to determine if it's a surface/subsurface image (validates filename format)
            self.time_utils.is_surface_image(img_path)
            
            return True
        except Exception:
            return False
    
    def filter_valid_images(self, img_list):
        """
        Filter a list of images to only include valid ones.
        
        Args:
            img_list: List of image paths
            
        Returns:
            tuple: (valid_images, invalid_images)
        """
        valid_images = []
        invalid_images = []
        
        for img_path in img_list:
            if self.validate_image(img_path):
                valid_images.append(img_path)
            else:
                invalid_images.append(img_path)
        
        if invalid_images and self.verbose:
            print(f"Warning: Found {len(invalid_images)} invalid images")
            for invalid in invalid_images[:5]:  # Show first 5
                print(f"  Invalid: {invalid}")
            if len(invalid_images) > 5:
                print(f"  ... and {len(invalid_images) - 5} more")
        
        return valid_images, invalid_images
    
    def get_processing_stats(self, results):
        """
        Calculate processing statistics from results.
        
        Args:
            results: List of (detection_count, model_type) tuples
            
        Returns:
            dict: Statistics dictionary
        """
        stats = {
            'total_processed': len([r for r in results if r[1] != "skipped" and r[1] != "error"]),
            'total_skipped': len([r for r in results if r[1] == "skipped"]),
            'total_errors': len([r for r in results if r[1] == "error"]),
            'surface_images': len([r for r in results if r[1] == "surface"]),
            'subsurface_images': len([r for r in results if r[1] == "subsurface"]),
            'surface_detections': sum(count for count, model_type in results if model_type == "surface"),
            'subsurface_detections': sum(count for count, model_type in results if model_type == "subsurface"),
            'total_detections': sum(count for count, model_type in results if model_type in ["surface", "subsurface"])
        }
        
        # Calculate averages
        if stats['surface_images'] > 0:
            stats['avg_surface_detections'] = stats['surface_detections'] / stats['surface_images']
        else:
            stats['avg_surface_detections'] = 0
            
        if stats['subsurface_images'] > 0:
            stats['avg_subsurface_detections'] = stats['subsurface_detections'] / stats['subsurface_images']
        else:
            stats['avg_subsurface_detections'] = 0
        
        return stats
    
    def print_processing_summary(self, results, duration):
        """
        Print a summary of processing results.
        
        Args:
            results: List of (detection_count, model_type) tuples
            duration: Processing duration in seconds
        """
        stats = self.get_processing_stats(results)
        
        print('\nImage Processing Summary:')
        print(f'Total images processed: {stats["total_processed"]}')
        
        if stats['total_skipped'] > 0:
            print(f'Images skipped: {stats["total_skipped"]}')
        
        if stats['total_errors'] > 0:
            print(f'Errors encountered: {stats["total_errors"]}')
        
        if self.mode in ["surface", "both"] and stats['surface_images'] > 0:
            print(f'Surface images: {stats["surface_images"]} '
                  f'(avg {stats["avg_surface_detections"]:.1f} detections per image)')
            print(f'Total surface detections: {stats["surface_detections"]}')
        
        if self.mode in ["subsurface", "both"] and stats['subsurface_images'] > 0:
            print(f'Subsurface images: {stats["subsurface_images"]} '
                  f'(avg {stats["avg_subsurface_detections"]:.1f} detections per image)')
            print(f'Total subsurface detections: {stats["subsurface_detections"]}')
        
        print(f'Total detections: {stats["total_detections"]}')
        print(f'Processing time: {duration:.2f} sec ({duration / 60.0:.2f} min)')
        
        if stats['total_processed'] > 0:
            print(f'Time per image: {duration / stats["total_processed"]:.2f} sec')