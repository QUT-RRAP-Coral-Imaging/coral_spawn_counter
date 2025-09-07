# processing/batch_processor.py
import os
import torch
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from ultralytics import YOLO


class BatchProcessor:
    """Handles batch processing of images for improved efficiency."""
    
    def __init__(self, config, model_manager, file_manager, detection_data_manager, time_utils):
        """
        Initialize the batch processor.
        
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
        
        # Calculate optimal batch size based on available memory
        self.batch_size = self._calculate_optimal_batch_size()
    
    def _calculate_optimal_batch_size(self):
        """Calculate optimal batch size based on available GPU memory."""
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory
            # Estimate 1.5GB per image in batch as rough heuristic
            batch_size = min(32, max(1, int(gpu_mem / (1.5 * 10**9))))
        else:
            # For CPU processing, use smaller batches
            batch_size = min(8, max(1, os.cpu_count() // 2))
        
        return batch_size
    
    def process_images_parallel_gpu(self, img_list, desc):
        """
        Process images in parallel using GPU batching with optimized memory usage.
        
        Args:
            img_list: List of image paths to process
            desc: Description for progress bar
            
        Returns:
            list: List of (detection_count, model_type) tuples
        """
        results = []
        
        # Pre-filter images by type (more efficient than checking in loop)
        surface_images = []
        subsurface_images = []
        
        # Use list comprehensions for faster filtering
        if self.mode in ["surface", "both"]:
            surface_images = [img for img in img_list if self.time_utils.is_surface_image(img)]
        
        if self.mode in ["subsurface", "both"]:
            subsurface_images = [img for img in img_list if not self.time_utils.is_surface_image(img)]
        
        print(f"Using batch size of {self.batch_size} based on available GPU memory")
        
        # Pre-create all required directories to avoid repeated checks
        self._precreate_output_directories(surface_images, 
                                         self.file_manager.surface_imgsave_dir, 
                                         self.file_manager.surface_txtsave_dir)
        self._precreate_output_directories(subsurface_images, 
                                         self.file_manager.subsurface_imgsave_dir, 
                                         self.file_manager.subsurface_txtsave_dir)
        
        # Process surface images in batches
        if surface_images:
            print(f"Processing {len(surface_images)} surface images with GPU batching")
            surface_model_info = self.model_manager.get_model_info("surface")
            surface_results = self._process_batch(
                surface_images, 
                surface_model_info["model"],
                surface_model_info["classes"],
                surface_model_info["class_colours"],
                self.file_manager.surface_imgsave_dir, 
                self.file_manager.surface_txtsave_dir,
                surface_model_info["model_path"],
                "surface",
                self.batch_size,
                f"{desc} (surface)"
            )
            results.extend(surface_results)
        
        # Process subsurface images in batches
        if subsurface_images:
            print(f"Processing {len(subsurface_images)} subsurface images with GPU batching")
            subsurface_model_info = self.model_manager.get_model_info("subsurface")
            subsurface_results = self._process_batch(
                subsurface_images, 
                subsurface_model_info["model"],
                subsurface_model_info["classes"],
                subsurface_model_info["class_colours"],
                self.file_manager.subsurface_imgsave_dir, 
                self.file_manager.subsurface_txtsave_dir,
                subsurface_model_info["model_path"],
                "subsurface",
                self.batch_size,
                f"{desc} (subsurface)"
            )
            results.extend(subsurface_results)
        
        return results
    
    def process_images_parallel(self, img_list, desc):
        """
        Process a list of images in parallel with CPU workers.
        
        Args:
            img_list: List of image paths to process
            desc: Description for the progress bar
            
        Returns:
            list: List of (detection_count, model_type) tuples
        """
        # Use GPU batching if CUDA is available
        if torch.cuda.is_available():
            print("GPU detected: Using batch processing for parallel GPU inference")
            return self.process_images_parallel_gpu(img_list, desc)
        
        # CPU-only parallel processing
        max_workers = min(os.cpu_count(), 8)  # Limit to prevent resource exhaustion
        print(f"Using {max_workers} parallel workers (CPU-only mode)")
        
        results = []
        
        def process_with_new_model(img_path):
            """Process a single image with a thread-local model instance."""
            is_surface = self.time_utils.is_surface_image(img_path)
            
            # Skip processing based on mode
            if (is_surface and self.mode == "subsurface") or (not is_surface and self.mode == "surface"):
                return 0, "skipped"
            
            model_type = "surface" if is_surface else "subsurface"
            
            # Create a fresh model instance to avoid thread conflicts
            if is_surface:
                model = YOLO(self.config.surface_weights_path, verbose=self.verbose).to('cpu')
                classes = self.model_manager.surface_classes
                class_colours = self.model_manager.surface_class_colours
                imgsave_dir = self.file_manager.surface_imgsave_dir
                txtsave_dir = self.file_manager.surface_txtsave_dir
                model_path = self.config.surface_weights_path
            else:
                model = YOLO(self.config.subsurface_weights_path, verbose=self.verbose).to('cpu')
                classes = self.model_manager.subsurface_classes
                class_colours = self.model_manager.subsurface_class_colours
                imgsave_dir = self.file_manager.subsurface_imgsave_dir
                txtsave_dir = self.file_manager.subsurface_txtsave_dir
                model_path = self.config.subsurface_weights_path
            
            # Run inference with thread-local model
            try:
                results = model.predict(
                    source=img_path, 
                    iou=self.iou_thresh, 
                    conf=self.conf_thresh, 
                    agnostic_nms=True, 
                    max_det=self.max_det,
                    verbose=self.verbose
                )
                
                # Process results and save outputs
                boxes = results[0].boxes
                pred = []
                for b in boxes:
                    xyxyn = b.xyxyn[0] if hasattr(b, 'xyxyn') else b.xyxy[0]
                    pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
                
                predictions = torch.tensor(pred) if pred else torch.zeros((0, 6))
                
                # Update detection tracking
                self._update_detection_tracking(img_path, len(pred), model_type)
                
                # Save outputs
                self._save_outputs(predictions, img_path, imgsave_dir, txtsave_dir, 
                                 model_path, classes, class_colours)
                
                return len(pred), model_type
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                return 0, "error"
        
        # Submit all tasks and process results with progress bar
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_with_new_model, img) for img in img_list]
            
            try:
                for future in tqdm(futures, desc=desc, unit="img", total=len(futures)):
                    result = future.result()
                    results.append(result)
            except ImportError:
                # Fallback if tqdm is not installed
                total = len(futures)
                completed = 0
                for future in futures:
                    result = future.result()
                    results.append(result)
                    completed += 1
                    if completed % 10 == 0 or completed == total:
                        progress = completed / total * 100
                        print(f"{desc}: {progress:.1f}% ({completed}/{total})", end="\r")
                print()
        
        return results
    
    def _process_batch(self, images, model, classes, class_colours, imgsave_dir, 
                      txtsave_dir, model_path, model_type, batch_size, desc):
        """
        Helper method to process batches of images with the same model.
        
        Args:
            images: List of image paths
            model: YOLO model instance
            classes: List/dict of class names
            class_colours: Dict of class colors
            imgsave_dir: Directory for saving annotated images
            txtsave_dir: Directory for saving detection files
            model_path: Path to the model file
            model_type: Type of model ("surface" or "subsurface")
            batch_size: Number of images per batch
            desc: Description for progress bar
            
        Returns:
            list: List of (detection_count, model_type) tuples
        """
        results = []
        
        # For batch-level aggregation of detections
        batch_detections = {}
        batch_total = 0
        
        for i in tqdm(range(0, len(images), batch_size), desc=desc, unit="batch"):
            batch = images[i:i+batch_size]
            
            # Run prediction on batch with torch.no_grad() for memory efficiency
            with torch.no_grad():
                try:
                    batch_results = model.predict(
                        source=batch, 
                        iou=self.iou_thresh, 
                        conf=self.conf_thresh, 
                        agnostic_nms=True, 
                        max_det=self.max_det,
                        verbose=self.verbose
                    )
                except Exception as e:
                    print(f"Error during batch prediction: {e}")
                    # Add error results for this batch
                    for _ in batch:
                        results.append((0, "error"))
                    continue
            
            # Process each result in the batch
            for idx, r in enumerate(batch_results):
                img_path = batch[idx]
                
                try:
                    boxes = r.boxes
                    pred_count = len(boxes) if boxes is not None else 0
                    
                    # Convert predictions to tensor format
                    pred = []
                    if boxes is not None:
                        for b in boxes:
                            xyxyn = b.xyxyn[0]
                            pred.append([xyxyn[0], xyxyn[1], xyxyn[2], xyxyn[3], b.conf, b.cls])
                    
                    predictions = torch.tensor(pred) if pred else torch.zeros((0, 6))
                    
                    # Extract timestamp from filename for detection tracking
                    self._update_batch_detection_tracking(img_path, pred_count, batch_detections)
                    batch_total += pred_count
                    
                    # Save outputs
                    self._save_batch_outputs(predictions, img_path, imgsave_dir, txtsave_dir, 
                                           model_path, classes, class_colours, pred_count, model_type)
                    
                    results.append((pred_count, model_type))
                    
                except Exception as e:
                    print(f"Error processing image {img_path} in batch: {e}")
                    results.append((0, "error"))
            
            # Update JSON file once per batch instead of per image
            if batch_detections:
                self.detection_data_manager.batch_update_detection_data(
                    batch_detections, batch_total, model_type
                )
                batch_detections = {}
                batch_total = 0
        
        return results
    
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
    
    def _update_batch_detection_tracking(self, img_path, pred_count, batch_detections):
        """Update batch detection tracking dictionary."""
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
        except (ValueError, IndexError):
            # If timestamp parsing fails, just skip tracking this for plotting
            pass
    
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
            # self.file_manager.save_txt_predictions_bb(predictions, img_path, txt_save_dir)
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
    
    def _save_batch_outputs(self, predictions, img_path, imgsave_dir, txtsave_dir, 
                           model_path, classes, class_colours, pred_count, model_type):
        """Save outputs for batch processing."""
        rel_path = os.path.relpath(os.path.dirname(img_path), self.img_dir)
        
        # Skip empty predictions to avoid unnecessary file operations
        if pred_count == 0 and not self.save_img:
            return
        
        # Save image predictions
        if self.save_img:
            img_save_subdir = os.path.join(imgsave_dir, rel_path)
            self.file_manager.save_image_predictions_bb(
                predictions, img_path, img_save_subdir, classes, class_colours
            )
        
        # Save text and JSON predictions
        if self.save_txt:
            txt_save_subdir = os.path.join(txtsave_dir, rel_path)
            # self.file_manager.save_txt_predictions_bb(predictions, img_path, txt_save_subdir)
            self.file_manager.save_json_predictions_bb(
                predictions, img_path, txt_save_subdir, model_path, classes
            )
        
        # Save bounding box text format if enabled
        if self.save_txt_bb:
            txt_save_subdir = os.path.join(txtsave_dir, rel_path)
            bb_txt_save_path = os.path.join(txt_save_subdir, os.path.basename(img_path)[:-4] + '_det_bb.txt')
            with open(bb_txt_save_path, "w") as file:
                for p in predictions:
                    x1, y1, x2, y2 = p[0:4].tolist()
                    conf = p[4]
                    cls = int(p[5])
                    line = f"{x1} {y1} {x2} {y2} {conf}\n"
                    file.write(line)
    
    def _precreate_output_directories(self, images, imgsave_dir, txtsave_dir):
        """Create all necessary output directories in advance."""
        needed_dirs = set()
        for img_path in images:
            rel_path = os.path.relpath(os.path.dirname(img_path), self.img_dir)
            if self.save_img:
                needed_dirs.add(os.path.join(imgsave_dir, rel_path))
            if self.save_txt or self.save_txt_bb:
                needed_dirs.add(os.path.join(txtsave_dir, rel_path))
        
        # Create all needed directories at once
        for directory in needed_dirs:
            os.makedirs(directory, exist_ok=True)
    
    def estimate_processing_time(self, total_images):
        """
        Estimate processing time based on batch size and number of images.
        
        Args:
            total_images: Total number of images to process
            
        Returns:
            dict: Dictionary with time estimates
        """
        # Rough estimates based on typical performance
        if torch.cuda.is_available():
            # GPU processing is much faster
            seconds_per_batch = 2.0  # Estimate 2 seconds per batch on GPU
            batches_needed = (total_images + self.batch_size - 1) // self.batch_size
            estimated_seconds = batches_needed * seconds_per_batch
        else:
            # CPU processing is slower
            seconds_per_image = 1.5  # Estimate 1.5 seconds per image on CPU
            estimated_seconds = total_images * seconds_per_image
        
        return {
            'total_images': total_images,
            'batch_size': self.batch_size,
            'estimated_seconds': estimated_seconds,
            'estimated_minutes': estimated_seconds / 60,
            'estimated_hours': estimated_seconds / 3600,
            'processing_mode': 'GPU' if torch.cuda.is_available() else 'CPU'
        }
    
    def print_processing_estimate(self, total_images):
        """Print an estimate of processing time."""
        estimate = self.estimate_processing_time(total_images)
        
        print(f"\nProcessing Estimate:")
        print(f"  Total images: {estimate['total_images']}")
        print(f"  Batch size: {estimate['batch_size']}")
        print(f"  Processing mode: {estimate['processing_mode']}")
        print(f"  Estimated time: {estimate['estimated_seconds']:.1f} seconds "
              f"({estimate['estimated_minutes']:.1f} minutes, "
              f"{estimate['estimated_hours']:.2f} hours)")