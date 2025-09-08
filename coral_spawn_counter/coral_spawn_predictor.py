# coral_spawn_predictor.py
import os
import time
from pathlib import Path
import torch

from config.config_manager import ConfigManager
from models.model_manager import ModelManager
from processing.image_processor import ImageProcessor
from processing.batch_processor import BatchProcessor
from data.detection_data_manager import DetectionDataManager
from data.file_manager import FileManager
from visualisation.detection_plotter import DetectionPlotter
from utils.time_utils import TimeUtils
from utils.resume_manager import ResumeManager


class CoralSpawnPredictor:
    """Main orchestrator class for coral spawn prediction."""
    
    def __init__(self, config_file):
        """Initialize the coral spawn predictor with a config file."""
        # Initialize configuration
        self.config = ConfigManager(config_file)
        
        # Validate configuration and print comprehensive summary
        config_valid = self.config.validate_and_print_config()
        if not config_valid:
            raise ValueError("Configuration validation failed. Please check the errors above.")
        
        # Initialize device
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Initialize utility classes
        self.time_utils = TimeUtils(self.config.submersion_datetime)
        
        # Initialize separate model managers for surface and subsurface
        self.surface_model_manager = None
        self.subsurface_model_manager = None
        
        # Initialize model managers based on mode
        if self.config.mode in ["surface", "both"]:
            if os.path.exists(self.config.surface_weights_path):
                self.surface_model_manager = ModelManager(self.config, self.device, model_type='surface')
            else:
                print(f"Warning: Surface model weights not found: {self.config.surface_weights_path}")
        
        if self.config.mode in ["subsurface", "both"]:
            if os.path.exists(self.config.subsurface_weights_path):
                self.subsurface_model_manager = ModelManager(self.config, self.device, model_type='subsurface')
            else:
                print(f"Warning: Subsurface model weights not found: {self.config.subsurface_weights_path}")
        
        # Initialize file manager
        self.file_manager = FileManager(self.config)
        
        # Initialize detection data manager
        self.detection_data_manager = DetectionDataManager(self.config)
        
        # Initialize resume manager
        self.resume_manager = ResumeManager(
            self.config, 
            self.config.img_dir,
            self.file_manager.surface_txtsave_dir,
            self.file_manager.subsurface_txtsave_dir
        )
        
        # Check if resume image exists (only when resume mode is enabled)
        if self.config.resume and self.config.resume_from_image:
            resume_path = self.resume_manager.get_resume_image_path()
            if not resume_path:
                print(f"Error: Resume image '{self.config.resume_from_image}' not found")
                return False
            print(f"Resume configuration validated: Will start from {self.config.resume_from_image}")
            
        
        # Initialize processors with the new model managers
        self.image_processor = ImageProcessor(
            self.config, 
            surface_model_manager=self.surface_model_manager,
            subsurface_model_manager=self.subsurface_model_manager,
            file_manager=self.file_manager, 
            detection_data_manager=self.detection_data_manager, 
            time_utils=self.time_utils
        )
        
        self.batch_processor = BatchProcessor(
            self.config, 
            surface_model_manager=self.surface_model_manager,
            subsurface_model_manager=self.subsurface_model_manager,
            file_manager=self.file_manager,
            detection_data_manager=self.detection_data_manager, 
            time_utils=self.time_utils
        )
        
        # Initialize plotter
        self.plotter = DetectionPlotter(
            self.config, self.detection_data_manager, self.time_utils
        )
    
    def run(self):
        """Run prediction on all images with resume capability."""
        print(f'Fetching image list in all subfolders from: {self.config.img_dir}')
        print(f'Processing mode: {self.config.mode}')
        print(f'Parallel processing: {"Enabled" if self.config.parallel else "Disabled"}')
        print(f'Resume mode: {"Enabled" if self.config.resume else "Disabled"}')
        
        # Print model information
        if self.surface_model_manager:
            print(f'Surface model: {self.surface_model_manager.get_model_name()}')
        if self.subsurface_model_manager:
            print(f'Subsurface model: {self.subsurface_model_manager.get_model_name()}')
        
        # Gather images
        start_time = time.time()
        img_list = sorted(Path(self.config.img_dir).rglob('cslics*_img.jpg'))
        print(f'Image list gathered in {time.time() - start_time:.2f} seconds')
        
        # Apply max_images limit if specified
        if self.config.max_images is not None and self.config.max_images > 0:
            img_list = img_list[:self.config.max_images]
            print(f'Limited to first {self.config.max_images} images')
        
        print(f'Number of images found: {len(img_list)}')
        print(f'Submersion time: {self.config.submersion_time}')
        
        # Apply resume filtering
        if self.config.resume:
            img_list = self.resume_manager.filter_images_for_resume(img_list)
            print(f'After resume filtering: {len(img_list)} images to process')
        
        # Apply image_skip filtering BEFORE separating by type
        image_skip = getattr(self.config, 'image_skip', 1)
        if image_skip > 1:
            original_count = len(img_list)
            img_list = [img for i, img in enumerate(img_list) if i % image_skip == 0]
            print(f'Applied image_skip={image_skip}: {len(img_list)} images selected from {original_count}')
    
        # Filter images by mode and resume settings
        start_time = time.time()
        surface_img_list = []
        subsurface_img_list = []
        
        if self.config.mode in ["surface", "both"] and self.surface_model_manager:
            surface_img_list = [
                img for img in img_list 
                if self.time_utils.is_surface_image(img) and 
                self.resume_manager.should_process_image(img)
            ]
            
        if self.config.mode in ["subsurface", "both"] and self.subsurface_model_manager:
            subsurface_img_list = [
                img for img in img_list 
                if not self.time_utils.is_surface_image(img) and 
                self.resume_manager.should_process_image(img)
            ]
    
        print(f'Images filtered by mode and resume settings in {time.time() - start_time:.2f} seconds')
        print(f'Found {len(surface_img_list)} surface images and {len(subsurface_img_list)} subsurface images to process')
        
        # Process images
        start_time = time.time()
        results = []
        
        if self.config.parallel and torch.cuda.is_available():
            print("Using parallel GPU processing")
            if self.config.mode == "both":
                combined_list = surface_img_list + subsurface_img_list
                results = self.batch_processor.process_images_parallel_gpu(combined_list, "Processing images")
            else:
                img_subset = surface_img_list if self.config.mode == "surface" else subsurface_img_list
                results = self.batch_processor.process_images_parallel_gpu(img_subset, "Processing images")
        elif self.config.parallel:
            print("Using parallel CPU processing")
            if self.config.mode in ["surface", "both"] and surface_img_list and self.surface_model_manager:
                print("\nProcessing surface images:")
                surface_results = self.batch_processor.process_images_parallel(surface_img_list, "Surface images")
                results.extend(surface_results)
            
            if self.config.mode in ["subsurface", "both"] and subsurface_img_list and self.subsurface_model_manager:
                print("\nProcessing subsurface images:")
                subsurface_results = self.batch_processor.process_images_parallel(subsurface_img_list, "Subsurface images")
                results.extend(subsurface_results)
        else:
            print("Using sequential processing")
            if self.config.mode in ["surface", "both"] and surface_img_list and self.surface_model_manager:
                print("\nProcessing surface images:")
                surface_results = self.image_processor.process_images_with_progress(surface_img_list, "Surface images")
                results.extend(surface_results)
            
            if self.config.mode in ["subsurface", "both"] and subsurface_img_list and self.subsurface_model_manager:
                print("\nProcessing subsurface images:")
                subsurface_results = self.image_processor.process_images_with_progress(subsurface_img_list, "Subsurface images")
                results.extend(subsurface_results)
        
        # Print statistics
        self._print_statistics(results, time.time() - start_time)
    
    def _print_statistics(self, results, duration):
        """Print processing statistics"""
        surface_detections = sum(count for count, model_type in results if model_type == "surface")
        subsurface_detections = sum(count for count, model_type in results if model_type == "subsurface")
        
        print('\nProcessing complete:')
        processed_count = len([r for r in results if r[1] != "skipped"])
        print(f'Total images processed: {processed_count}')
        
        if self.config.mode in ["surface", "both"] and self.surface_model_manager:
            surface_count = sum(1 for _, model_type in results if model_type == "surface")
            print(f'Surface model detections: {surface_detections} in {surface_count} images')
        
        if self.config.mode in ["subsurface", "both"] and self.subsurface_model_manager:
            subsurface_count = sum(1 for _, model_type in results if model_type == "subsurface")
            print(f'Subsurface model detections: {subsurface_detections} in {subsurface_count} images')
        
        print(f'Run time: {duration:.2f} sec ({duration / 60.0:.2f} min, {duration / 3600.0:.2f} hrs)')
        if processed_count > 0:
            print(f'Time per image: {duration / processed_count:.2f} sec')
    
    # Model access methods for other components
    def get_surface_model_manager(self):
        """Get the surface model manager."""
        return self.surface_model_manager
    
    def get_subsurface_model_manager(self):
        """Get the subsurface model manager."""
        return self.subsurface_model_manager
    
    def get_model_manager(self, model_type):
        """Get model manager by type."""
        if model_type == 'surface':
            return self.surface_model_manager
        elif model_type == 'subsurface':
            return self.subsurface_model_manager
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
    
    # Plotting methods that delegate to the plotter
    def plot_surface_detections(self, save_path=None):
        """Plot surface detections."""
        return self.plotter.plot_surface_detections(save_path)
    
    def plot_subsurface_detections(self, save_path=None):
        """Plot subsurface detections."""
        return self.plotter.plot_subsurface_detections(save_path)
    
    def plot_all_detections(self, save_path=None):
        """Plot all detections."""
        return self.plotter.plot_all_detections(save_path)
    
    def cleanup(self):
        """Clean up model resources."""
        if self.surface_model_manager:
            self.surface_model_manager.cleanup()
        if self.subsurface_model_manager:
            self.subsurface_model_manager.cleanup()


if __name__ == "__main__":
    
    # Default config file path if not specified
    config_file = "/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/spawn_predictor_20231205_t4_alor_cslics08.json"
    
    # Load and update config with command line arguments
    with open(config_file, 'r') as f:
        import json
        config = json.load(f)
        
    # Write updated config back
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Initialize and run predictor
    predictor = CoralSpawnPredictor(config_file)
    
    try:
        predictor.run()
        
        # Generate plots
        print("Generating detection plots...")
        if predictor.config.mode in ["surface", "both"] and predictor.surface_model_manager:
            predictor.plot_surface_detections()
        if predictor.config.mode in ["subsurface", "both"] and predictor.subsurface_model_manager:
            predictor.plot_subsurface_detections()
        if predictor.config.mode == "both" and predictor.surface_model_manager and predictor.subsurface_model_manager:
            predictor.plot_all_detections()
            
        print("Processing complete")
    
    finally:
        # Clean up resources
        predictor.cleanup()