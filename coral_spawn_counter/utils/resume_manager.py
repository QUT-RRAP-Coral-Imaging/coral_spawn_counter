# utils/resume_manager.py
import os
from pathlib import Path


class ResumeManager:
    """Handles resume functionality for interrupted processing runs."""
    
    def __init__(self, config, img_dir, surface_txtsave_dir, subsurface_txtsave_dir):
        self.config = config
        self.img_dir = img_dir
        self.surface_txtsave_dir = surface_txtsave_dir
        self.subsurface_txtsave_dir = subsurface_txtsave_dir
        
        self.resume = config.resume
        self.resume_from_image = config.resume_from_image
        self.mode = config.mode
        
        # Cache for processed images
        self._processed_images = None
        
        if self.resume:
            self._processed_images = self._get_processed_images()
    
    def _get_processed_images(self):
        """
        Get a set of already processed image filenames by scanning output directories.
        
        Returns:
            set: Set of processed image base filenames (without extension)
        """
        processed_images = set()
        
        # Check both surface and subsurface output directories based on mode
        if self.mode in ["surface", "both"] and os.path.exists(self.surface_txtsave_dir):
            for json_file in Path(self.surface_txtsave_dir).rglob('*_det.json'):
                processed_images.add(json_file.stem[:-4])  # Remove '_det' suffix
        
        if self.mode in ["subsurface", "both"] and os.path.exists(self.subsurface_txtsave_dir):
            for json_file in Path(self.subsurface_txtsave_dir).rglob('*_det.json'):
                processed_images.add(json_file.stem[:-4])  # Remove '_det' suffix
        
        print(f"Found {len(processed_images)} already processed images")
        return processed_images
    
    def should_process_image(self, img_path):
        """
        Determine if an image should be processed based on resume settings.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            bool: True if the image should be processed, False if it should be skipped
        """
        # If not resuming, process all images
        if not self.resume:
            return True
        
        # If resume_from_image is provided, check if we've reached that image
        if self.resume_from_image:
            img_filename = Path(img_path).name
            if img_filename == self.resume_from_image:
                print(f"Resume point reached: {img_filename}")
                self.resume_from_image = None  # Clear so subsequent images are processed
                return True
            elif self.resume_from_image is not None:
                # Still looking for the resume point
                return False
        
        # Check if image has already been processed
        if self._processed_images is not None:
            filename = Path(img_path).stem
            return filename not in self._processed_images
        
        return True
    
    def filter_images_for_resume(self, img_list):
        """
        Filter the image list based on resume settings.
        
        Args:
            img_list: List of image paths
            
        Returns:
            list: Filtered list of images to process
        """
        if not self.resume:
            return img_list
        
        # If resuming from a specific image, find that point
        if self.resume_from_image:
            resume_index = None
            for i, img_path in enumerate(img_list):
                if Path(img_path).name == self.resume_from_image:
                    resume_index = i
                    break
            
            if resume_index is not None:
                print(f"Resuming from image {self.resume_from_image} (index {resume_index})")
                return img_list[resume_index:]
            else:
                print(f"Warning: Resume image {self.resume_from_image} not found in image list")
                return img_list
        
        # Filter out already processed images
        return [img for img in img_list if self.should_process_image(img)]
    
    def get_resume_image_path(self):
        """
        Get the full path to the resume image if specified.
        
        Returns:
            str or None: Full path to the resume image, or None if not found/specified
        """
        if not self.resume_from_image:
            return None
        
        # Search for the image in the image directory structure
        resume_image_path = self._find_image_in_directory(self.resume_from_image, self.img_dir)
        
        if resume_image_path:
            print(f"Found resume image at: {resume_image_path}")
            return resume_image_path
        else:
            print(f"Warning: Resume image '{self.resume_from_image}' not found in {self.img_dir}")
            return None
        
    def _find_image_in_directory(self, image_name, search_dir):
        """
        Recursively search for an image file in directory structure.
        
        Args:
            image_name: Name of the image file to find
            search_dir: Directory to search in
            
        Returns:
            str or None: Full path to the image if found, None otherwise
        """
        import os
        
        for root, dirs, files in os.walk(search_dir):
            if image_name in files:
                return os.path.join(root, image_name)
        return None