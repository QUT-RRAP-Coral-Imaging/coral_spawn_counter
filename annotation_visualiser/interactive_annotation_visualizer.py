#!/usr/bin/env python3

"""
Interactive Annotation Pipeline Visualizer

This script allows you to visualize the effects of the AnnotationPipeline
on a single image with interactive threshold adjustment capabilities.
"""

import os
import sys
import yaml
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, CheckButtons, Button
import tkinter as tk
from tkinter import filedialog, messagebox
import copy
import time
import threading
from functools import wraps

# Add annotation directory to path (relative to parent directory)
script_dir = os.path.dirname(__file__)
annotation_dir = os.path.join(script_dir, '..', 'annotation')
annotation_dir = os.path.abspath(annotation_dir)
sys.path.insert(0, annotation_dir)

# Import filters with error handling
try:
    from FilterSift import FilterSift   
    from FilterHue import FilterHue
    from FilterSaturation import FilterSaturation
    from FilterLaplacian import FilterLaplacian
    from FilterValue import FilterValue
    from FilterEdge import FilterEdge
except ImportError as e:
    print(f"Warning: Could not import filter modules: {e}")
    print("Make sure you're running this from the coral_spawn_counter directory")
    FilterSift = FilterHue = FilterSaturation = FilterLaplacian = FilterValue = FilterEdge = None


class InteractiveAnnotationVisualizer:
    def __init__(self, config_path, image_path=None):
        """
        Initialize the interactive visualizer.
        
        Args:
            config_path (str): Path to the YAML configuration file.
            image_path (str): Path to the image file (optional, can be selected later).
        """
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.original_config = copy.deepcopy(self.config)
        self.image_path = image_path
        self.image = None
        self.image_small = None  # Smaller version for faster processing
        self.scale_factor = 1.0
        self.filters = {}
        
        # Initialize matplotlib figure and subplots
        self.fig = None
        self.axes = {}
        self.sliders = {}
        self.checkboxes = {}
        
        # Results storage
        self.masks = {}
        self.combined_mask = None
        self.bounding_boxes = []
        
        # Performance and responsiveness
        self.last_update_time = 0
        self.update_delay = 0.1  # Minimum delay between updates (100ms)
        self.processing = False
        self.pending_update = False
        self.max_image_size = 800  # Max dimension for real-time processing
        
        # Status text
        self.status_text = None
        
        self.initialize_filters()
        
        if image_path:
            self.load_image(image_path)
    
    def initialize_filters(self):
        """Initialize all filter objects based on configuration."""
        self.filters = {}
        
        # Only initialize filters if the filter classes are available
        filter_classes = {
            'sift': FilterSift,
            'hue': FilterHue,
            'saturation': FilterSaturation,
            'value': FilterValue,
            'laplacian': FilterLaplacian,
            'edge': FilterEdge
        }
        
        for filter_name, filter_class in filter_classes.items():
            if (filter_name in self.config and 
                self.config[filter_name]['do'] and 
                filter_class is not None):
                try:
                    self.filters[filter_name] = filter_class(config=self.config[filter_name])
                except Exception as e:
                    print(f"Warning: Could not initialize {filter_name} filter: {e}")
                    self.filters[filter_name] = None
            else:
                self.filters[filter_name] = None
    
    def load_image(self, image_path):
        """Load an image from the specified path."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        print("Loading image...")
        self.image_path = image_path
        self.image = cv.imread(image_path)
        if self.image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Create a smaller version for faster processing
        height, width = self.image.shape[:2]
        max_dim = max(height, width)
        
        if max_dim > self.max_image_size:
            self.scale_factor = self.max_image_size / max_dim
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            self.image_small = cv.resize(self.image, (new_width, new_height))
            print(f"Resized image for processing: {width}x{height} -> {new_width}x{new_height}")
        else:
            self.image_small = self.image.copy()
            self.scale_factor = 1.0
        
        print(f"Loaded image: {image_path}")
        print(f"Original shape: {self.image.shape}")
        print(f"Processing shape: {self.image_small.shape}")
        print(f"Scale factor: {self.scale_factor:.2f}")
    
    def select_image(self):
        """Open a file dialog to select an image."""
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.load_image(file_path)
                return True
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                return False
        return False
    
    def debounced_update(self):
        """Update display with debouncing to prevent too frequent updates."""
        current_time = time.time()
        
        # If we're already processing, mark that an update is pending
        if self.processing:
            self.pending_update = True
            return
        
        # If not enough time has passed since last update, schedule for later
        time_since_last = current_time - self.last_update_time
        if time_since_last < self.update_delay:
            self.pending_update = True
            threading.Timer(self.update_delay - time_since_last, self.debounced_update).start()
            return
        
        # Perform the update
        self.update_display()
    
    def update_status(self, message, clear_after=None):
        """Update status message."""
        if self.status_text:
            self.status_text.set_text(f"Status: {message}")
            self.fig.canvas.draw_idle()
            
            # Auto-clear after specified time
            if clear_after:
                threading.Timer(clear_after, lambda: self.update_status("Ready")).start()
    
    def create_masks(self):
        """Create masks using all enabled filters with progress feedback."""
        if self.image_small is None:
            return
        
        self.update_status("Processing filters...")
        self.masks = {}
        
        # Use the smaller image for processing
        processing_image = self.image_small
        
        # Process with SIFT filter if enabled
        if self.filters['sift'] is not None and self.config['sift']['do']:
            try:
                self.update_status("Processing SIFT filter...")
                # Update filter config
                self.filters['sift'].config = self.config['sift']
                kp = self.filters['sift'].get_best_sift_features(processing_image)
                self.masks['sift'] = self.filters['sift'].create_sift_mask(processing_image, kp)
            except Exception as e:
                print(f"SIFT filter error: {e}")
                self.masks['sift'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Hue filter if enabled
        if self.filters['hue'] is not None and self.config['hue']['do']:
            try:
                self.update_status("Processing Hue filter...")
                # Update filter config
                self.filters['hue'].config = self.config['hue']
                self.masks['hue'] = self.filters['hue'].create_hue_mask(processing_image)
            except Exception as e:
                print(f"Hue filter error: {e}")
                self.masks['hue'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Saturation filter if enabled
        if self.filters['saturation'] is not None and self.config['saturation']['do']:
            try:
                self.update_status("Processing Saturation filter...")
                # Update filter config
                self.filters['saturation'].config = self.config['saturation']
                self.masks['saturation'] = self.filters['saturation'].create_saturation_mask(processing_image)
            except Exception as e:
                print(f"Saturation filter error: {e}")
                self.masks['saturation'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Value filter if enabled
        if self.filters['value'] is not None and self.config['value']['do']:
            try:
                self.update_status("Processing Value filter...")
                # Update filter config
                self.filters['value'].config = self.config['value']
                self.masks['value'] = self.filters['value'].create_value_mask(processing_image)
            except Exception as e:
                print(f"Value filter error: {e}")
                self.masks['value'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Laplacian filter if enabled
        if self.filters['laplacian'] is not None and self.config['laplacian']['do']:
            try:
                self.update_status("Processing Laplacian filter...")
                # Update filter config
                self.filters['laplacian'].config = self.config['laplacian']
                self.masks['laplacian'] = self.filters['laplacian'].create_laplacian_mask(processing_image)
            except Exception as e:
                print(f"Laplacian filter error: {e}")
                self.masks['laplacian'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Edge filter if enabled
        if self.filters['edge'] is not None and self.config['edge']['do']:
            try:
                self.update_status("Processing Edge filter...")
                # Update filter config
                self.filters['edge'].config = self.config['edge']
                self.masks['edge'] = self.filters['edge'].create_edge_mask(processing_image)
            except Exception as e:
                print(f"Edge filter error: {e}")
                self.masks['edge'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        self.update_status("Filters processed", clear_after=1.0)
    
    def combine_masks(self):
        """Combine all enabled masks using bitwise AND operation."""
        if not self.masks:
            return
        
        self.update_status("Combining masks...")
        enabled_masks = []
        for filter_name, filter_obj in self.filters.items():
            if (filter_obj is not None and 
                filter_name in self.config and 
                self.config[filter_name]['do'] and 
                filter_name in self.masks):
                enabled_masks.append(self.masks[filter_name])
        
        if enabled_masks:
            self.combined_mask = enabled_masks[0].copy()
            for mask in enabled_masks[1:]:
                self.combined_mask = cv.bitwise_and(self.combined_mask, mask)
        else:
            self.combined_mask = np.zeros(self.image_small.shape[:2], dtype=np.uint8)
    
    def extract_bounding_boxes(self):
        """Extract bounding boxes from the combined mask."""
        if self.combined_mask is None:
            return
        
        self.update_status("Extracting detections...")
        img_width, img_height = self.combined_mask.shape[1], self.combined_mask.shape[0]
        contours, _ = cv.findContours(self.combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        self.bounding_boxes = []
        for c in contours:
            # Get bounding box
            x, y, w, h = cv.boundingRect(c)
            
            # Scale back to original image coordinates
            x = x / self.scale_factor
            y = y / self.scale_factor
            w = w / self.scale_factor
            h = h / self.scale_factor
            
            # Convert to normalized coordinates
            orig_width, orig_height = self.image.shape[1], self.image.shape[0]
            xcen = (x + w/2.0)/orig_width
            ycen = (y + h/2.0)/orig_height
            
            # Store in YOLO format: class x_center y_center width height
            self.bounding_boxes.append([
                self.config['class']['label'], 
                xcen, 
                ycen, 
                w/orig_width, 
                h/orig_height
            ])
    
    def setup_interactive_plot(self):
        """Set up the interactive matplotlib plot with sliders and checkboxes."""
        if self.image is None:
            if not self.select_image():
                return
        
        # Create figure with better spacing - wider to accommodate more controls
        self.fig = plt.figure(figsize=(28, 14))
        self.fig.suptitle('Interactive Annotation Pipeline Visualizer', fontsize=16, y=0.98)
        
        # Define subplot layout with more space for controls
        # Top three rows for images, bottom row for controls
        gs = self.fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.6], width_ratios=[1, 1, 1, 1],
                                  hspace=0.3, wspace=0.2, top=0.93, bottom=0.15)
        
        # Row 1: Original and main filters
        self.axes['original'] = self.fig.add_subplot(gs[0, 0])
        self.axes['original'].set_title('Original Image')
        self.axes['original'].axis('off')
        
        self.axes['hue'] = self.fig.add_subplot(gs[0, 1])
        self.axes['hue'].set_title('Hue Filter')
        self.axes['hue'].axis('off')
        
        self.axes['sift'] = self.fig.add_subplot(gs[0, 2])
        self.axes['sift'].set_title('SIFT Filter')
        self.axes['sift'].axis('off')
        
        self.axes['laplacian'] = self.fig.add_subplot(gs[0, 3])
        self.axes['laplacian'].set_title('Laplacian Filter')
        self.axes['laplacian'].axis('off')
        
        # Row 2: Additional filters
        self.axes['saturation'] = self.fig.add_subplot(gs[1, 0])
        self.axes['saturation'].set_title('Saturation Filter')
        self.axes['saturation'].axis('off')
        
        self.axes['value'] = self.fig.add_subplot(gs[1, 1])
        self.axes['value'].set_title('Value Filter')
        self.axes['value'].axis('off')
        
        self.axes['edge'] = self.fig.add_subplot(gs[1, 2])
        self.axes['edge'].set_title('Edge Filter')
        self.axes['edge'].axis('off')
        
        self.axes['combined'] = self.fig.add_subplot(gs[1, 3])
        self.axes['combined'].set_title('Combined Mask')
        self.axes['combined'].axis('off')
        
        # Row 3: Final result
        self.axes['result'] = self.fig.add_subplot(gs[2, 0:2])
        self.axes['result'].set_title('Final Result with Bounding Boxes')
        self.axes['result'].axis('off')
        
        # Status area (use a thin horizontal strip at the top of control area)
        status_ax = self.fig.add_axes([0.1, 0.12, 0.8, 0.02])
        status_ax.axis('off')
        self.status_text = status_ax.text(0.02, 0.5, 'Status: Ready', fontsize=11, 
                                         transform=status_ax.transAxes, 
                                         verticalalignment='center')
        
        # Performance info
        if hasattr(self, 'image_small') and self.image_small is not None:
            perf_text = f"Processing: {self.image_small.shape[1]}x{self.image_small.shape[0]} (scale: {self.scale_factor:.2f})"
        else:
            perf_text = "No image loaded"
        status_ax.text(0.5, 0.5, perf_text, fontsize=10, 
                      transform=status_ax.transAxes, 
                      verticalalignment='center', horizontalalignment='center')
        
        self.setup_controls()
        self.update_display()
    
    def setup_controls(self):
        """Set up interactive controls (sliders and checkboxes) with organized layout."""
        # Control panel dimensions
        panel_bottom = 0.02
        panel_height = 0.09
        
        # Column layout
        col_width = 0.18
        col_spacing = 0.02
        slider_height = 0.015
        slider_spacing = 0.005
        
        # Define columns for different filter types
        columns = {
            'hue': 0.05,
            'sift': 0.05 + col_width + col_spacing,
            'laplacian': 0.05 + 2 * (col_width + col_spacing),
            'buttons': 0.05 + 3 * (col_width + col_spacing),
            'checkboxes': 0.05 + 3 * (col_width + col_spacing) + 0.15  # Increased spacing
        }
        
        # HUE CONTROLS COLUMN
        if 'hue' in self.config and self.config['hue']['do']:
            col_x = columns['hue']
            y_start = panel_bottom + panel_height - 0.02
            
            # Hue range sliders
            ax_hue_min = plt.axes([col_x, y_start - slider_height, col_width, slider_height])
            self.sliders['hue_min'] = Slider(
                ax_hue_min, 'Hue Min', 0, 179, 
                valinit=self.config['hue']['hue_min'], 
                valfmt='%d'
            )
            self.sliders['hue_min'].on_changed(self.update_hue_min)
            
            ax_hue_max = plt.axes([col_x, y_start - 2*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['hue_max'] = Slider(
                ax_hue_max, 'Hue Max', 0, 179, 
                valinit=self.config['hue']['hue_max'], 
                valfmt='%d'
            )
            self.sliders['hue_max'].on_changed(self.update_hue_max)
            
            # Hue area sliders
            ax_hue_min_area = plt.axes([col_x, y_start - 3*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['hue_min_area'] = Slider(
                ax_hue_min_area, 'Min Area', 100, 10000, 
                valinit=self.config['hue']['min_area'], 
                valfmt='%d'
            )
            self.sliders['hue_min_area'].on_changed(self.update_hue_min_area)
            
            ax_hue_max_area = plt.axes([col_x, y_start - 4*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['hue_max_area'] = Slider(
                ax_hue_max_area, 'Max Area', 1000, 50000, 
                valinit=self.config['hue']['max_area'], 
                valfmt='%d'
            )
            self.sliders['hue_max_area'].on_changed(self.update_hue_max_area)
        
        # SIFT CONTROLS COLUMN
        if 'sift' in self.config and self.config['sift']['do']:
            col_x = columns['sift']
            y_start = panel_bottom + panel_height - 0.02
            
            ax_sift_contrast = plt.axes([col_x, y_start - slider_height, col_width, slider_height])
            self.sliders['sift_contrast'] = Slider(
                ax_sift_contrast, 'SIFT Contrast', 0.001, 0.1, 
                valinit=self.config['sift']['contrast_threshold'], 
                valfmt='%.3f'
            )
            self.sliders['sift_contrast'].on_changed(self.update_sift_contrast)
            
            ax_sift_edge = plt.axes([col_x, y_start - 2*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['sift_edge'] = Slider(
                ax_sift_edge, 'SIFT Edge', 1, 200, 
                valinit=self.config['sift']['edge_threshold'], 
                valfmt='%d'
            )
            self.sliders['sift_edge'].on_changed(self.update_sift_edge)
            
            ax_sift_dilate = plt.axes([col_x, y_start - 3*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['sift_dilate'] = Slider(
                ax_sift_dilate, 'SIFT Dilate', 1, 100, 
                valinit=self.config['sift']['dilate'], 
                valfmt='%d'
            )
            self.sliders['sift_dilate'].on_changed(self.update_sift_dilate)
        
        # LAPLACIAN CONTROLS COLUMN
        if 'laplacian' in self.config and self.config['laplacian']['do']:
            col_x = columns['laplacian']
            y_start = panel_bottom + panel_height - 0.02
            
            ax_lapl_thresh = plt.axes([col_x, y_start - slider_height, col_width, slider_height])
            self.sliders['lapl_threshold'] = Slider(
                ax_lapl_thresh, 'Lapl Thresh', 1, 100, 
                valinit=self.config['laplacian']['laplacian_threshold'], 
                valfmt='%d'
            )
            self.sliders['lapl_threshold'].on_changed(self.update_lapl_threshold)
            
            ax_lapl_kernel = plt.axes([col_x, y_start - 2*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['lapl_kernel'] = Slider(
                ax_lapl_kernel, 'Lapl Kernel', 1, 15, 
                valinit=self.config['laplacian']['laplacian_kernel_size'], 
                valfmt='%d'
            )
            self.sliders['lapl_kernel'].on_changed(self.update_lapl_kernel)
            
            ax_lapl_min_area = plt.axes([col_x, y_start - 3*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['lapl_min_area'] = Slider(
                ax_lapl_min_area, 'Lapl Min Area', 100, 5000, 
                valinit=self.config['laplacian']['min_area'], 
                valfmt='%d'
            )
            self.sliders['lapl_min_area'].on_changed(self.update_lapl_min_area)
        
        # SATURATION CONTROLS (add a fourth column)
        if 'saturation' in self.config and self.config['saturation']['do']:
            sat_col_x = 0.05 + 4 * (col_width + col_spacing)  # Add a fourth column
            y_start = panel_bottom + panel_height - 0.02
            
            ax_sat_min_area = plt.axes([sat_col_x, y_start - slider_height, col_width, slider_height])
            self.sliders['sat_min_area'] = Slider(
                ax_sat_min_area, 'Sat Min Area', 100, 10000, 
                valinit=self.config['saturation'].get('min_area', 2000), 
                valfmt='%d'
            )
            self.sliders['sat_min_area'].on_changed(self.update_sat_min_area)
            
            ax_sat_max_area = plt.axes([sat_col_x, y_start - 2*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['sat_max_area'] = Slider(
                ax_sat_max_area, 'Sat Max Area', 1000, 50000, 
                valinit=self.config['saturation'].get('max_area', 30000), 
                valfmt='%d'
            )
            self.sliders['sat_max_area'].on_changed(self.update_sat_max_area)
        
        # VALUE CONTROLS (add to the fourth column)
        if 'value' in self.config and self.config['value']['do']:
            val_col_x = 0.05 + 4 * (col_width + col_spacing)  # Same column as saturation
            y_start = panel_bottom + panel_height - 0.02 - 3*(slider_height + slider_spacing)  # Start below saturation
            
            ax_val_min = plt.axes([val_col_x, y_start - slider_height, col_width, slider_height])
            self.sliders['val_min'] = Slider(
                ax_val_min, 'Value Min', 0, 255, 
                valinit=self.config['value'].get('value_min', 0), 
                valfmt='%d'
            )
            self.sliders['val_min'].on_changed(self.update_val_min)
            
            ax_val_max = plt.axes([val_col_x, y_start - 2*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['val_max'] = Slider(
                ax_val_max, 'Value Max', 0, 255, 
                valinit=self.config['value'].get('value_max', 150), 
                valfmt='%d'
            )
            self.sliders['val_max'].on_changed(self.update_val_max)
        
        # EDGE CONTROLS (add to a fifth column or adjust layout)
        if 'edge' in self.config and self.config['edge']['do']:
            edge_col_x = 0.05 + 5 * (col_width + col_spacing)  # Fifth column
            y_start = panel_bottom + panel_height - 0.02
            
            ax_edge_lower = plt.axes([edge_col_x, y_start - slider_height, col_width, slider_height])
            self.sliders['edge_lower'] = Slider(
                ax_edge_lower, 'Edge Lower', 1, 50, 
                valinit=self.config['edge'].get('canny_lower_thresh', 8), 
                valfmt='%d'
            )
            self.sliders['edge_lower'].on_changed(self.update_edge_lower)
            
            ax_edge_upper = plt.axes([edge_col_x, y_start - 2*(slider_height + slider_spacing), col_width, slider_height])
            self.sliders['edge_upper'] = Slider(
                ax_edge_upper, 'Edge Upper', 1, 100, 
                valinit=self.config['edge'].get('canny_upper_thresh', 15), 
                valfmt='%d'
            )
            self.sliders['edge_upper'].on_changed(self.update_edge_upper)
        
        # BUTTONS COLUMN
        button_x = 0.05 + 3 * (col_width + col_spacing)  # Keep buttons in 4th column position
        button_width = 0.1
        button_height = 0.025
        y_start = panel_bottom + panel_height - 0.02
        
        # Reset button
        ax_reset = plt.axes([button_x, y_start - button_height, button_width, button_height])
        self.reset_button = Button(ax_reset, 'Reset')
        self.reset_button.on_clicked(self.reset_parameters)
        
        # Save config button
        ax_save = plt.axes([button_x, y_start - 2*(button_height + slider_spacing), button_width, button_height])
        self.save_button = Button(ax_save, 'Save Config')
        self.save_button.on_clicked(self.save_config)
        
        # Load image button
        ax_load = plt.axes([button_x, y_start - 3*(button_height + slider_spacing), button_width, button_height])
        self.load_button = Button(ax_load, 'Load Image')
        self.load_button.on_clicked(self.load_image_button)
        
        # CHECKBOXES COLUMN
        checkbox_x = columns['checkboxes']
        filter_names = ['hue', 'sift', 'laplacian', 'saturation', 'value', 'edge']
        labels = []
        actives = []
        
        for name in filter_names:
            if name in self.config:
                labels.append(name.capitalize())
                actives.append(self.config[name]['do'])
        
        if labels:
            ax_checkbox = plt.axes([checkbox_x, panel_bottom + 0.01, 0.1, panel_height - 0.02])
            self.checkboxes['filters'] = CheckButtons(ax_checkbox, labels, actives)
            self.checkboxes['filters'].on_clicked(self.toggle_filter)
    
    def update_hue_min(self, val):
        """Update hue minimum threshold."""
        self.config['hue']['hue_min'] = int(val)
        self.debounced_update()
    
    def update_hue_max(self, val):
        """Update hue maximum threshold."""
        self.config['hue']['hue_max'] = int(val)
        self.debounced_update()
    
    def update_hue_min_area(self, val):
        """Update hue minimum area threshold."""
        self.config['hue']['min_area'] = int(val)
        self.debounced_update()
    
    def update_hue_max_area(self, val):
        """Update hue maximum area threshold."""
        self.config['hue']['max_area'] = int(val)
        self.debounced_update()
    
    def update_sift_contrast(self, val):
        """Update SIFT contrast threshold."""
        self.config['sift']['contrast_threshold'] = float(val)
        self.debounced_update()
    
    def update_sift_edge(self, val):
        """Update SIFT edge threshold."""
        self.config['sift']['edge_threshold'] = int(val)
        self.debounced_update()
    
    def update_sift_dilate(self, val):
        """Update SIFT dilation size."""
        self.config['sift']['dilate'] = int(val)
        self.debounced_update()
    
    def update_lapl_threshold(self, val):
        """Update Laplacian threshold."""
        self.config['laplacian']['laplacian_threshold'] = int(val)
        self.debounced_update()
    
    def update_lapl_kernel(self, val):
        """Update Laplacian kernel size."""
        self.config['laplacian']['laplacian_kernel_size'] = int(val)
        self.debounced_update()
    
    def update_lapl_min_area(self, val):
        """Update Laplacian minimum area."""
        self.config['laplacian']['min_area'] = int(val)
        self.debounced_update()
    
    def update_sat_min_area(self, val):
        """Update saturation minimum area."""
        self.config['saturation']['min_area'] = int(val)
        self.debounced_update()
    
    def update_sat_max_area(self, val):
        """Update saturation maximum area."""
        self.config['saturation']['max_area'] = int(val)
        self.debounced_update()
    
    def update_val_min(self, val):
        """Update value minimum threshold."""
        self.config['value']['value_min'] = int(val)
        self.debounced_update()
    
    def update_val_max(self, val):
        """Update value maximum threshold."""
        self.config['value']['value_max'] = int(val)
        self.debounced_update()
    
    def update_edge_lower(self, val):
        """Update edge detection lower threshold."""
        self.config['edge']['canny_lower_thresh'] = int(val)
        self.debounced_update()
    
    def update_edge_upper(self, val):
        """Update edge detection upper threshold."""
        self.config['edge']['canny_upper_thresh'] = int(val)
        self.debounced_update()
    
    def toggle_filter(self, label):
        """Toggle filter on/off."""
        filter_name = label.lower()
        if filter_name in self.config:
            self.config[filter_name]['do'] = not self.config[filter_name]['do']
            # Reinitialize filters
            self.initialize_filters()
            self.debounced_update()
    
    def reset_parameters(self, event):
        """Reset all parameters to original values."""
        self.config = copy.deepcopy(self.original_config)
        self.initialize_filters()
        
        # Update slider values
        for slider_name, slider in self.sliders.items():
            if slider_name == 'hue_min':
                slider.reset()
                slider.set_val(self.config['hue']['hue_min'])
            elif slider_name == 'hue_max':
                slider.reset()
                slider.set_val(self.config['hue']['hue_max'])
            # Add more slider resets as needed
        
        self.update_display()
    
    def save_config(self, event):
        """Save current configuration to a YAML file."""
        root = tk.Tk()
        root.withdraw()
        
        file_path = filedialog.asksaveasfilename(
            title="Save configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False)
                messagebox.showinfo("Success", f"Configuration saved to {file_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def load_image_button(self, event):
        """Load a new image via button click."""
        self.update_status("Loading new image...")
        if self.select_image():
            self.update_status("Image loaded successfully!", clear_after=2.0)
            self.update_display()
        else:
            self.update_status("Failed to load image", clear_after=2.0)
    
    def update_display(self):
        """Update all displays with current parameters."""
        if self.image is None or self.processing:
            return
        
        self.processing = True
        self.last_update_time = time.time()
        start_time = time.time()
        
        try:
            # Process the image with current parameters
            self.create_masks()
            self.combine_masks()
            self.extract_bounding_boxes()
            
            # Clear all axes efficiently
            for ax in self.axes.values():
                ax.clear()
                ax.axis('off')
            
            # Display original image (use smaller version for display if very large)
            display_image = self.image_small if self.scale_factor < 1.0 else self.image
            self.axes['original'].imshow(cv.cvtColor(display_image, cv.COLOR_BGR2RGB))
            self.axes['original'].set_title('Original Image')
            
            # Display individual filter masks
            if 'hue' in self.masks:
                self.axes['hue'].imshow(self.masks['hue'], cmap='gray')
                self.axes['hue'].set_title(f'Hue Filter ({self.config["hue"]["hue_min"]}-{self.config["hue"]["hue_max"]})')
            
            if 'sift' in self.masks:
                self.axes['sift'].imshow(self.masks['sift'], cmap='gray')
                self.axes['sift'].set_title(f'SIFT Filter (thresh: {self.config["sift"]["contrast_threshold"]:.3f})')
            
            if 'laplacian' in self.masks:
                self.axes['laplacian'].imshow(self.masks['laplacian'], cmap='gray')
                self.axes['laplacian'].set_title(f'Laplacian Filter (thresh: {self.config["laplacian"]["laplacian_threshold"]})')
            
            if 'saturation' in self.masks:
                self.axes['saturation'].imshow(self.masks['saturation'], cmap='gray')
                self.axes['saturation'].set_title('Saturation Filter')
            
            if 'value' in self.masks:
                self.axes['value'].imshow(self.masks['value'], cmap='gray')
                self.axes['value'].set_title(f'Value Filter ({self.config["value"].get("value_min", 0)}-{self.config["value"].get("value_max", 150)})')
            
            if 'edge' in self.masks:
                self.axes['edge'].imshow(self.masks['edge'], cmap='gray')
                self.axes['edge'].set_title(f'Edge Filter ({self.config["edge"].get("canny_lower_thresh", 8)}-{self.config["edge"].get("canny_upper_thresh", 15)})')
            
            # Display combined mask
            if self.combined_mask is not None:
                self.axes['combined'].imshow(self.combined_mask, cmap='gray')
                self.axes['combined'].set_title('Combined Mask')
            
            # Display final result with bounding boxes
            result_image = cv.cvtColor(display_image.copy(), cv.COLOR_BGR2RGB)
            img_height, img_width = result_image.shape[:2]
            
            # Draw bounding boxes (scaled appropriately)
            for bbox in self.bounding_boxes:
                # Convert from normalized coordinates to display image coordinates
                x_center, y_center, width, height = bbox[1:5]
                
                # Scale to display image size
                display_width = display_image.shape[1]
                display_height = display_image.shape[0]
                
                x = int((x_center - width/2) * display_width)
                y = int((y_center - height/2) * display_height)
                w = int(width * display_width)
                h = int(height * display_height)
                
                # Draw bounding box
                rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                       edgecolor='red', facecolor='none')
                self.axes['result'].add_patch(rect)
            
            self.axes['result'].imshow(result_image)
            self.axes['result'].set_title(f'Final Result ({len(self.bounding_boxes)} detections)')
            
            # Update the display efficiently
            self.fig.canvas.draw_idle()
            
            # Update performance stats
            processing_time = time.time() - start_time
            enabled_filters = [name for name, filt in self.filters.items() 
                              if filt is not None and name in self.config and self.config[name]['do']]
            
            # Update status with performance info
            self.update_status(f"Updated in {processing_time:.2f}s | Filters: {', '.join(enabled_filters)} | Detections: {len(self.bounding_boxes)}")
            
            print(f"Display updated in {processing_time:.2f}s - Detections: {len(self.bounding_boxes)}")
            
        except Exception as e:
            print(f"Error updating display: {e}")
            self.update_status(f"Error: {str(e)}", clear_after=3.0)
        finally:
            self.processing = False
            
            # Check if another update is pending
            if self.pending_update:
                self.pending_update = False
                threading.Timer(0.1, self.debounced_update).start()
    
    def run(self):
        """Start the interactive visualizer."""
        self.setup_interactive_plot()
        plt.tight_layout()
        plt.show()


def main():
    """Main function to run the interactive visualizer."""
    # Default configuration file (relative to parent directory)
    config_path = "../data_yaml_files/annotation_cslics_2024_nov_amil_tank3_10000000f620da42.yaml"
    
    # Convert to absolute path
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, config_path)
    config_path = os.path.abspath(config_path)
    
    # Check if config file exists
    if not os.path.exists(config_path):
        print(f"Configuration file not found: {config_path}")
        root = tk.Tk()
        root.withdraw()
        config_path = filedialog.askopenfilename(
            title="Select configuration file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")]
        )
        if not config_path:
            print("No configuration file selected. Exiting.")
            return
    
    try:
        # Create and run the visualizer
        visualizer = InteractiveAnnotationVisualizer(config_path)
        print("Starting Interactive Annotation Pipeline Visualizer...")
        print("Use the controls to adjust parameters and see real-time effects.")
        visualizer.run()
        
    except Exception as e:
        print(f"Error running visualizer: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
