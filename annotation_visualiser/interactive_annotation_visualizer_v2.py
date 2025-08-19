#!/usr/bin/env python3

"""
Interactive Annotation Pipeline Visualizer v2

This script provides a comprehensive interface for visualizing and adjusting
all annotation pipeline parameters with organized tabbed controls.
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
from tkinter import filedialog, messagebox, ttk
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


class InteractiveAnnotationVisualizerV2:
    def __init__(self, config_path, image_path=None):
        """
        Initialize the interactive visualizer with tabbed interface.
        
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
        
        # Control window and tabs
        self.control_window = None
        self.notebook = None
        
        self.initialize_filters()
        
        if image_path:
            self.load_image(image_path)
    
    def initialize_filters(self):
        """Initialize all filter objects based on configuration."""
        self.filters = {}
        
        filter_classes = {
            'sift': FilterSift,
            'hue': FilterHue,
            'saturation': FilterSaturation,
            'value': FilterValue,
            'laplacian': FilterLaplacian,
            'edge': FilterEdge
        }
        
        for filter_name, filter_class in filter_classes.items():
            if filter_class and filter_name in self.config and self.config[filter_name].get('do', False):
                try:
                    self.filters[filter_name] = filter_class(config=self.config[filter_name])
                except Exception as e:
                    print(f"Warning: Could not initialize {filter_name} filter: {e}")
                    self.filters[filter_name] = None
    
    def load_image(self, image_path=None):
        """Load an image for processing."""
        if image_path is None:
            root = tk.Tk()
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select an image",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
            )
            root.destroy()
            
        if not image_path:
            return False
            
        self.image_path = image_path
        self.image = cv.imread(image_path)
        
        if self.image is None:
            print(f"Error: Could not load image from {image_path}")
            return False
        
        # Create smaller version for processing
        height, width = self.image.shape[:2]
        max_dimension = max(width, height)
        
        if max_dimension > self.max_image_size:
            self.scale_factor = self.max_image_size / max_dimension
            new_width = int(width * self.scale_factor)
            new_height = int(height * self.scale_factor)
            self.image_small = cv.resize(self.image, (new_width, new_height))
        else:
            self.scale_factor = 1.0
            self.image_small = self.image.copy()
        
        print(f"Loaded image: {width}x{height}, processing at: {self.image_small.shape[1]}x{self.image_small.shape[0]} (scale: {self.scale_factor:.2f})")
        return True
    
    def select_image(self):
        """Open file dialog to select an image."""
        return self.load_image()
    
    def update_status(self, message, clear_after=None):
        """Update the status message."""
        if self.status_text:
            self.status_text.set_text(f"Status: {message}")
            self.fig.canvas.draw_idle()
            
            if clear_after:
                def clear_status():
                    if self.status_text:
                        self.status_text.set_text("Status: Ready")
                        self.fig.canvas.draw_idle()
                
                threading.Timer(clear_after, clear_status).start()
    
    def debounced_update(self):
        """Update display with debouncing to prevent excessive processing."""
        current_time = time.time()
        
        if self.processing:
            self.pending_update = True
            return
            
        if current_time - self.last_update_time < self.update_delay:
            self.pending_update = True
            
            def delayed_update():
                time.sleep(self.update_delay)
                if self.pending_update and not self.processing:
                    self.pending_update = False
                    self.update_display()
            
            threading.Thread(target=delayed_update, daemon=True).start()
            return
        
        self.last_update_time = current_time
        self.update_display()
    
    def process_filters(self):
        """Process all active filters and return masks."""
        if self.image_small is None:
            return {}
        
        masks = {}
        start_time = time.time()
        
        # Use the smaller image for processing
        processing_image = self.image_small
        
        # Process with SIFT filter if enabled
        if 'sift' in self.config and self.config['sift']['do']:
            try:
                self.update_status("Processing SIFT filter...")
                # Reinitialize filter with current config to ensure all parameters are up-to-date
                sift_filter = FilterSift(config=self.config['sift'])
                kp = sift_filter.get_best_sift_features(processing_image)
                masks['sift'] = sift_filter.create_sift_mask(processing_image, kp)
            except Exception as e:
                print(f"Error processing sift filter: {e}")
                masks['sift'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Hue filter if enabled
        if 'hue' in self.config and self.config['hue']['do']:
            try:
                self.update_status("Processing Hue filter...")
                # Reinitialize filter with current config to ensure all parameters are up-to-date
                hue_filter = FilterHue(config=self.config['hue'])
                masks['hue'] = hue_filter.create_hue_mask(processing_image)
            except Exception as e:
                print(f"Error processing hue filter: {e}")
                masks['hue'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Saturation filter if enabled
        if 'saturation' in self.config and self.config['saturation']['do']:
            try:
                self.update_status("Processing Saturation filter...")
                # Reinitialize filter with current config to ensure all parameters are up-to-date
                saturation_filter = FilterSaturation(config=self.config['saturation'])
                masks['saturation'] = saturation_filter.create_saturation_mask(processing_image)
            except Exception as e:
                print(f"Error processing saturation filter: {e}")
                masks['saturation'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Value filter if enabled
        if 'value' in self.config and self.config['value']['do']:
            try:
                self.update_status("Processing Value filter...")
                # Reinitialize filter with current config to ensure all parameters are up-to-date
                value_filter = FilterValue(config=self.config['value'])
                masks['value'] = value_filter.create_value_mask(processing_image)
            except Exception as e:
                print(f"Error processing value filter: {e}")
                masks['value'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Laplacian filter if enabled
        if 'laplacian' in self.config and self.config['laplacian']['do']:
            try:
                self.update_status("Processing Laplacian filter...")
                # Reinitialize filter with current config to ensure all parameters are up-to-date
                laplacian_filter = FilterLaplacian(config=self.config['laplacian'])
                masks['laplacian'] = laplacian_filter.create_laplacian_mask(processing_image)
            except Exception as e:
                print(f"Error processing laplacian filter: {e}")
                masks['laplacian'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        # Process with Edge filter if enabled
        if 'edge' in self.config and self.config['edge']['do']:
            try:
                self.update_status("Processing Edge filter...")
                # Reinitialize filter with current config to ensure all parameters are up-to-date
                edge_filter = FilterEdge(config=self.config['edge'])
                masks['edge'] = edge_filter.create_edge_mask(processing_image)
            except Exception as e:
                print(f"Error processing edge filter: {e}")
                masks['edge'] = np.zeros(processing_image.shape[:2], dtype=np.uint8)
        
        processing_time = time.time() - start_time
        self.update_status(f"Processing completed in {processing_time:.2f}s", clear_after=3)
        
        return masks
    
    def combine_masks(self, masks):
        """Combine individual filter masks into a single mask."""
        if not masks:
            return None
        
        combined = None
        for filter_name, mask in masks.items():
            if mask is not None:
                if combined is None:
                    combined = mask.copy()
                else:
                    combined = cv.bitwise_or(combined, mask)
        
        return combined
    
    def detect_bounding_boxes(self, mask):
        """Detect bounding boxes from the combined mask."""
        if mask is None:
            return []
        
        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        
        bounding_boxes = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv.boundingRect(contour)
            
            # Scale back to original image coordinates
            orig_height, orig_width = self.image.shape[:2]
            x = int(x / self.scale_factor)
            y = int(y / self.scale_factor)
            w = int(w / self.scale_factor)
            h = int(h / self.scale_factor)
            
            # Convert to YOLO format (normalized coordinates)
            xcen = (x + w/2) / orig_width
            ycen = (y + h/2) / orig_height
            w_norm = w / orig_width
            h_norm = h / orig_height
            
            bounding_boxes.append([xcen, ycen, w_norm, h_norm])
        
        return bounding_boxes
    
    def setup_visualization(self):
        """Set up the main visualization window."""
        if self.image is None:
            if not self.select_image():
                return
        
        # Create figure with better spacing - wider to accommodate more plots
        self.fig = plt.figure(figsize=(20, 12))
        self.fig.suptitle('Interactive Annotation Pipeline Visualizer v2', fontsize=16, y=0.95)
        
        # Define subplot layout for images only (controls will be in separate window)
        gs = self.fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1],
                                  hspace=0.3, wspace=0.2, top=0.90, bottom=0.1)
        
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
        
        # Row 3: Final result (span two columns)
        self.axes['result'] = self.fig.add_subplot(gs[2, 0:2])
        self.axes['result'].set_title('Final Result with Bounding Boxes')
        self.axes['result'].axis('off')
        
        # Status area
        status_ax = self.fig.add_axes([0.1, 0.02, 0.8, 0.03])
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
        
        self.setup_control_window()
        self.update_display()
    
    def setup_control_window(self):
        """Set up the separate control window with tabbed interface."""
        self.control_window = tk.Toplevel()
        self.control_window.title("Filter Controls")
        self.control_window.geometry("800x600")
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.control_window)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs for each filter
        self.create_hue_tab()
        self.create_sift_tab()
        self.create_laplacian_tab()
        self.create_saturation_tab()
        self.create_value_tab()
        self.create_edge_tab()
        self.create_general_tab()
    
    def create_hue_tab(self):
        """Create the hue filter control tab."""
        if 'hue' not in self.config:
            return
            
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Hue Filter")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row = 0
        
        # Enable/Disable checkbox
        hue_var = tk.BooleanVar(value=self.config['hue'].get('do', False))
        tk.Checkbutton(scrollable_frame, text="Enable Hue Filter", variable=hue_var,
                      command=lambda: self.toggle_filter_config('hue', hue_var.get())).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        # Hue range
        self.create_slider_control(scrollable_frame, 'hue', 'hue_min', "Hue Min", 0, 179, row)
        row += 1
        self.create_slider_control(scrollable_frame, 'hue', 'hue_max', "Hue Max", 0, 179, row)
        row += 1
        
        # Area constraints
        self.create_slider_control(scrollable_frame, 'hue', 'min_area', "Min Area", 100, 50000, row)
        row += 1
        self.create_slider_control(scrollable_frame, 'hue', 'max_area', "Max Area", 1000, 100000, row)
        row += 1
        
        # Circularity constraints
        self.create_slider_control(scrollable_frame, 'hue', 'min_circularity', "Min Circularity", 0.0, 1.0, row, resolution=0.01)
        row += 1
        self.create_slider_control(scrollable_frame, 'hue', 'max_circularity', "Max Circularity", 0.0, 1.0, row, resolution=0.01)
        row += 1
        
        # Processing options
        tk.Label(scrollable_frame, text="Processing Steps:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        for process_option in ['process_denoise', 'process_thresh', 'process_morph', 'process_fill', 'process_filter']:
            if process_option in self.config['hue']:
                var = tk.BooleanVar(value=self.config['hue'].get(process_option, False))
                tk.Checkbutton(scrollable_frame, text=process_option.replace('process_', '').title(), 
                              variable=var, command=lambda opt=process_option, v=var: self.update_boolean_config('hue', opt, v.get())).grid(row=row, column=0, sticky='w', pady=2)
                row += 1
        
        # Advanced parameters
        tk.Label(scrollable_frame, text="Advanced Parameters:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        advanced_params = [
            ('kernel_size', 1, 51, 2),
            ('denoise_template_window_size', 1, 21, 2),
            ('denoise_search_window_size', 1, 51, 2),
            ('denoise_strength', 1, 20, 1),
            ('edge_dilation_kernel_size', 1, 21, 2)
        ]
        
        for param, min_val, max_val, step in advanced_params:
            if param in self.config['hue']:
                self.create_slider_control(scrollable_frame, 'hue', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_sift_tab(self):
        """Create the SIFT filter control tab."""
        if 'sift' not in self.config:
            return
            
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="SIFT Filter")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row = 0
        
        # Enable/Disable checkbox
        sift_var = tk.BooleanVar(value=self.config['sift'].get('do', False))
        tk.Checkbutton(scrollable_frame, text="Enable SIFT Filter", variable=sift_var,
                      command=lambda: self.toggle_filter_config('sift', sift_var.get())).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        # SIFT parameters
        sift_params = [
            ('contrast_threshold', 0.001, 0.1, 0.001),
            ('edge_threshold', 1, 200, 1),
            ('sigma', 0.5, 3.0, 0.1),
            ('min_size', 1, 100, 1),
            ('max_size', 50, 500, 1),
            ('dilate', 1, 200, 1)
        ]
        
        for param, min_val, max_val, step in sift_params:
            if param in self.config['sift']:
                self.create_slider_control(scrollable_frame, 'sift', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        # Advanced parameters
        tk.Label(scrollable_frame, text="Advanced Parameters:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        advanced_params = [
            ('denoise_template_window_size', 1, 21, 2),
            ('denoise_search_window_size', 1, 51, 2),
            ('denoise_strength', 1, 20, 1)
        ]
        
        for param, min_val, max_val, step in advanced_params:
            if param in self.config['sift']:
                self.create_slider_control(scrollable_frame, 'sift', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_laplacian_tab(self):
        """Create the Laplacian filter control tab."""
        if 'laplacian' not in self.config:
            return
            
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Laplacian Filter")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row = 0
        
        # Enable/Disable checkbox
        lapl_var = tk.BooleanVar(value=self.config['laplacian'].get('do', False))
        tk.Checkbutton(scrollable_frame, text="Enable Laplacian Filter", variable=lapl_var,
                      command=lambda: self.toggle_filter_config('laplacian', lapl_var.get())).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        # Laplacian parameters
        lapl_params = [
            ('laplacian_threshold', 1, 100, 1),
            ('laplacian_kernel_size', 1, 15, 2),
            ('min_area', 100, 50000, 100),
            ('max_area', 1000, 100000, 100),
            ('min_circularity', 0.0, 1.0, 0.01),
            ('max_circularity', 0.0, 1.0, 0.01),
            ('edge_dilation_kernel_size', 1, 201, 2)
        ]
        
        for param, min_val, max_val, step in lapl_params:
            if param in self.config['laplacian']:
                self.create_slider_control(scrollable_frame, 'laplacian', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        # Processing options
        tk.Label(scrollable_frame, text="Processing Steps:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        for process_option in ['process_denoise', 'process_thresh', 'process_morph', 'process_fill', 'process_filter']:
            if process_option in self.config['laplacian']:
                var = tk.BooleanVar(value=self.config['laplacian'].get(process_option, False))
                tk.Checkbutton(scrollable_frame, text=process_option.replace('process_', '').title(), 
                              variable=var, command=lambda opt=process_option, v=var: self.update_boolean_config('laplacian', opt, v.get())).grid(row=row, column=0, sticky='w', pady=2)
                row += 1
        
        # Advanced parameters
        tk.Label(scrollable_frame, text="Advanced Parameters:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        advanced_params = [
            ('kernel_size', 1, 51, 2),
            ('denoise_template_window_size', 1, 21, 2),
            ('denoise_search_window_size', 1, 51, 2),
            ('denoise_strength', 1, 20, 1)
        ]
        
        for param, min_val, max_val, step in advanced_params:
            if param in self.config['laplacian']:
                self.create_slider_control(scrollable_frame, 'laplacian', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_saturation_tab(self):
        """Create the saturation filter control tab."""
        if 'saturation' not in self.config:
            return
            
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Saturation Filter")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row = 0
        
        # Enable/Disable checkbox
        sat_var = tk.BooleanVar(value=self.config['saturation'].get('do', False))
        tk.Checkbutton(scrollable_frame, text="Enable Saturation Filter", variable=sat_var,
                      command=lambda: self.toggle_filter_config('saturation', sat_var.get())).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        # Saturation parameters
        sat_params = [
            ('min_area', 100, 50000, 100),
            ('max_area', 1000, 100000, 100),
            ('min_circularity', 0.0, 1.0, 0.01),
            ('max_circularity', 0.0, 1.0, 0.01)
        ]
        
        for param, min_val, max_val, step in sat_params:
            if param in self.config['saturation']:
                self.create_slider_control(scrollable_frame, 'saturation', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        # Processing options
        tk.Label(scrollable_frame, text="Processing Steps:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        for process_option in ['process_denoise', 'process_thresh', 'process_morph', 'process_fill', 'process_filter']:
            if process_option in self.config['saturation']:
                var = tk.BooleanVar(value=self.config['saturation'].get(process_option, False))
                tk.Checkbutton(scrollable_frame, text=process_option.replace('process_', '').title(), 
                              variable=var, command=lambda opt=process_option, v=var: self.update_boolean_config('saturation', opt, v.get())).grid(row=row, column=0, sticky='w', pady=2)
                row += 1
        
        # Advanced parameters
        tk.Label(scrollable_frame, text="Advanced Parameters:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        advanced_params = [
            ('kernel_size', 1, 51, 2),
            ('denoise_template_window_size', 1, 21, 2),
            ('denoise_search_window_size', 1, 51, 2),
            ('denoise_strength', 1, 20, 1)
        ]
        
        for param, min_val, max_val, step in advanced_params:
            if param in self.config['saturation']:
                self.create_slider_control(scrollable_frame, 'saturation', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_value_tab(self):
        """Create the value filter control tab."""
        if 'value' not in self.config:
            return
            
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Value Filter")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row = 0
        
        # Enable/Disable checkbox
        val_var = tk.BooleanVar(value=self.config['value'].get('do', False))
        tk.Checkbutton(scrollable_frame, text="Enable Value Filter", variable=val_var,
                      command=lambda: self.toggle_filter_config('value', val_var.get())).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        # Value parameters
        val_params = [
            ('value_min', 0, 255, 1),
            ('value_max', 0, 255, 1),
            ('min_area', 100, 50000, 100),
            ('max_area', 1000, 100000, 100),
            ('min_circularity', 0.0, 1.0, 0.01),
            ('max_circularity', 0.0, 1.0, 0.01),
            ('edge_dilation_kernel_size', 1, 21, 2)
        ]
        
        for param, min_val, max_val, step in val_params:
            if param in self.config['value']:
                self.create_slider_control(scrollable_frame, 'value', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        # Processing options
        tk.Label(scrollable_frame, text="Processing Steps:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        for process_option in ['process_denoise', 'process_thresh', 'process_morph', 'process_fill', 'process_filter']:
            if process_option in self.config['value']:
                var = tk.BooleanVar(value=self.config['value'].get(process_option, False))
                tk.Checkbutton(scrollable_frame, text=process_option.replace('process_', '').title(), 
                              variable=var, command=lambda opt=process_option, v=var: self.update_boolean_config('value', opt, v.get())).grid(row=row, column=0, sticky='w', pady=2)
                row += 1
        
        # Advanced parameters
        tk.Label(scrollable_frame, text="Advanced Parameters:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        advanced_params = [
            ('kernel_size', 1, 51, 2),
            ('denoise_template_window_size', 1, 21, 2),
            ('denoise_search_window_size', 1, 51, 2),
            ('denoise_strength', 1, 20, 1)
        ]
        
        for param, min_val, max_val, step in advanced_params:
            if param in self.config['value']:
                self.create_slider_control(scrollable_frame, 'value', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_edge_tab(self):
        """Create the edge filter control tab."""
        if 'edge' not in self.config:
            return
            
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="Edge Filter")
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        row = 0
        
        # Enable/Disable checkbox
        edge_var = tk.BooleanVar(value=self.config['edge'].get('do', False))
        tk.Checkbutton(scrollable_frame, text="Enable Edge Filter", variable=edge_var,
                      command=lambda: self.toggle_filter_config('edge', edge_var.get())).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        # Edge parameters
        edge_params = [
            ('canny_lower_thresh', 1, 100, 1),
            ('canny_upper_thresh', 1, 200, 1),
            ('edge_dilation', 1, 51, 2),
            ('gaussian_kernel', 1, 51, 2),
            ('min_area', 100, 50000, 100),
            ('max_area', 1000, 100000, 100),
            ('min_circularity', 0.0, 1.0, 0.01),
            ('max_circularity', 0.0, 1.0, 0.01)
        ]
        
        for param, min_val, max_val, step in edge_params:
            if param in self.config['edge']:
                self.create_slider_control(scrollable_frame, 'edge', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        # Processing options
        tk.Label(scrollable_frame, text="Processing Steps:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        for process_option in ['process_denoise', 'process_thresh', 'process_morph', 'process_fill', 'process_filter']:
            if process_option in self.config['edge']:
                var = tk.BooleanVar(value=self.config['edge'].get(process_option, False))
                tk.Checkbutton(scrollable_frame, text=process_option.replace('process_', '').title(), 
                              variable=var, command=lambda opt=process_option, v=var: self.update_boolean_config('edge', opt, v.get())).grid(row=row, column=0, sticky='w', pady=2)
                row += 1
        
        # Advanced parameters
        tk.Label(scrollable_frame, text="Advanced Parameters:", font=('TkDefaultFont', 10, 'bold')).grid(row=row, column=0, columnspan=2, sticky='w', pady=(10,5))
        row += 1
        
        advanced_params = [
            ('kernel_size', 1, 51, 2),
            ('denoise_template_window_size', 1, 21, 2),
            ('denoise_search_window_size', 1, 51, 2),
            ('denoise_strength', 1, 20, 1)
        ]
        
        for param, min_val, max_val, step in advanced_params:
            if param in self.config['edge']:
                self.create_slider_control(scrollable_frame, 'edge', param, param.replace('_', ' ').title(), min_val, max_val, row, resolution=step)
                row += 1
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
    
    def create_general_tab(self):
        """Create the general controls tab."""
        frame = ttk.Frame(self.notebook)
        self.notebook.add(frame, text="General")
        
        # Main buttons
        button_frame = ttk.Frame(frame)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Load New Image", command=self.load_image_button).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Reset All Parameters", command=self.reset_parameters).pack(side=tk.LEFT, padx=10)
        ttk.Button(button_frame, text="Save Configuration", command=self.save_config).pack(side=tk.LEFT, padx=10)
        
        # Status area
        status_frame = ttk.LabelFrame(frame, text="Status Information")
        status_frame.pack(fill=tk.X, padx=20, pady=10)
        
        self.status_label = tk.Label(status_frame, text="Ready", fg="green")
        self.status_label.pack(pady=10)
        
        # Filter summary
        summary_frame = ttk.LabelFrame(frame, text="Active Filters")
        summary_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.summary_text = tk.Text(summary_frame, height=10, width=50)
        summary_scrollbar = ttk.Scrollbar(summary_frame, orient="vertical", command=self.summary_text.yview)
        self.summary_text.configure(yscrollcommand=summary_scrollbar.set)
        
        self.summary_text.pack(side="left", fill="both", expand=True)
        summary_scrollbar.pack(side="right", fill="y")
        
        self.update_filter_summary()
    
    def create_slider_control(self, parent, filter_name, param_name, label, min_val, max_val, row, resolution=1):
        """Create a slider control for a parameter."""
        tk.Label(parent, text=f"{label}:").grid(row=row, column=0, sticky='w', padx=(0, 10))
        
        current_val = self.config[filter_name].get(param_name, min_val)
        
        var = tk.DoubleVar(value=current_val)
        
        slider = tk.Scale(parent, from_=min_val, to=max_val, resolution=resolution, 
                         orient=tk.HORIZONTAL, variable=var, length=300,
                         command=lambda val: self.update_config_param(filter_name, param_name, float(val) if resolution < 1 else int(float(val))))
        slider.grid(row=row, column=1, sticky='ew', padx=(0, 10))
        
        value_label = tk.Label(parent, text=str(current_val), width=10)
        value_label.grid(row=row, column=2, sticky='w')
        
        # Store reference to update the label
        slider.value_label = value_label
        
        # Configure column weights
        parent.grid_columnconfigure(1, weight=1)
        
        return slider
    
    def update_config_param(self, filter_name, param_name, value):
        """Update a configuration parameter and refresh the display."""
        self.config[filter_name][param_name] = value
        
        # Update any associated label
        for widget in self.control_window.winfo_children():
            if hasattr(widget, 'winfo_children'):
                self.update_value_labels(widget, filter_name, param_name, value)
        
        # Update display (filters will be recreated during processing)
        self.debounced_update()
    
    def update_value_labels(self, widget, filter_name, param_name, value):
        """Recursively update value labels."""
        try:
            for child in widget.winfo_children():
                if hasattr(child, 'value_label'):
                    child.value_label.config(text=str(value))
                elif hasattr(child, 'winfo_children'):
                    self.update_value_labels(child, filter_name, param_name, value)
        except:
            pass
    
    def toggle_filter_config(self, filter_name, enabled):
        """Toggle a filter on/off."""
        self.config[filter_name]['do'] = enabled
        self.update_filter_summary()
        self.debounced_update()
    
    def update_boolean_config(self, filter_name, param_name, value):
        """Update a boolean configuration parameter."""
        self.config[filter_name][param_name] = value
        self.debounced_update()
    
    def update_filter_summary(self):
        """Update the filter summary in the general tab."""
        if hasattr(self, 'summary_text'):
            self.summary_text.delete(1.0, tk.END)
            
            summary = "Active Filters:\\n\\n"
            for filter_name in ['hue', 'sift', 'saturation', 'value', 'laplacian', 'edge']:
                if filter_name in self.config and self.config[filter_name].get('do', False):
                    summary += f"✓ {filter_name.title()} Filter\\n"
                    
                    # Add key parameters
                    if filter_name == 'hue':
                        summary += f"  Range: {self.config[filter_name].get('hue_min', 0)}-{self.config[filter_name].get('hue_max', 179)}\\n"
                    elif filter_name == 'sift':
                        summary += f"  Contrast: {self.config[filter_name].get('contrast_threshold', 0.025)}\\n"
                    elif filter_name == 'laplacian':
                        summary += f"  Threshold: {self.config[filter_name].get('laplacian_threshold', 25)}\\n"
                    elif filter_name == 'value':
                        summary += f"  Range: {self.config[filter_name].get('value_min', 0)}-{self.config[filter_name].get('value_max', 150)}\\n"
                    elif filter_name == 'edge':
                        summary += f"  Canny: {self.config[filter_name].get('canny_lower_thresh', 8)}-{self.config[filter_name].get('canny_upper_thresh', 15)}\\n"
                    
                    summary += "\\n"
                else:
                    summary += f"✗ {filter_name.title()} Filter (disabled)\\n\\n"
            
            self.summary_text.insert(tk.END, summary)
    
    def update_display(self):
        """Update the visualization display."""
        if self.image is None or self.processing:
            return
        
        self.processing = True
        start_time = time.time()
        
        try:
            # Process filters
            self.masks = self.process_filters()
            
            # Combine masks
            self.combined_mask = self.combine_masks(self.masks)
            
            # Detect bounding boxes
            self.bounding_boxes = self.detect_bounding_boxes(self.combined_mask)
            
            # Display original image
            if 'original' in self.axes:
                self.axes['original'].clear()
                self.axes['original'].imshow(cv.cvtColor(self.image, cv.COLOR_BGR2RGB))
                self.axes['original'].set_title('Original Image')
                self.axes['original'].axis('off')
            
            # Display individual filter results
            for filter_name, mask in self.masks.items():
                if filter_name in self.axes and mask is not None:
                    self.axes[filter_name].clear()
                    self.axes[filter_name].imshow(mask, cmap='gray')
                    active_count = np.sum(mask > 0) if mask is not None else 0
                    self.axes[filter_name].set_title(f'{filter_name.title()} Filter\n({active_count} active pixels)')
                    self.axes[filter_name].axis('off')
            
            # Display combined mask
            if 'combined' in self.axes and self.combined_mask is not None:
                self.axes['combined'].clear()
                self.axes['combined'].imshow(self.combined_mask, cmap='gray')
                active_count = np.sum(self.combined_mask > 0)
                self.axes['combined'].set_title(f'Combined Mask\n({active_count} active pixels)')
                self.axes['combined'].axis('off')
            
            # Display final result with bounding boxes
            if 'result' in self.axes:
                self.axes['result'].clear()
                result_img = cv.cvtColor(self.image, cv.COLOR_BGR2RGB)
                self.axes['result'].imshow(result_img)
                
                # Draw bounding boxes
                height, width = self.image.shape[:2]
                for bbox in self.bounding_boxes:
                    xcen, ycen, w_norm, h_norm = bbox
                    x = (xcen - w_norm/2) * width
                    y = (ycen - h_norm/2) * height
                    w = w_norm * width
                    h = h_norm * height
                    
                    rect = patches.Rectangle((x, y), w, h, linewidth=2, 
                                           edgecolor='red', facecolor='none')
                    self.axes['result'].add_patch(rect)
                
                self.axes['result'].set_title(f'Final Result ({len(self.bounding_boxes)} detections)')
                self.axes['result'].axis('off')
            
            # Update status
            processing_time = time.time() - start_time
            status_msg = f"Updated in {processing_time:.2f}s - {len(self.bounding_boxes)} detections"
            self.update_status(status_msg, clear_after=3)
            
            # Update filter summary
            self.update_filter_summary()
            
            # Refresh the plot
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            print(f"Error updating display: {e}")
            self.update_status(f"Error: {str(e)}", clear_after=5)
        finally:
            self.processing = False
            
            # Process any pending updates
            if self.pending_update:
                self.pending_update = False
                threading.Thread(target=lambda: (time.sleep(0.1), self.update_display()), daemon=True).start()
    
    def load_image_button(self):
        """Button callback to load a new image."""
        if self.load_image():
            self.update_display()
    
    def reset_parameters(self):
        """Reset all parameters to original configuration."""
        self.config = copy.deepcopy(self.original_config)
        self.initialize_filters()
        
        # Update all controls
        if self.control_window:
            self.control_window.destroy()
            self.setup_control_window()
        
        self.update_display()
    
    def save_config(self):
        """Save current configuration to a file."""
        root = tk.Tk()
        root.withdraw()
        
        filename = filedialog.asksaveasfilename(
            title="Save Configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
        )
        
        root.destroy()
        
        if filename:
            try:
                with open(filename, 'w') as file:
                    yaml.dump(self.config, file, default_flow_style=False, indent=2)
                
                self.update_status(f"Configuration saved to {filename}", clear_after=3)
                print(f"Configuration saved to {filename}")
                
            except Exception as e:
                error_msg = f"Error saving configuration: {e}"
                self.update_status(error_msg, clear_after=5)
                print(error_msg)
    
    def run(self):
        """Run the interactive visualizer."""
        self.setup_visualization()
        plt.show()


def main():
    """Main function to run the interactive visualizer."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Interactive Annotation Pipeline Visualizer v2')
    parser.add_argument('--config', type=str, 
                       default='../data_yaml_files/annotation_20231102_aant_tank3_cslics06.yaml',
                       help='Path to the configuration YAML file')
    parser.add_argument('--image', type=str, help='Path to an image file to load initially')
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Configuration file not found: {args.config}")
        print("Please specify a valid configuration file with --config")
        return
    
    # Create and run the visualizer
    visualizer = InteractiveAnnotationVisualizerV2(args.config, args.image)
    visualizer.run()


if __name__ == "__main__":
    main()
