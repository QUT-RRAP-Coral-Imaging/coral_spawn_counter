#!/usr/bin/env python3

"""
Simple Filter Editor - Interactive Annotation Pipeline Parameter Tuner

This provides a clean interface with:
- Original image on left, current filter effect on right
- Buttons to switch between filters
- Parameter controls for each filter
- Final output preview
- Save YAML and image functionality
"""

import os
import sys
import yaml
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import copy
import threading
import time

# Add annotation directory to path
script_dir = os.path.dirname(__file__)
annotation_dir = os.path.join(script_dir, '..', 'annotation')
annotation_dir = os.path.abspath(annotation_dir)
sys.path.insert(0, annotation_dir)

# Import filters
try:
    from FilterSift import FilterSift   
    from FilterHue import FilterHue
    from FilterSaturation import FilterSaturation
    from FilterLaplacian import FilterLaplacian
    from FilterValue import FilterValue
    from FilterEdge import FilterEdge
except ImportError as e:
    print(f"Warning: Could not import filter modules: {e}")
    FilterSift = FilterHue = FilterSaturation = FilterLaplacian = FilterValue = FilterEdge = None


class SimpleFilterEditor:
    def __init__(self, config_path, image_path=None):
        """Initialize the simple filter editor."""
        # Load configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
        
        self.original_config = copy.deepcopy(self.config)
        self.config_path = config_path
        self.image_path = image_path
        self.image = None
        self.image_small = None
        self.overlay_image = None  # For storing the overlay image
        self.scale_factor = 1.0
        self.max_image_size = 800
        
        # Current filter state
        self.current_filter = 'hue'  # Default filter
        self.current_mask = None
        self.current_combined_mask = None
        self.filter_widgets = {}
        self.update_timer = None  # For debouncing parameter updates
        
        # Available filters
        self.filter_classes = {
            'hue': FilterHue,
            'sift': FilterSift,
            'saturation': FilterSaturation,
            'value': FilterValue,
            'laplacian': FilterLaplacian,
            'edge': FilterEdge
        }
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Simple Filter Editor - Coral Spawn Counter")
        self.root.geometry("1500x1000")  # Increased height for three stacked images
        
        self.setup_ui()
        
        if image_path:
            self.load_image(image_path)
    
    def setup_ui(self):
        """Set up the user interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Top toolbar
        toolbar = ttk.Frame(main_frame)
        toolbar.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(toolbar, text="Load Image", command=self.load_image_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Load Config", command=self.load_config_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Save Config", command=self.save_config_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Save Image", command=self.save_image_dialog).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(toolbar, text="Reset", command=self.reset_config).pack(side=tk.LEFT, padx=(0, 5))
        
        # Status label
        self.status_label = ttk.Label(toolbar, text="Ready", foreground="blue")
        self.status_label.pack(side=tk.RIGHT)
        
        # Main content area
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Large final output
        left_frame = ttk.Frame(content_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        # Final output (large, main display)
        ttk.Label(left_frame, text="Final Combined Output", font=("Arial", 14, "bold")).pack(pady=(0, 5))
        self.output_canvas = tk.Canvas(left_frame, bg="gray")
        self.output_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Right side - Three smaller images stacked vertically
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 10))
        
        # Original image (top)
        ttk.Label(right_frame, text="Original Image", font=("Arial", 11, "bold")).pack(pady=(0, 3))
        self.original_canvas = tk.Canvas(right_frame, bg="gray", width=350, height=200)
        self.original_canvas.pack(pady=(0, 10))
        
        # Filter overlay on original (middle)
        self.overlay_label = ttk.Label(right_frame, text="Filter Overlay", font=("Arial", 11, "bold"))
        self.overlay_label.pack(pady=(0, 3))
        self.overlay_canvas = tk.Canvas(right_frame, bg="gray", width=350, height=200)
        self.overlay_canvas.pack(pady=(0, 10))
        
        # Current filter mask result (bottom)
        self.filter_label = ttk.Label(right_frame, text="Filter: Hue", font=("Arial", 11, "bold"))
        self.filter_label.pack(pady=(0, 3))
        self.filter_canvas = tk.Canvas(right_frame, bg="gray", width=350, height=200)
        self.filter_canvas.pack()
        
        # Far right side - Controls
        control_frame = ttk.Frame(content_frame)
        control_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        control_frame.config(width=350)  # Fixed width for controls
        
        # Filter selection buttons
        filter_button_frame = ttk.LabelFrame(control_frame, text="Select Filter", padding=10)
        filter_button_frame.pack(fill=tk.X, pady=(0, 10))
        
        self.filter_buttons = {}
        for i, (filter_name, filter_class) in enumerate(self.filter_classes.items()):
            if filter_name in self.config:
                btn = ttk.Button(filter_button_frame, text=filter_name.capitalize(), 
                               command=lambda fn=filter_name: self.switch_filter(fn))
                btn.grid(row=i//2, column=i%2, sticky="ew", padx=2, pady=2)
                self.filter_buttons[filter_name] = btn
        
        # Configure grid weights
        filter_button_frame.columnconfigure(0, weight=1)
        filter_button_frame.columnconfigure(1, weight=1)
        
        # Filter parameters
        self.params_frame = ttk.LabelFrame(control_frame, text="Filter Parameters", padding=10)
        self.params_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Action buttons
        action_frame = ttk.Frame(control_frame)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Button(action_frame, text="Refresh Filter", command=self.refresh_filter).pack(fill=tk.X, pady=(0, 5))
        ttk.Button(action_frame, text="Update Final Output", command=self.update_final_output).pack(fill=tk.X)
        
        # Load initial filter parameters
        self.update_filter_controls()
    
    def load_image_dialog(self):
        """Open dialog to load an image."""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if file_path:
            self.load_image(file_path)
    
    def load_image(self, image_path):
        """Load an image from file."""
        try:
            self.image_path = image_path
            self.image = cv.imread(image_path)
            
            if self.image is None:
                messagebox.showerror("Error", f"Could not load image: {image_path}")
                return
            
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
            
            self.display_original_image()
            self.refresh_filter()
            self.status_label.config(text=f"Loaded: {os.path.basename(image_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_original_image(self):
        """Display the original image in the canvas."""
        if self.image_small is None:
            return
        
        self._display_image_in_canvas(self.image_small, self.original_canvas, 'original_photo')
    
    def display_overlay_image(self):
        """Display the original image with the current filter's effects highlighted as an overlay."""
        if self.image_small is None or self.current_mask is None:
            # If no mask available yet, just display the original image in overlay canvas
            if self.image_small is not None:
                self._display_image_in_canvas(self.image_small, self.overlay_canvas, 'overlay_photo')
            return
        
        try:
            # Create overlay: original image with filter mask highlighted
            overlay = self.image_small.copy()
            
            # Create colored overlay for the filter (green tint with transparency)
            colored_mask = np.zeros_like(overlay)
            colored_mask[:, :, 1] = 255  # Green channel
            
            # Apply mask to colored overlay
            mask_3channel = cv.cvtColor(self.current_mask, cv.COLOR_GRAY2BGR)
            mask_normalized = mask_3channel.astype(np.float32) / 255.0
            
            # Blend: original image + semi-transparent colored mask where filter is active
            alpha = 0.4  # Slightly increased transparency for better visibility
            overlay = overlay.astype(np.float32)
            colored_overlay = colored_mask.astype(np.float32) * mask_normalized * alpha
            overlay = overlay + colored_overlay
            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
            
            # Display the overlay image
            self._display_image_in_canvas(overlay, self.overlay_canvas, 'overlay_photo')
            
        except Exception as e:
            print(f"Error displaying overlay: {e}")
    
    def _display_image_in_canvas(self, image, canvas, photo_attr_name):
        """Helper method to display an image in a canvas."""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            else:
                image_rgb = cv.cvtColor(image, cv.COLOR_GRAY2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(image_rgb)
            
            # Resize to fit canvas
            canvas_width = canvas.winfo_width()
            canvas_height = canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            # Store photo reference to prevent garbage collection
            photo = ImageTk.PhotoImage(pil_image)
            setattr(self, photo_attr_name, photo)
            
            # Clear and display
            canvas.delete("all")
            canvas.create_image(
                canvas.winfo_width()//2,
                canvas.winfo_height()//2,
                image=photo
            )
            
        except Exception as e:
            print(f"Error displaying image in canvas: {e}")
    
    def switch_filter(self, filter_name):
        """Switch to a different filter."""
        self.current_filter = filter_name
        self.filter_label.config(text=f"Filter: {filter_name.capitalize()}")
        self.overlay_label.config(text=f"Overlay: {filter_name.capitalize()}")
        self.update_filter_controls()
        self.refresh_filter()
        
        # Update button styles
        for name, btn in self.filter_buttons.items():
            if name == filter_name:
                btn.state(['pressed'])
            else:
                btn.state(['!pressed'])
    
    def update_filter_controls(self):
        """Update the parameter controls for the current filter."""
        # Clear existing controls
        for widget in self.params_frame.winfo_children():
            widget.destroy()
        
        self.filter_widgets[self.current_filter] = {}
        
        if self.current_filter not in self.config:
            ttk.Label(self.params_frame, text="Filter not configured").pack()
            return
        
        filter_config = self.config[self.current_filter]
        
        # Create scrollable frame
        canvas = tk.Canvas(self.params_frame)
        scrollbar = ttk.Scrollbar(self.params_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        row = 0
        
        # Enable/disable checkbox
        var = tk.BooleanVar(value=filter_config.get('do', True))
        ttk.Checkbutton(scrollable_frame, text="Enable Filter", variable=var,
                       command=lambda: self.update_boolean_param('do', var.get())).grid(
                           row=row, column=0, columnspan=2, sticky='w', pady=5)
        self.filter_widgets[self.current_filter]['do'] = var
        row += 1
        
        # Parameter controls based on filter type
        self.create_filter_specific_controls(scrollable_frame, filter_config, row)
    
    def create_filter_specific_controls(self, parent, filter_config, start_row):
        """Create filter-specific parameter controls."""
        row = start_row
        
        # Common parameters for all filters
        common_params = [
            ('denoise_template_window_size', 'Denoise Template Size', 1, 21, 2),
            ('denoise_search_window_size', 'Denoise Search Size', 1, 41, 2),
            ('denoise_strength', 'Denoise Strength', 0.1, 10.0, 0.1),
            ('min_area', 'Min Area', 100, 50000, 100),
            ('max_area', 'Max Area', 1000, 100000, 100),
            ('min_circularity', 'Min Circularity', 0.0, 1.0, 0.01),
            ('max_circularity', 'Max Circularity', 0.0, 1.0, 0.01),
            ('kernel_size', 'Kernel Size', 1, 51, 2),
        ]
        
        # Filter-specific parameters
        if self.current_filter == 'hue':
            specific_params = [
                ('hue_min', 'Hue Min', 0, 179, 1),
                ('hue_max', 'Hue Max', 0, 179, 1),
                ('edge_dilation_kernel_size', 'Edge Dilation Kernel', 1, 101, 2),
            ]
        elif self.current_filter == 'sift':
            specific_params = [
                ('contrast_threshold', 'Contrast Threshold', 0.001, 0.2, 0.001),
                ('edge_threshold', 'Edge Threshold', 1, 200, 1),
                ('dilate', 'Dilate', 1, 100, 1),
            ]
        elif self.current_filter == 'laplacian':
            specific_params = [
                ('laplacian_threshold', 'Laplacian Threshold', 1, 100, 1),
                ('laplacian_kernel_size', 'Laplacian Kernel Size', 1, 15, 2),
            ]
        elif self.current_filter == 'saturation':
            specific_params = []
        elif self.current_filter == 'value':
            specific_params = [
                ('value_min', 'Value Min', 0, 255, 1),
                ('value_max', 'Value Max', 0, 255, 1),
            ]
        elif self.current_filter == 'edge':
            specific_params = [
                ('canny_lower_thresh', 'Canny Lower Threshold', 1, 100, 1),
                ('canny_upper_thresh', 'Canny Upper Threshold', 1, 200, 1),
            ]
        else:
            specific_params = []
        
        # Combine all parameters
        all_params = specific_params + common_params
        
        # Create controls for each parameter
        for param_name, display_name, min_val, max_val, step in all_params:
            if param_name in filter_config:
                current_value = filter_config[param_name]
                
                # Parameter label
                ttk.Label(parent, text=display_name).grid(row=row, column=0, sticky='w', pady=2)
                
                # Value display
                value_var = tk.StringVar(value=str(current_value))
                value_label = ttk.Label(parent, textvariable=value_var, width=10, 
                                      background='white', relief='sunken')
                value_label.grid(row=row, column=1, sticky='e', pady=2, padx=(5, 0))
                
                # Slider
                if step < 1:
                    # Float slider
                    slider = ttk.Scale(parent, from_=min_val, to=max_val, 
                                     orient=tk.HORIZONTAL, length=200,
                                     command=lambda val, p=param_name, v=value_var: self.update_param(p, float(val), v))
                    slider.set(current_value)
                else:
                    # Integer slider
                    slider = ttk.Scale(parent, from_=min_val, to=max_val, 
                                     orient=tk.HORIZONTAL, length=200,
                                     command=lambda val, p=param_name, v=value_var: self.update_param(p, int(float(val)), v))
                    slider.set(current_value)
                
                slider.grid(row=row+1, column=0, columnspan=2, sticky='ew', pady=2)
                
                self.filter_widgets[self.current_filter][param_name] = (slider, value_var)
                row += 2
        
        # Boolean parameters
        boolean_params = ['process_denoise', 'process_thresh', 'process_morph', 'process_fill', 'process_filter']
        
        if row > start_row:
            ttk.Separator(parent, orient='horizontal').grid(row=row, column=0, columnspan=2, sticky='ew', pady=10)
            row += 1
        
        ttk.Label(parent, text="Processing Options", font=("Arial", 10, "bold")).grid(row=row, column=0, columnspan=2, sticky='w', pady=5)
        row += 1
        
        for param_name in boolean_params:
            if param_name in filter_config:
                var = tk.BooleanVar(value=filter_config[param_name])
                ttk.Checkbutton(parent, text=param_name.replace('_', ' ').title(), variable=var,
                               command=lambda p=param_name, v=var: self.update_boolean_param(p, v.get())).grid(
                                   row=row, column=0, columnspan=2, sticky='w', pady=2)
                self.filter_widgets[self.current_filter][param_name] = var
                row += 1
    
    def update_param(self, param_name, value, value_var):
        """Update a parameter value with debouncing."""
        # Ensure integer parameters remain integers
        if param_name in ['dilate', 'kernel_size', 'hue_min', 'hue_max', 'edge_dilation_kernel_size',
                         'edge_threshold', 'laplacian_threshold', 'laplacian_kernel_size',
                         'min_area', 'max_area', 'denoise_template_window_size', 'denoise_search_window_size',
                         'value_min', 'value_max', 'canny_lower_thresh', 'canny_upper_thresh']:
            value = int(value)
        
        self.config[self.current_filter][param_name] = value
        value_var.set(str(value))
        
        # Debounce the filter update for smoother interaction
        if self.update_timer:
            self.root.after_cancel(self.update_timer)
        self.update_timer = self.root.after(100, self.refresh_filter)  # 100ms delay
    
    def update_boolean_param(self, param_name, value):
        """Update a boolean parameter."""
        self.config[self.current_filter][param_name] = value
    
    def refresh_filter(self):
        """Refresh the current filter display."""
        if self.image_small is None:
            return
        
        self.status_label.config(text="Processing filter...")
        self.root.update_idletasks()
        
        threading.Thread(target=self._process_filter, daemon=True).start()
    
    def _process_filter(self):
        """Process the current filter (runs in background thread)."""
        try:
            if self.current_filter not in self.config or not self.config[self.current_filter].get('do', True):
                self.current_mask = np.zeros(self.image_small.shape[:2], dtype=np.uint8)
            else:
                filter_class = self.filter_classes[self.current_filter]
                if filter_class:
                    filter_obj = filter_class(config=self.config[self.current_filter])
                    
                    if self.current_filter == 'hue':
                        self.current_mask = filter_obj.create_hue_mask(self.image_small)
                    elif self.current_filter == 'sift':
                        kp = filter_obj.get_best_sift_features(self.image_small)
                        self.current_mask = filter_obj.create_sift_mask(self.image_small, kp)
                    elif self.current_filter == 'saturation':
                        self.current_mask = filter_obj.create_saturation_mask(self.image_small)
                    elif self.current_filter == 'value':
                        self.current_mask = filter_obj.create_value_mask(self.image_small)
                    elif self.current_filter == 'laplacian':
                        self.current_mask = filter_obj.create_laplacian_mask(self.image_small)
                    elif self.current_filter == 'edge':
                        self.current_mask = filter_obj.create_edge_mask(self.image_small)
                    else:
                        self.current_mask = np.zeros(self.image_small.shape[:2], dtype=np.uint8)
                else:
                    self.current_mask = np.zeros(self.image_small.shape[:2], dtype=np.uint8)
            
            # Schedule UI update in main thread
            self.root.after(0, self._update_filter_display)
            
        except Exception as e:
            print(f"Error processing filter: {e}")
            self.current_mask = np.zeros(self.image_small.shape[:2], dtype=np.uint8)
            error_msg = str(e)
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {error_msg}"))
    
    def _update_filter_display(self):
        """Update the filter display (runs in main thread)."""
        if self.current_mask is None:
            return
        
        try:
            # Display the filter mask
            self._display_image_in_canvas(self.current_mask, self.filter_canvas, 'filter_photo')
            
            # Also update overlay image
            self.display_overlay_image()
            
            self.status_label.config(text="Filter updated")
            
        except Exception as e:
            print(f"Error updating display: {e}")
            self.status_label.config(text=f"Display error: {str(e)}")
    
    def update_final_output(self):
        """Update the final combined output."""
        if self.image_small is None:
            return
        
        self.status_label.config(text="Generating final output...")
        self.root.update_idletasks()
        
        threading.Thread(target=self._process_final_output, daemon=True).start()
    
    def _process_final_output(self):
        """Process the final combined output (runs in background thread)."""
        try:
            masks = []
            
            # Process all enabled filters
            for filter_name, filter_class in self.filter_classes.items():
                if (filter_name in self.config and 
                    self.config[filter_name].get('do', False) and 
                    filter_class):
                    
                    filter_obj = filter_class(config=self.config[filter_name])
                    
                    if filter_name == 'hue':
                        mask = filter_obj.create_hue_mask(self.image_small)
                    elif filter_name == 'sift':
                        kp = filter_obj.get_best_sift_features(self.image_small)
                        mask = filter_obj.create_sift_mask(self.image_small, kp)
                    elif filter_name == 'saturation':
                        mask = filter_obj.create_saturation_mask(self.image_small)
                    elif filter_name == 'value':
                        mask = filter_obj.create_value_mask(self.image_small)
                    elif filter_name == 'laplacian':
                        mask = filter_obj.create_laplacian_mask(self.image_small)
                    elif filter_name == 'edge':
                        mask = filter_obj.create_edge_mask(self.image_small)
                    else:
                        continue
                    
                    masks.append(mask)
            
            # Combine masks
            if masks:
                self.current_combined_mask = masks[0].copy()
                for mask in masks[1:]:
                    self.current_combined_mask = cv.bitwise_and(self.current_combined_mask, mask)
            else:
                self.current_combined_mask = np.zeros(self.image_small.shape[:2], dtype=np.uint8)
            
            # Schedule UI update in main thread
            self.root.after(0, self._update_final_display)
            
        except Exception as e:
            print(f"Error processing final output: {e}")
            error_msg = str(e)
            self.root.after(0, lambda: self.status_label.config(text=f"Error: {error_msg}"))
    
    def _update_final_display(self):
        """Update the final output display (runs in main thread)."""
        if self.current_combined_mask is None:
            return
        
        try:
            # Create result image with bounding boxes
            result_image = self.image_small.copy()
            
            # Find contours and draw bounding boxes
            contours, _ = cv.findContours(self.current_combined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv.boundingRect(contour)
                cv.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Convert BGR to RGB
            result_rgb = cv.cvtColor(result_image, cv.COLOR_BGR2RGB)
            
            # Convert to PIL Image
            pil_image = Image.fromarray(result_rgb)
            
            # Resize to fit canvas
            canvas_width = self.output_canvas.winfo_width()
            canvas_height = self.output_canvas.winfo_height()
            if canvas_width > 1 and canvas_height > 1:
                pil_image.thumbnail((canvas_width, canvas_height), Image.Resampling.LANCZOS)
            
            self.output_photo = ImageTk.PhotoImage(pil_image)
            
            # Clear and display
            self.output_canvas.delete("all")
            self.output_canvas.create_image(
                self.output_canvas.winfo_width()//2,
                self.output_canvas.winfo_height()//2,
                image=self.output_photo
            )
            
            detection_count = len(contours)
            self.status_label.config(text=f"Final output: {detection_count} detections")
            
        except Exception as e:
            print(f"Error updating final display: {e}")
            self.status_label.config(text=f"Display error: {str(e)}")
    
    def load_config_dialog(self):
        """Load a new configuration file."""
        file_path = filedialog.askopenfilename(
            title="Select configuration file",
            filetypes=[("YAML files", "*.yaml *.yml")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as file:
                    self.config = yaml.safe_load(file)
                self.original_config = copy.deepcopy(self.config)
                self.config_path = file_path
                self.update_filter_controls()
                self.status_label.config(text=f"Config loaded: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load config: {str(e)}")
    
    def save_config_dialog(self):
        """Save the current configuration."""
        file_path = filedialog.asksaveasfilename(
            title="Save configuration",
            defaultextension=".yaml",
            filetypes=[("YAML files", "*.yaml *.yml")]
        )
        if file_path:
            try:
                with open(file_path, 'w') as file:
                    yaml.dump(self.config, file, default_flow_style=False)
                messagebox.showinfo("Success", f"Configuration saved to {file_path}")
                self.status_label.config(text=f"Config saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save config: {str(e)}")
    
    def save_image_dialog(self):
        """Save the current result image."""
        if self.current_combined_mask is None:
            messagebox.showwarning("Warning", "No result image to save. Generate final output first.")
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save result image",
            defaultextension=".jpg",
            filetypes=[("JPEG files", "*.jpg"), ("PNG files", "*.png")]
        )
        if file_path:
            try:
                # Create result image with bounding boxes
                result_image = self.image.copy()  # Use full-size image
                
                # Scale mask back to original size
                if self.scale_factor != 1.0:
                    full_mask = cv.resize(self.current_combined_mask, 
                                        (self.image.shape[1], self.image.shape[0]))
                else:
                    full_mask = self.current_combined_mask
                
                # Find contours and draw bounding boxes
                contours, _ = cv.findContours(full_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    x, y, w, h = cv.boundingRect(contour)
                    cv.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                cv.imwrite(file_path, result_image)
                messagebox.showinfo("Success", f"Image saved to {file_path}")
                self.status_label.config(text=f"Image saved: {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save image: {str(e)}")
    
    def reset_config(self):
        """Reset configuration to original values."""
        self.config = copy.deepcopy(self.original_config)
        self.update_filter_controls()
        self.refresh_filter()
        self.status_label.config(text="Configuration reset")
    
    def run(self):
        """Start the application."""
        # Bind canvas resize events
        self.original_canvas.bind('<Configure>', lambda e: self.display_original_image())
        self.overlay_canvas.bind('<Configure>', lambda e: self.display_overlay_image())
        self.filter_canvas.bind('<Configure>', lambda e: self._update_filter_display())
        self.output_canvas.bind('<Configure>', lambda e: self._update_final_display())
        
        self.root.mainloop()


def main():
    """Main function."""
    # Default configuration file
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
            filetypes=[("YAML files", "*.yaml *.yml")]
        )
        root.destroy()
        if not config_path:
            print("No configuration file selected. Exiting.")
            return
    
    try:
        # Create and run the editor
        editor = SimpleFilterEditor(config_path)
        print("Starting Simple Filter Editor...")
        editor.run()
        
    except Exception as e:
        print(f"Error running editor: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
