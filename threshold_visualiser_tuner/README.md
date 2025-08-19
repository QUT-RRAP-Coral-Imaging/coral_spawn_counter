# Simple Filter Editor

An interactive GUI tool for tuning coral spawn detection filter parameters with real-time visual feedback.

## Overview

The Simple Filter Editor provides an intuitive interface for adjusting filter parameters in the coral spawn annotation pipeline. It features three synchronized image views and smooth real-time updates to help researchers optimize detection accuracy.

## Features

### 🖼️ **Three-Panel Image Display**
- **Final Combined Output**: Large main view showing detection results with bounding boxes
- **Original Image**: Reference view of the unprocessed image
- **Filter Overlay**: Original image with current filter effects highlighted in green
- **Filter Mask**: Binary mask output from the current filter

### 🎛️ **Interactive Controls**
- **Filter Selection**: Easy switching between filter types (Hue, SIFT, Saturation, Value, Laplacian, Edge)
- **Parameter Sliders**: Real-time adjustment with debounced updates for smooth interaction
- **Enable/Disable**: Individual filter toggle controls
- **Processing Options**: Fine-grained control over filter processing steps

### 💾 **Data Management**
- **YAML Configuration**: Load, save, and reset filter configurations
- **Image Export**: Save final detection results with bounding boxes
- **Multi-format Support**: Works with JPG, PNG, TIFF, and BMP images

### ⚡ **Performance Optimizations**
- **Background Processing**: Non-blocking filter computation in separate threads
- **Debounced Updates**: Smooth parameter adjustment without lag
- **Smart Resizing**: Automatic image scaling for optimal performance
- **Responsive UI**: Canvas resize handling and memory-efficient display

## Quick Start

### Launch the Editor
```bash
cd threshold_visualiser_tuner
python launch_simple_editor.py
```

### Or run directly
```bash
python simple_filter_editor.py
```

## Usage Guide

### 1. **Load an Image**
- Click "Load Image" and select your coral spawn image
- Supported formats: JPG, JPEG, PNG, BMP, TIFF

### 2. **Load Configuration**
- Click "Load Config" to import a YAML configuration file
- Default configs are available in `../data_yaml_files/`

### 3. **Select and Tune Filters**
- Click filter buttons (Hue, SIFT, etc.) to switch between filters
- Adjust parameters using sliders - changes update in real-time
- Enable/disable filters using checkboxes
- Watch the overlay image to see filter effects highlighted

### 4. **Generate Final Output**
- Click "Update Final Output" to combine all enabled filters
- View detection count and bounding boxes in the main display

### 5. **Save Results**
- **Save Config**: Export current parameter settings as YAML
- **Save Image**: Export final detection image with bounding boxes
- **Reset**: Restore original configuration settings

## Filter Types

### **Hue Filter**
- Color-based detection using HSV hue values
- Parameters: hue_min, hue_max, edge_dilation_kernel_size
- Best for: Distinct color targets

### **SIFT Filter**
- Scale-Invariant Feature Transform detection
- Parameters: contrast_threshold, edge_threshold, dilate
- Best for: Textural features and distinct shapes

### **Saturation Filter**
- HSV saturation-based detection
- Parameters: Standard morphological operations
- Best for: Color intensity discrimination

### **Value Filter**
- HSV value (brightness) based detection
- Parameters: value_min, value_max
- Best for: Brightness-based segmentation

### **Laplacian Filter**
- Edge detection using Laplacian operator
- Parameters: laplacian_threshold, laplacian_kernel_size
- Best for: Sharp edge detection

### **Edge Filter**
- Canny edge detection
- Parameters: canny_lower_thresh, canny_upper_thresh
- Best for: General edge detection

## Common Parameters

All filters support these common processing parameters:

- **Denoising**: `denoise_template_window_size`, `denoise_search_window_size`, `denoise_strength`
- **Area Filtering**: `min_area`, `max_area`
- **Shape Filtering**: `min_circularity`, `max_circularity`
- **Morphological**: `kernel_size`
- **Processing Steps**: `process_denoise`, `process_thresh`, `process_morph`, `process_fill`, `process_filter`

## Technical Details

### **Architecture**
- Built with Tkinter for cross-platform compatibility
- Uses OpenCV for image processing
- PIL/Pillow for image display
- Threading for responsive UI
- YAML for configuration management

### **Performance**
- Automatic image scaling (max 800px) for processing speed
- Background thread processing to maintain UI responsiveness
- Debounced parameter updates (100ms delay) for smooth interaction
- Memory-efficient image display with garbage collection protection

### **Dependencies**
```python
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk
import yaml
import threading
```

## File Structure

```
threshold_visualiser_tuner/
├── README.md                    # This file
├── __init__.py                  # Package initialization
├── simple_filter_editor.py     # Main application
├── launch_simple_editor.py     # Launcher script
└── __pycache__/                 # Python cache files
```

## Configuration Files

The editor works with YAML configuration files containing filter parameters. Example structure:

```yaml
hue:
  do: true
  hue_min: 60
  hue_max: 120
  denoise_strength: 3.0
  min_area: 500
  max_area: 5000
  # ... other parameters

sift:
  do: false
  contrast_threshold: 0.04
  edge_threshold: 10
  # ... other parameters
```

## Troubleshooting

### **Common Issues**

1. **Filter modules not found**
   - Ensure annotation directory filters are available
   - Check Python path includes annotation directory

2. **Image won't load**
   - Verify image format is supported
   - Check file permissions

3. **Slow performance**
   - Large images are automatically scaled
   - Close other applications if needed
   - Consider using smaller test images

4. **UI doesn't update**
   - Check that image is loaded first
   - Verify filter is enabled
   - Try refreshing the filter manually

### **Performance Tips**

- Use images under 2000x2000 pixels for best performance
- Enable only necessary filters for faster processing
- Save configurations frequently during tuning sessions
- Use the reset button to restore known good settings

## Development

### **Extending the Editor**

To add new filter types:

1. Create filter class in annotation directory
2. Add to `filter_classes` dictionary in `SimpleFilterEditor.__init__`
3. Add parameter definitions in `create_filter_specific_controls`
4. Add processing logic in `_process_filter`

### **Customization**

The editor can be customized by modifying:
- Window size: `self.root.geometry()`
- Canvas sizes: Canvas width/height parameters
- Update delay: `self.update_timer` delay value
- Image scaling: `self.max_image_size`

## License

Part of the Coral Spawn Counter project.

## Authors

Coral Spawn Counter Team - 2025
