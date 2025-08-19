"""
Simple Filter Editor Package

Interactive visualization tool for the coral spawn annotation pipeline.

This package contains:
- SimpleFilterEditor: Tkinter-based GUI for filter parameter tuning with real-time preview
- Launch script: Easy launcher for the filter editor

Features:
- Real-time filter preview with original, overlay, and mask views
- Interactive parameter controls with debounced updates
- Support for all filter types (hue, sift, saturation, value, laplacian, edge)
- Combined output generation with bounding box detection
- YAML configuration save/load
- Result image export

Usage:
    cd threshold_visualiser_tuner
    python launch_simple_editor.py
    
    or
    
    python simple_filter_editor.py
"""

__version__ = "2.0.0"
__author__ = "Coral Spawn Counter Team"

# Make classes available at package level
try:
    from .simple_filter_editor import SimpleFilterEditor
except ImportError:
    pass
