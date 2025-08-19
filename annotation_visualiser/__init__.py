"""
Annotation Visualiser Package

Interactive visualization tools for the coral spawn annotation pipeline.

This package contains:
- SimpleAnnotationVisualizer: Tkinter-based GUI for hue filter parameter tuning
- InteractiveAnnotationVisualizer: Matplotlib-based interface for all filters
- Launcher script: Menu-driven interface for selecting visualizers

Usage:
    cd annotation_visualiser
    python launch_visualizer.py
    
    or
    
    python simple_annotation_visualizer.py
"""

__version__ = "1.0.0"
__author__ = "Coral Spawn Counter Team"

# Make classes available at package level
try:
    from .simple_annotation_visualizer import SimpleAnnotationVisualizer
except ImportError:
    pass

try:
    from .interactive_annotation_visualizer import InteractiveAnnotationVisualizer
except ImportError:
    pass
