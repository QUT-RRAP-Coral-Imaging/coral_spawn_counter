#!/usr/bin/env python3

"""
Launch script for the Interactive Annotation Pipeline Visualizer v2

This script provides an easy way to launch the tabbed interface visualizer
with proper configuration and image selection.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

def main():
    """Main launcher function."""
    print("🚀 Interactive Annotation Pipeline Visualizer v2")
    print("=" * 60)
    
    # Create root window for file dialogs
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    
    # Select configuration file
    config_dir = os.path.join('..', 'data_yaml_files')
    config_path = filedialog.askopenfilename(
        title="Select Configuration File",
        initialdir=config_dir if os.path.exists(config_dir) else '.',
        filetypes=[("YAML files", "*.yaml"), ("All files", "*.*")]
    )
    
    if not config_path:
        messagebox.showinfo("Cancelled", "No configuration file selected. Exiting.")
        return
    
    # Ask if user wants to select an image now
    load_image = messagebox.askyesno(
        "Load Image", 
        "Would you like to select an image file now?\\n\\n" +
        "You can also load images later using the interface."
    )
    
    image_path = None
    if load_image:
        image_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
    
    root.destroy()
    
    # Launch the visualizer
    try:
        from interactive_annotation_visualizer_v2 import InteractiveAnnotationVisualizerV2
        
        print(f"📁 Configuration: {config_path}")
        if image_path:
            print(f"🖼️  Image: {image_path}")
        print("\\n⚡ Starting visualizer...")
        
        visualizer = InteractiveAnnotationVisualizerV2(config_path, image_path)
        visualizer.run()
        
    except ImportError as e:
        print(f"❌ Error importing visualizer: {e}")
        print("Make sure you're running this from the annotation_visualiser directory")
    except Exception as e:
        print(f"❌ Error starting visualizer: {e}")


if __name__ == "__main__":
    main()
