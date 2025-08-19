#!/usr/bin/env python3

"""
Launcher for Simple Filter Editor

This script launches the simple filter editor with default configuration.
"""

import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

def main():
    """Launch the simple filter editor."""
    print("🎯 Simple Filter Editor - Coral Spawn Counter")
    print("=" * 60)
    
    # Get the script directory
    script_dir = os.path.dirname(__file__)
    
    # Default paths
    default_config = os.path.join(script_dir, "..", "data_yaml_files", 
                                 "annotation_cslics_2024_oct_maeq_tank5_100000000846a7ff.yaml")
    default_config = os.path.abspath(default_config)
    
    default_image_dir = "/home/java/hpc-home/Data/cslics/2024_spawn_tanks_data"
    
    # Check if default config exists
    config_path = default_config
    if not os.path.exists(config_path):
        print(f"❌ Default config not found: {config_path}")
        print("📁 Please select a configuration file...")
        
        root = tk.Tk()
        root.withdraw()
        config_path = filedialog.askopenfilename(
            title="Select configuration file",
            filetypes=[("YAML files", "*.yaml *.yml")]
        )
        root.destroy()
        
        if not config_path:
            print("❌ No configuration file selected. Exiting.")
            return
    
    print(f"📁 Configuration: {config_path}")
    
    # Check for sample images
    image_path = None
    if os.path.exists(default_image_dir):
        # Look for sample images
        sample_paths = [
            "/home/java/hpc-home/Data/cslics/2024_spawn_tanks_data/maeq/maeq_oct_438d/subfolder_01/2024-10-27_04-45-19_clean.jpg",
            "/home/java/hpc-home/Data/cslics/2024_spawn_tanks_data/maeq/maeq_oct_438d/subfolder_02/2024-10-27_13-39-49_clean.jpg"
        ]
        
        for path in sample_paths:
            if os.path.exists(path):
                image_path = path
                break
    
    if image_path:
        print(f"🖼️  Image: {image_path}")
    else:
        print("🖼️  Image: Will be selected in the editor")
    
    print("\\n⚡ Starting editor...")
    
    try:
        # Import and run the editor
        from simple_filter_editor import SimpleFilterEditor
        
        editor = SimpleFilterEditor(config_path, image_path)
        editor.run()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running this from the annotation_visualiser directory")
    except Exception as e:
        print(f"❌ Error starting editor: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
