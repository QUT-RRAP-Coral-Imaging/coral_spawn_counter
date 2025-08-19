#!/usr/bin/env python3

"""
Launcher script for the Annotation Visualizers
"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import messagebox, filedialog


def check_dependencies():
    """Check if required dependencies are installed."""
    missing = []
    
    try:
        import cv2
    except ImportError:
        missing.append("opencv-python")
    
    try:
        import numpy
    except ImportError:
        missing.append("numpy")
    
    try:
        import yaml
    except ImportError:
        missing.append("PyYAML")
    
    try:
        import matplotlib
    except ImportError:
        missing.append("matplotlib")
    
    return missing


def install_dependencies(packages):
    """Install missing dependencies."""
    if not packages:
        return True
    
    answer = messagebox.askyesno(
        "Missing Dependencies", 
        f"The following packages are missing:\n{', '.join(packages)}\n\n"
        "Would you like to install them now?"
    )
    
    if answer:
        try:
            for package in packages:
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            messagebox.showinfo("Success", "Dependencies installed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Failed to install dependencies: {e}")
            return False
    
    return False


def select_config_file():
    """Select a configuration file."""
    # Default config file (relative to parent directory)
    script_dir = os.path.dirname(__file__)
    default_config = os.path.join(script_dir, "..", "data_yaml_files", "annotation_cslics_2024_nov_amil_tank3_10000000f620da42.yaml")
    default_config = os.path.abspath(default_config)
    
    if os.path.exists(default_config):
        use_default = messagebox.askyesno(
            "Configuration File",
            f"Use the default configuration file?\n{os.path.basename(default_config)}"
        )
        if use_default:
            return default_config
    
    # Select custom config file
    config_path = filedialog.askopenfilename(
        title="Select configuration file",
        filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        initialdir=os.path.join(script_dir, "..", "data_yaml_files")
    )
    
    return config_path


def launch_simple_visualizer(config_path):
    """Launch the simple visualizer."""
    try:
        # Import and run the simple visualizer
        sys.path.insert(0, os.path.dirname(__file__))
        from simple_annotation_visualizer import SimpleAnnotationVisualizer
        
        visualizer = SimpleAnnotationVisualizer(config_path)
        visualizer.run()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch simple visualizer: {e}")


def launch_full_visualizer(config_path):
    """Launch the full visualizer."""
    try:
        # Import and run the full visualizer
        sys.path.insert(0, os.path.dirname(__file__))
        from interactive_annotation_visualizer import InteractiveAnnotationVisualizer
        
        visualizer = InteractiveAnnotationVisualizer(config_path)
        visualizer.run()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch full visualizer: {e}")


def main():
    """Main launcher interface."""
    # Check dependencies
    missing_deps = check_dependencies()
    if missing_deps:
        if not install_dependencies(missing_deps):
            print("Cannot continue without required dependencies.")
            return
    
    # Create main window
    root = tk.Tk()
    root.title("Annotation Visualizer Launcher")
    root.geometry("500x400")
    
    # Title
    title_label = tk.Label(
        root, 
        text="Coral Spawn Annotation Visualizer", 
        font=("Arial", 16, "bold")
    )
    title_label.pack(pady=20)
    
    # Description
    desc_text = """
    Choose a visualizer to interactively adjust annotation parameters:
    
    • Simple Visualizer: Hue-based filtering with Tkinter GUI
      (Recommended for beginners, more stable)
    
    • Full Visualizer: All filters with Matplotlib interface
      (Requires all filter modules to be available)
    """
    
    desc_label = tk.Label(root, text=desc_text, justify=tk.LEFT)
    desc_label.pack(pady=10, padx=20)
    
    # Buttons frame
    button_frame = tk.Frame(root)
    button_frame.pack(pady=20)
    
    def launch_simple():
        """Launch simple visualizer with config selection."""
        config_path = select_config_file()
        if config_path:
            root.destroy()
            launch_simple_visualizer(config_path)
    
    def launch_full():
        """Launch full visualizer with config selection."""
        config_path = select_config_file()
        if config_path:
            root.destroy()
            launch_full_visualizer(config_path)
    
    def show_help():
        """Show help information."""
        help_text = """
        Simple Visualizer:
        - Easier to use, more stable GUI
        - Focuses on hue-based filtering
        - Good for beginners and quick parameter tuning
        
        Full Visualizer:
        - Access to all filter types (SIFT, Laplacian, etc.)
        - More advanced parameter control
        - Requires all filter modules to be available
        
        Both visualizers allow you to:
        - Load images and see real-time filtering effects
        - Adjust parameters with sliders and controls
        - Save optimized configurations for batch processing
        """
        messagebox.showinfo("Help", help_text)
    
    # Buttons
    tk.Button(
        button_frame, 
        text="Launch Simple Visualizer", 
        command=launch_simple,
        width=25, height=2,
        bg="lightgreen"
    ).pack(pady=5)
    
    tk.Button(
        button_frame, 
        text="Launch Full Visualizer", 
        command=launch_full,
        width=25, height=2,
        bg="lightblue"
    ).pack(pady=5)
    
    tk.Button(
        button_frame, 
        text="Help", 
        command=show_help,
        width=25, height=1
    ).pack(pady=5)
    
    tk.Button(
        button_frame, 
        text="Exit", 
        command=root.destroy,
        width=25, height=1
    ).pack(pady=5)
    
    # Status
    status_label = tk.Label(
        root, 
        text="Ready to launch visualizer", 
        font=("Arial", 10), 
        fg="gray"
    )
    status_label.pack(side=tk.BOTTOM, pady=10)
    
    # Center the window
    root.eval('tk::PlaceWindow . center')
    
    root.mainloop()


if __name__ == "__main__":
    main()
