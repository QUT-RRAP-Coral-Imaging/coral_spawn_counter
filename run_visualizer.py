#!/usr/bin/env python3

"""
Convenience launcher for the Annotation Visualizer
Run this from the main coral_spawn_counter directory.
"""

import os
import sys
import subprocess

def main():
    """Launch the annotation visualizer from the parent directory."""
    script_dir = os.path.dirname(__file__)
    visualizer_dir = os.path.join(script_dir, "annotation_visualiser")
    
    if not os.path.exists(visualizer_dir):
        print("❌ annotation_visualiser directory not found!")
        print("Make sure you're running this from the coral_spawn_counter directory.")
        return
    
    print("🚀 Launching Annotation Visualizer...")
    print(f"Changing to directory: {visualizer_dir}")
    
    try:
        # Change to the visualizer directory and run the launcher
        launcher_script = os.path.join(visualizer_dir, "launch_visualizer.py")
        subprocess.run([sys.executable, launcher_script], cwd=visualizer_dir)
    except Exception as e:
        print(f"❌ Error launching visualizer: {e}")
        print("\nAlternatively, you can run manually:")
        print(f"  cd {visualizer_dir}")
        print("  python launch_visualizer.py")

if __name__ == "__main__":
    main()
