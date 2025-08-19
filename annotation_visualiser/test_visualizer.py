#!/usr/bin/env python3

"""
Test script to verify the visualizer components work correctly.
"""

import os
import sys
import yaml

def test_config_loading():
    """Test loading the configuration file."""
    # Default config file (relative to parent directory)
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, "..", "data_yaml_files", "annotation_cslics_2024_nov_amil_tank3_10000000f620da42.yaml")
    config_path = os.path.abspath(config_path)
    
    print(f"Testing config loading from: {config_path}")
    
    if not os.path.exists(config_path):
        print("❌ Configuration file not found!")
        return False
    
    try:
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        print("✅ Configuration loaded successfully!")
        
        # Check key sections
        required_sections = ['run', 'class', 'hue', 'sift', 'laplacian']
        for section in required_sections:
            if section in config:
                print(f"✅ Found section: {section}")
            else:
                print(f"❌ Missing section: {section}")
        
        # Print some key parameters
        print("\nKey parameters:")
        if 'hue' in config:
            print(f"  Hue range: {config['hue']['hue_min']}-{config['hue']['hue_max']}")
            print(f"  Area range: {config['hue']['min_area']}-{config['hue']['max_area']}")
        
        if 'class' in config:
            print(f"  Class: {config['class']['name']} (label: {config['class']['label']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False


def test_dependencies():
    """Test if required dependencies are available."""
    print("\nTesting dependencies:")
    
    # Test basic dependencies
    try:
        import numpy
        print("✅ numpy available")
    except ImportError:
        print("❌ numpy not available")
    
    try:
        import yaml
        print("✅ PyYAML available")
    except ImportError:
        print("❌ PyYAML not available")
    
    try:
        import cv2
        print("✅ OpenCV available")
        print(f"   OpenCV version: {cv2.__version__}")
    except ImportError:
        print("❌ OpenCV not available")
    
    try:
        import matplotlib
        print("✅ matplotlib available")
        print(f"   matplotlib version: {matplotlib.__version__}")
    except ImportError:
        print("❌ matplotlib not available (optional)")
    
    try:
        import tkinter
        print("✅ tkinter available")
    except ImportError:
        print("❌ tkinter not available")


def test_annotation_modules():
    """Test if annotation modules are available."""
    print("\nTesting annotation modules:")
    
    script_dir = os.path.dirname(__file__)
    annotation_dir = os.path.join(script_dir, "..", "annotation")
    annotation_dir = os.path.abspath(annotation_dir)
    
    if not os.path.exists(annotation_dir):
        print(f"❌ Annotation directory not found: {annotation_dir}")
        return False
    
    print(f"✅ Annotation directory found: {annotation_dir}")
    
    # Check for key filter files
    filter_files = [
        'FilterHue.py',
        'FilterSift.py', 
        'FilterLaplacian.py',
        'FilterValue.py',
        'FilterSaturation.py',
        'FilterCommon.py'
    ]
    
    for filter_file in filter_files:
        filter_path = os.path.join(annotation_dir, filter_file)
        if os.path.exists(filter_path):
            print(f"✅ Found: {filter_file}")
        else:
            print(f"❌ Missing: {filter_file}")


def main():
    """Run all tests."""
    print("🧪 Testing Annotation Visualizer Components")
    print("=" * 50)
    
    # Test 1: Configuration loading
    config_ok = test_config_loading()
    
    # Test 2: Dependencies
    test_dependencies()
    
    # Test 3: Annotation modules
    test_annotation_modules()
    
    print("\n" + "=" * 50)
    if config_ok:
        print("✅ Basic setup looks good!")
        print("\nTo run the visualizer:")
        print("  python launch_visualizer.py")
        print("  or")
        print("  python simple_annotation_visualizer.py")
    else:
        print("❌ Some issues found. Please check the configuration file.")
    
    print("\nInstall missing dependencies with:")
    print("  pip install -r requirements_visualizer.txt")


if __name__ == "__main__":
    main()
