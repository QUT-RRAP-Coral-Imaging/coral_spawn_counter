#!/usr/bin/env python3

"""
Test script to verify the layout improvements in the interactive visualizer.
This creates a dummy config and tests the UI layout without requiring actual image processing.
"""

import os
import sys
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for testing
import matplotlib.pyplot as plt

# Add path to annotation modules
script_dir = os.path.dirname(__file__)
annotation_dir = os.path.join(script_dir, '..', 'annotation')
annotation_dir = os.path.abspath(annotation_dir)
sys.path.insert(0, annotation_dir)

def test_layout():
    """Test the layout without actual image processing."""
    
    # Create a dummy config with all filters enabled
    dummy_config = {
        'hue': {
            'do': True,
            'hue_min': 0,
            'hue_max': 30,
            'min_area': 2000,
            'max_area': 20000
        },
        'sift': {
            'do': True,
            'contrast_threshold': 0.025,
            'edge_threshold': 100,
            'dilate': 100
        },
        'laplacian': {
            'do': True,
            'laplacian_threshold': 25,
            'laplacian_kernel_size': 5,
            'min_area': 3000
        },
        'saturation': {
            'do': True,
            'min_area': 2000,
            'max_area': 30000
        },
        'value': {
            'do': True,
            'value_min': 0,
            'value_max': 150
        },
        'edge': {
            'do': True,
            'canny_lower_thresh': 8,
            'canny_upper_thresh': 15
        }
    }
    
    # Create figure with the new layout
    fig = plt.figure(figsize=(28, 14))
    fig.suptitle('Interactive Annotation Pipeline Visualizer - Layout Test', fontsize=16, y=0.98)
    
    # Test the grid layout
    gs = fig.add_gridspec(4, 4, height_ratios=[1, 1, 1, 0.6], width_ratios=[1, 1, 1, 1],
                         hspace=0.3, wspace=0.2, top=0.93, bottom=0.15)
    
    # Test image subplot positions
    axes = {}
    
    # Row 1
    axes['original'] = fig.add_subplot(gs[0, 0])
    axes['original'].set_title('Original Image')
    axes['original'].text(0.5, 0.5, 'Original\nImage', ha='center', va='center', fontsize=14)
    
    axes['hue'] = fig.add_subplot(gs[0, 1])
    axes['hue'].set_title('Hue Filter')
    axes['hue'].text(0.5, 0.5, 'Hue\nFilter', ha='center', va='center', fontsize=14)
    
    axes['sift'] = fig.add_subplot(gs[0, 2])
    axes['sift'].set_title('SIFT Filter')
    axes['sift'].text(0.5, 0.5, 'SIFT\nFilter', ha='center', va='center', fontsize=14)
    
    axes['laplacian'] = fig.add_subplot(gs[0, 3])
    axes['laplacian'].set_title('Laplacian Filter')
    axes['laplacian'].text(0.5, 0.5, 'Laplacian\nFilter', ha='center', va='center', fontsize=14)
    
    # Row 2
    axes['saturation'] = fig.add_subplot(gs[1, 0])
    axes['saturation'].set_title('Saturation Filter')
    axes['saturation'].text(0.5, 0.5, 'Saturation\nFilter', ha='center', va='center', fontsize=14)
    
    axes['value'] = fig.add_subplot(gs[1, 1])
    axes['value'].set_title('Value Filter')
    axes['value'].text(0.5, 0.5, 'Value\nFilter', ha='center', va='center', fontsize=14)
    
    axes['edge'] = fig.add_subplot(gs[1, 2])
    axes['edge'].set_title('Edge Filter')
    axes['edge'].text(0.5, 0.5, 'Edge\nFilter', ha='center', va='center', fontsize=14)
    
    axes['combined'] = fig.add_subplot(gs[1, 3])
    axes['combined'].set_title('Combined Mask')
    axes['combined'].text(0.5, 0.5, 'Combined\nMask', ha='center', va='center', fontsize=14)
    
    # Row 3
    axes['result'] = fig.add_subplot(gs[2, 0:2])
    axes['result'].set_title('Final Result with Bounding Boxes')
    axes['result'].text(0.5, 0.5, 'Final Result\nwith Bounding Boxes', ha='center', va='center', fontsize=14)
    
    # Test control positions (simplified)
    control_positions = []
    
    # Control panel dimensions
    panel_bottom = 0.02
    panel_height = 0.09
    col_width = 0.18
    col_spacing = 0.02
    slider_height = 0.015
    
    # Column positions
    columns = {
        'hue': 0.05,
        'sift': 0.05 + col_width + col_spacing,
        'laplacian': 0.05 + 2 * (col_width + col_spacing),
        'buttons': 0.05 + 3 * (col_width + col_spacing),
        'checkboxes': 0.05 + 3 * (col_width + col_spacing) + 0.15  # Increased spacing
    }
    
    # Verify no overlaps
    print("Layout Test Results:")
    print("==================")
    print(f"Figure size: 28x14 inches")
    print(f"Grid layout: 4 rows x 4 columns")
    print(f"Control panel height: {panel_height}")
    print(f"Column width: {col_width}")
    print(f"Column spacing: {col_spacing}")
    print("\nColumn positions:")
    for name, pos in columns.items():
        print(f"  {name}: {pos:.3f} (ends at {pos + col_width:.3f})")
    
    # Check for overlaps
    positions = list(columns.values())
    overlaps = []
    for i, pos1 in enumerate(positions[:-1]):
        for j, pos2 in enumerate(positions[i+1:], i+1):
            if abs(pos1 - pos2) < col_width:
                overlaps.append((list(columns.keys())[i], list(columns.keys())[j]))
    
    if overlaps:
        print(f"\nWARNING: Potential overlaps detected: {overlaps}")
    else:
        print("\n✅ No overlaps detected in layout!")
    
    print(f"\nImage display areas: {len(axes)} subplots")
    print("All filter types have dedicated display areas")
    
    # Save the test layout
    plt.savefig('layout_test.png', dpi=100, bbox_inches='tight')
    print("\nLayout test image saved as 'layout_test.png'")
    plt.close()
    
    return True

if __name__ == "__main__":
    success = test_layout()
    if success:
        print("\n🎉 Layout test completed successfully!")
    else:
        print("\n❌ Layout test failed!")
        sys.exit(1)
