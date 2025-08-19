#!/usr/bin/env python3

"""
Performance test for the improved interactive visualizer
"""

import os
import sys
import time
import numpy as np
import cv2 as cv

def create_test_image(width=1920, height=1080):
    """Create a test image with some colored circles."""
    print(f"Creating test image: {width}x{height}")
    
    # Create a test image
    image = np.random.randint(0, 128, (height, width, 3), dtype=np.uint8)
    
    # Add some colored circles that should be detected
    num_circles = 10
    for i in range(num_circles):
        center_x = np.random.randint(50, width-50)
        center_y = np.random.randint(50, height-50)
        radius = np.random.randint(10, 30)
        
        # Create red/orange circles (hue 0-40)
        color = (0, 0, 255) if i % 2 == 0 else (0, 128, 255)  # Red or orange in BGR
        cv.circle(image, (center_x, center_y), radius, color, -1)
    
    return image

def test_image_scaling():
    """Test the image scaling functionality."""
    print("Testing image scaling...")
    
    # Test different image sizes
    sizes = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
    max_size = 800
    
    for width, height in sizes:
        max_dim = max(width, height)
        if max_dim > max_size:
            scale_factor = max_size / max_dim
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            print(f"{width}x{height} -> {new_width}x{new_height} (scale: {scale_factor:.2f})")
        else:
            print(f"{width}x{height} -> no scaling needed")

def test_simple_hue_filter():
    """Test a simple hue filter implementation."""
    print("\nTesting simple hue filter...")
    
    # Create test image
    image = create_test_image(800, 600)
    
    start_time = time.time()
    
    # Convert to HSV
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    
    # Create hue mask (red/orange: 0-40)
    hue_min, hue_max = 0, 40
    mask = cv.inRange(hsv[:, :, 0], hue_min, hue_max)
    
    # Apply morphological operations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11, 11))
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    
    # Find contours and filter by area
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    
    for contour in contours:
        area = cv.contourArea(contour)
        if 500 <= area <= 5000:  # Filter by area
            filtered_contours.append(contour)
    
    processing_time = time.time() - start_time
    print(f"Processed {image.shape[1]}x{image.shape[0]} image in {processing_time:.3f}s")
    print(f"Found {len(filtered_contours)} objects")
    
    return processing_time

def main():
    """Run performance tests."""
    print("🧪 Interactive Visualizer Performance Tests")
    print("=" * 50)
    
    # Test 1: Image scaling
    test_image_scaling()
    
    # Test 2: Processing speed
    print(f"\nTesting processing speeds...")
    times = []
    for i in range(3):
        processing_time = test_simple_hue_filter()
        times.append(processing_time)
    
    avg_time = np.mean(times)
    print(f"\nAverage processing time: {avg_time:.3f}s")
    print(f"Expected frame rate: {1/avg_time:.1f} FPS")
    
    print("\n" + "=" * 50)
    print("✅ Performance tests completed!")
    print("\nExpected improvements in the interactive visualizer:")
    print("- Automatic image scaling for faster processing")
    print("- Debounced updates (100ms delay)")
    print("- Progress feedback during processing")
    print("- Efficient canvas updates (draw_idle)")
    print("- Status messages and performance stats")

if __name__ == "__main__":
    main()
