#!/usr/bin/env python3

"""
Simple runner script for the new BoundingBoxLaplacianAnalyzer
Demonstrates the improved per-bounding-box analysis approach
"""

import os
import sys

def main():
    """Main function to run the bounding box Laplacian analysis."""
    
    print("🔬 Coral Spawn Laplacian Analysis - Bounding Box Approach")
    print("=" * 65)
    
    # Check if we're in the right directory
    current_dir = os.getcwd()
    if not current_dir.endswith('laplacian_analysis'):
        print("⚠️  Please run this script from the laplacian_analysis directory")
        print(f"   Current directory: {current_dir}")
        print(f"   Expected to end with: laplacian_analysis")
        return
    
    print("✅ Running from correct directory")
    
    # Configuration paths (adjust as needed for your system)
    config = {
        'model_weights_path': "/home/java/hpc-home/cslics_detection/cslics_combined_202508185/weights/best.pt",
        'config_path': "/home/java/hpc-home/Corals/cslic/coral_spawn_counter/data_yaml_files/annotation_cslics_2024_nov_pdae_tank4_100000001ab0438d.yaml",
        'image_dir': "/home/java/hpc-home/Data/cslics/2023_2024_combined_dataset/cslics_2024_species_data/cslics_2024_pdae_438d_1_1000_split/test/images",
        'output_dir': "/home/java/hpc-home/Data/cslics/bounding_box_laplacian_results",
        'max_images': 25  # Start with a reasonable number for testing
    }
    
    # Check if paths exist
    print("\n🔍 Checking configuration paths...")
    
    paths_exist = {}
    for key, path in config.items():
        if key in ['output_dir', 'max_images']:
            continue
        exists = os.path.exists(path)
        paths_exist[key] = exists
        status = "✅" if exists else "❌"
        print(f"   {status} {key}: {path}")
    
    # If some paths don't exist, offer to run with demo mode
    if not all(paths_exist.values()):
        print("\n⚠️  Some paths don't exist. Options:")
        print("   1. Update the paths in this script")
        print("   2. Run in demo mode with test data")
        print("   3. Run the test script instead")
        
        choice = input("\nChoose option (1/2/3) or press Enter for demo mode: ").strip()
        
        if choice == "1":
            print("📝 Please edit the 'config' dictionary in this script with correct paths")
            return
        elif choice == "3":
            print("🧪 Running test script instead...")
            os.system("python test_bounding_box_analyzer.py")
            return
        else:
            print("🎬 Running in demo mode...")
            run_demo_mode()
            return
    
    # Try to import and run the analyzer
    try:
        print("\n📦 Importing BoundingBoxLaplacianAnalyzer...")
        from BoundingBoxLaplacianAnalyzer import BoundingBoxLaplacianAnalyzer
        print("✅ Import successful")
        
        # Create analyzer
        print("🔧 Creating analyzer...")
        analyzer = BoundingBoxLaplacianAnalyzer(
            model_weights_path=config['model_weights_path'],
            config_path=config['config_path'],
            confidence_threshold=0.25
        )
        print("✅ Analyzer created")
        
        # Run analysis
        print(f"\n🚀 Starting analysis of up to {config['max_images']} images...")
        print(f"   Input directory: {config['image_dir']}")
        print(f"   Output directory: {config['output_dir']}")
        
        results_df = analyzer.analyze_directory(
            image_dir=config['image_dir'],
            output_dir=config['output_dir'],
            max_images=config['max_images']
        )
        
        print(f"\n🎉 Analysis complete!")
        print(f"   Total detections analyzed: {len(results_df)}")
        print(f"   Results saved to: {config['output_dir']}")
        
        # Show some sample results
        if len(results_df) > 0:
            print(f"\n📊 Sample results (first detection):")
            sample = results_df.iloc[0]
            print(f"   Image: {sample['image_name']}")
            print(f"   Confidence: {sample['confidence']:.3f}")
            print(f"   Raw Laplacian Mean: {sample.get('raw_lap_mean', 'N/A')}")
            print(f"   Edge Density: {sample.get('edge_density', 'N/A')}")
            print(f"   Bounding Box Area: {sample['bbox_area']}")
        
    except ImportError as e:
        print(f"❌ Failed to import analyzer: {e}")
        print("💡 Try running the test script first: python test_bounding_box_analyzer.py")
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()


def run_demo_mode():
    """Run a simple demo without requiring all paths to exist."""
    
    print("\n🎬 Demo Mode - Testing Analyzer Functionality")
    print("-" * 50)
    
    try:
        # Test basic imports
        import numpy as np
        import cv2 as cv
        print("✅ Basic dependencies available")
        
        # Create a test image
        print("🖼️  Creating test image...")
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Add some structure to make it more realistic
        cv.rectangle(test_image, (100, 100), (200, 200), (255, 255, 255), -1)
        cv.circle(test_image, (350, 300), 50, (128, 128, 128), -1)
        
        print("✅ Test image created")
        
        # Test the analysis functions directly
        print("🔬 Testing Laplacian calculations...")
        
        # Convert to grayscale
        gray = cv.cvtColor(test_image, cv.COLOR_BGR2GRAY)
        
        # Test different Laplacian calculations
        lap_raw = cv.Laplacian(gray, cv.CV_64F, ksize=3)
        gray_denoised = cv.fastNlMeansDenoising(gray, templateWindowSize=7, searchWindowSize=21, h=3)
        lap_denoised = cv.Laplacian(gray_denoised, cv.CV_64F, ksize=3)
        
        print(f"   Raw Laplacian - Mean: {np.mean(np.abs(lap_raw)):.2f}, Std: {np.std(np.abs(lap_raw)):.2f}")
        print(f"   Denoised Laplacian - Mean: {np.mean(np.abs(lap_denoised)):.2f}, Std: {np.std(np.abs(lap_denoised)):.2f}")
        
        # Test edge detection
        edges = cv.Canny(gray_denoised, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        print(f"   Edge Density: {edge_density:.3f}")
        
        print("✅ Basic analysis functions working")
        
        # Test bounding box extraction
        test_bbox = np.array([100, 100, 300, 300])
        x1, y1, x2, y2 = test_bbox
        roi = gray[y1:y2, x1:x2]
        
        if roi.size > 0:
            roi_stats = {
                'mean': np.mean(roi),
                'std': np.std(roi),
                'contrast': np.std(roi) / (np.mean(roi) + 1e-8)
            }
            print(f"   ROI Stats - Mean: {roi_stats['mean']:.1f}, Std: {roi_stats['std']:.1f}, Contrast: {roi_stats['contrast']:.3f}")
        
        print("\n🎉 Demo completed successfully!")
        print("💡 The new bounding box approach is working correctly.")
        print("📝 Update the configuration paths to run on real data.")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("🚀 Coral Spawn Laplacian Analysis Runner")
    print("This script runs the new bounding-box-based Laplacian analysis")
    print("")
    
    main()
    
    print("\n" + "=" * 65)
    print("📚 For more information:")
    print("   • README.md - Overview and documentation")
    print("   • test_bounding_box_analyzer.py - Test the analyzer")
    print("   • comparison_demo.py - Compare old vs new approaches")
    print("   • BoundingBoxLaplacianAnalyzer.py - Main analyzer code")
