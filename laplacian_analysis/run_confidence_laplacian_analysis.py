#!/usr/bin/env python3

"""
Example script to run the Confidence-Laplacian correlation analysis.
This script demonstrates how to use the ConfidenceLaplacianAnalyzer to investigate
the relationship between model confidence and Laplacian threshold values.
"""

import os
import sys
from ConfidenceLaplacianAnalyzer import ConfidenceLaplacianAnalyzer

def main():
    """
    Run the confidence-Laplacian correlation analysis.
    """
    
    # Configuration - Update these paths for your specific setup
    print("Setting up Confidence-Laplacian Analysis...")
    
    # Path to your trained YOLO model weights
    # This should be a .pt file from a trained YOLOv8 model
    model_weights_path = "/home/java/hpc-home/Corals/cslic/coral_spawn_counter/weights/your_model.pt"
    
    # Path to the configuration file (same as used in AnnotationPipeline)
    config_path = "../data_yaml_files/annotation_cslics_2024_oct_amag_tank2_100000009c23b5af.yaml"
    
    # Directory containing test images
    image_dir = "/path/to/your/test/images"
    
    # Output directory for analysis results
    output_dir = "./confidence_laplacian_analysis_results"
    
    # Analysis parameters
    confidence_threshold = 0.25  # Lower threshold to capture more detections
    iou_threshold = 0.5
    max_images = 500  # Limit for initial analysis
    
    print(f"Model weights: {model_weights_path}")
    print(f"Config file: {config_path}")
    print(f"Image directory: {image_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {confidence_threshold}")
    print(f"IoU threshold: {iou_threshold}")
    print(f"Max images to process: {max_images}")
    
    # Check if paths exist
    if not os.path.exists(model_weights_path):
        print(f"Error: Model weights file not found: {model_weights_path}")
        print("Please update the model_weights_path in this script.")
        return
    
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print("Please update the config_path in this script.")
        return
    
    if not os.path.exists(image_dir):
        print(f"Error: Image directory not found: {image_dir}")
        print("Please update the image_dir in this script.")
        return
    
    try:
        # Create the analyzer
        print("\nInitializing analyzer...")
        analyzer = ConfidenceLaplacianAnalyzer(
            model_weights_path=model_weights_path,
            config_path=config_path,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        # Run the analysis
        print("\nStarting analysis...")
        results_df = analyzer.analyze_directory(
            image_dir=image_dir,
            output_dir=output_dir,
            image_pattern='*.jpg',
            max_images=max_images,
            save_visualizations=True
        )
        
        if len(results_df) > 0:
            print(f"\nAnalysis completed successfully!")
            print(f"Total detections analyzed: {len(results_df)}")
            print(f"Results saved to: {output_dir}")
            print(f"\nKey files generated:")
            print(f"  - confidence_laplacian_analysis.csv: Raw data")
            print(f"  - confidence_laplacian_analysis.pkl: Pickle format for fast loading")
            print(f"  - analysis_summary.yaml: Summary statistics and correlations")
            print(f"  - *.png: Visualization plots")
            
            # Print some quick statistics
            print(f"\nQuick Statistics:")
            print(f"  Mean confidence: {results_df['confidence'].mean():.3f}")
            print(f"  Std confidence: {results_df['confidence'].std():.3f}")
            print(f"  Confidence range: [{results_df['confidence'].min():.3f}, {results_df['confidence'].max():.3f}]")
            
            # Print top correlations if available
            laplacian_cols = [col for col in results_df.columns if 'laplacian' in col]
            if laplacian_cols:
                print(f"\nTop correlations with confidence:")
                for col in laplacian_cols[:5]:  # Show top 5
                    corr = results_df['confidence'].corr(results_df[col])
                    print(f"  {col}: {corr:.3f}")
        else:
            print("No detections found. Check your model weights and image directory.")
    
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        print("Please check your configuration and try again.")
        return


def run_example_with_sample_data():
    """
    Example function showing how to set up the analysis for the coral spawn dataset.
    Modify the paths below to match your actual data locations.
    """
    
    print("Example configuration for coral spawn analysis:")
    print()
    
    # Example paths based on the existing codebase structure
    example_configs = {
        "October 2024 - Tank 2": {
            "model_weights_path": "/home/java/hpc-home/Corals/cslic/coral_spawn_counter/weights/best_model_oct2024.pt",
            "config_path": "../data_yaml_files/annotation_cslics_2024_oct_amag_tank2_100000009c23b5af.yaml",
            "image_dir": "/path/to/cslics_2024_october_subsurface_dataset/100000009c23b5af/images",
            "output_dir": "./analysis_oct2024_tank2"
        },
        "November 2024 - Tank 3": {
            "model_weights_path": "/home/java/hpc-home/Corals/cslic/coral_spawn_counter/weights/best_model_nov2024.pt", 
            "config_path": "../data_yaml_files/annotation_cslics_2024_nov_amil_tank3_10000000f620da42.yaml",
            "image_dir": "/path/to/cslics_2024_november_subsurface_dataset/10000000f620da42/images",
            "output_dir": "./analysis_nov2024_tank3"
        }
    }
    
    print("Example configurations:")
    for name, config in example_configs.items():
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\nTo use one of these configurations:")
    print("1. Update the paths to match your actual file locations")
    print("2. Ensure you have a trained YOLO model (.pt file)")
    print("3. Run the main() function with your updated paths")


if __name__ == "__main__":
    print("Confidence-Laplacian Correlation Analysis Tool")
    print("=" * 50)
    
    # Check command line arguments
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        run_example_with_sample_data()
    else:
        main()
