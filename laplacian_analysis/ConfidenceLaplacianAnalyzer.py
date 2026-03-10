#!/usr/bin/env python3

"""
Analyzer to investigate the relationship between model confidence and Laplacian threshold
for coral spawn detections.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import yaml
import time
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Optional
import torch
from ultralytics import YOLO
from ultralytics.engine.results import Results, Boxes

# Import existing filters
import sys
sys.path.append('../annotation')
from annotation.FilterLaplacian import FilterLaplacian
from annotation.FilterCommon import FilterCommon


class ConfidenceLaplacianAnalyzer:
    def __init__(self, 
                 model_weights_path: str,
                 config_path: str,
                 confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.5):
        """
        Initialize the analyzer with model and configuration.
        
        Args:
            model_weights_path: Path to YOLO model weights (.pt file)
            config_path: Path to configuration YAML file
            confidence_threshold: Minimum confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Load YOLO model
        try:
            print(f"Loading YOLO model from: {model_weights_path}")
            self.model = YOLO(model_weights_path)
            self.model.to(self.device)
            print(f"✓ Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
        
        # Load filter configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Initialize Laplacian filter
        self.laplacian_filter = FilterLaplacian(self.config)
        
        # Define range of thresholds to test
        self.laplacian_thresholds = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
        
    def predict_image(self, image_path: str) -> Tuple[List[Dict], np.ndarray]:
        """
        Run YOLO prediction on an image and extract detection data.
        
        Returns:
            List of detection dictionaries and the loaded image
        """
        # Load image
        image = cv.imread(image_path)
        if image is None:
            return [], None
            
        try:
            # Run prediction
            results = self.model(image_path, 
                               conf=self.confidence_threshold,
                               iou=self.iou_threshold,
                               verbose=False)
            
            detections = []
            
            if len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                
                # Extract bounding boxes and confidences directly from tensor
                if len(boxes.xyxy) > 0:
                    # Get bounding boxes in xyxy format
                    bboxes = boxes.xyxy.cpu().numpy()  # Shape: (N, 4)
                    confidences = boxes.conf.cpu().numpy()  # Shape: (N,)
                    classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.zeros(len(bboxes))
                    
                    for i, (bbox, conf, cls) in enumerate(zip(bboxes, confidences, classes)):
                        detection = {
                            'bbox': bbox,  # [x1, y1, x2, y2]
                            'confidence': float(conf),
                            'class': int(cls)
                        }
                        detections.append(detection)
            
            return detections, image
            
        except Exception as e:
            print(f"Error predicting {image_path}: {e}")
            return [], image
    
    def calculate_laplacian_stats(self, image: np.ndarray, bbox: np.ndarray, threshold: float) -> Dict:
        """
        Calculate Laplacian statistics for a detection region at a given threshold.
        
        Args:
            image: Input image (BGR)
            bbox: Bounding box [x1, y1, x2, y2]
            threshold: Laplacian threshold value
            
        Returns:
            Dictionary of Laplacian statistics
        """
        # Convert to grayscale
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        
        # Apply denoising
        image_denoised = cv.fastNlMeansDenoising(
            image_gray, 
            templateWindowSize=self.config['laplacian']['denoise_template_window_size'],
            searchWindowSize=self.config['laplacian']['denoise_search_window_size'],
            h=self.config['laplacian']['denoise_strength']
        )
        
        # Calculate Laplacian response
        laplacian = cv.Laplacian(
            image_denoised, 
            cv.CV_16S, 
            ksize=self.config['laplacian']['laplacian_kernel_size']
        )
        abs_laplacian = cv.convertScaleAbs(laplacian)
        
        # Extract region of interest
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        roi_laplacian = abs_laplacian[y1:y2, x1:x2]
        
        if roi_laplacian.size == 0:
            return {
                'laplacian_threshold': threshold,
                'mean_laplacian': 0,
                'max_laplacian': 0,
                'min_laplacian': 0,
                'std_laplacian': 0,
                'median_laplacian': 0,
                'percentile_75': 0,
                'percentile_90': 0,
                'percentile_95': 0,
                'above_threshold_ratio': 0
            }
        
        # Calculate comprehensive statistics
        stats = {
            'laplacian_threshold': threshold,
            'mean_laplacian': np.mean(roi_laplacian),
            'max_laplacian': np.max(roi_laplacian),
            'min_laplacian': np.min(roi_laplacian),
            'std_laplacian': np.std(roi_laplacian),
            'median_laplacian': np.median(roi_laplacian),
            'percentile_75': np.percentile(roi_laplacian, 75),
            'percentile_90': np.percentile(roi_laplacian, 90),
            'percentile_95': np.percentile(roi_laplacian, 95),
            'above_threshold_ratio': np.sum(roi_laplacian > threshold) / roi_laplacian.size
        }
        
        return stats
    
    def process_image_with_thresholds(self, image_path: str) -> List[Dict]:
        """
        Process an image with multiple Laplacian thresholds.
        
        Returns:
            List of detection data dictionaries
        """
        # Get model predictions
        detections, image = self.predict_image(image_path)
        
        if image is None or len(detections) == 0:
            return []
        
        all_detection_data = []
        
        for detection in detections:
            bbox = detection['bbox']
            confidence = detection['confidence']
            class_id = detection['class']
            
            # Test each Laplacian threshold
            for threshold in self.laplacian_thresholds:
                # Calculate Laplacian stats for this threshold
                laplacian_stats = self.calculate_laplacian_stats(image, bbox, threshold)
                
                # Create comprehensive detection record
                detection_dict = {
                    'image_path': image_path,
                    'image_name': os.path.basename(image_path),
                    'confidence': confidence,
                    'class': class_id,
                    'bbox_x1': bbox[0],
                    'bbox_y1': bbox[1],
                    'bbox_x2': bbox[2],
                    'bbox_y2': bbox[3],
                    'bbox_width': bbox[2] - bbox[0],
                    'bbox_height': bbox[3] - bbox[1],
                    'bbox_area': (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]),
                    **laplacian_stats
                }
                
                all_detection_data.append(detection_dict)
        
        return all_detection_data
    
    def analyze_directory(self, 
                         image_dir: str, 
                         output_dir: str, 
                         image_pattern: str = '*.jpg',
                         max_images: int = 1000,
                         save_visualizations: bool = True) -> pd.DataFrame:
        """
        Analyze a directory of images for confidence-Laplacian correlations.
        
        Args:
            image_dir: Directory containing images
            output_dir: Directory to save results
            image_pattern: Glob pattern for image files
            max_images: Maximum number of images to process
            save_visualizations: Whether to create and save plots
            
        Returns:
            DataFrame with all detection and analysis data
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image list
        image_list = sorted(glob.glob(os.path.join(image_dir, image_pattern)))[:max_images]
        
        if len(image_list) == 0:
            print(f"No images found in {image_dir} with pattern {image_pattern}")
            return pd.DataFrame()
        
        print(f"Processing {len(image_list)} images with {len(self.laplacian_thresholds)} thresholds each...")
        print(f"Laplacian thresholds: {self.laplacian_thresholds}")
        
        all_data = []
        start_time = time.time()
        
        for i, image_path in enumerate(image_list):
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start_time
                eta = elapsed / (i + 1) * len(image_list) - elapsed
                print(f"Processing {i+1}/{len(image_list)}: {os.path.basename(image_path)} "
                      f"(ETA: {eta/60:.1f}m)")
            
            detection_data = self.process_image_with_thresholds(image_path)
            all_data.extend(detection_data)
            
            # Save intermediate results every 100 images
            if (i + 1) % 100 == 0 and self.config.get('analysis', {}).get('save_intermediate_results', False):
                temp_df = pd.DataFrame(all_data)
                temp_df.to_csv(os.path.join(output_dir, f'temp_results_{i+1}.csv'), index=False)
        
        # Create DataFrame from all data
        df = pd.DataFrame(all_data)
        
        if len(df) == 0:
            print("No detections found in any images!")
            return df
        
        print(f"\nAnalysis completed in {(time.time() - start_time)/60:.1f} minutes")
        print(f"Total detections: {len(df)}")
        print(f"Unique images: {df['image_name'].nunique()}")
        print(f"Average detections per image: {len(df) / df['image_name'].nunique():.1f}")
        
        # Save raw results
        csv_path = os.path.join(output_dir, 'confidence_laplacian_analysis.csv')
        df.to_csv(csv_path, index=False)
        print(f"Raw data saved to: {csv_path}")
        
        # Save pickle for fast loading
        pickle_path = os.path.join(output_dir, 'confidence_laplacian_analysis.pkl')
        df.to_pickle(pickle_path)
        print(f"Pickle data saved to: {pickle_path}")
        
        # Calculate correlations
        self._calculate_and_save_correlations(df, output_dir)
        
        # Create visualizations
        if save_visualizations:
            self._create_visualizations(df, output_dir)
        
        return df
    
    def _calculate_and_save_correlations(self, df: pd.DataFrame, output_dir: str):
        """Calculate and save correlation analysis."""
        print("\nCalculating correlations...")
        
        # Correlation with confidence
        laplacian_cols = [col for col in df.columns if 'laplacian' in col and col != 'laplacian_threshold']
        
        correlations = {}
        for col in laplacian_cols:
            pearson_corr, pearson_p = pearsonr(df['confidence'], df[col])
            spearman_corr, spearman_p = spearmanr(df['confidence'], df[col])
            
            correlations[col] = {
                'pearson_correlation': pearson_corr,
                'pearson_p_value': pearson_p,
                'spearman_correlation': spearman_corr,
                'spearman_p_value': spearman_p
            }
        
        # Special analysis for threshold relationship
        threshold_corr = df['confidence'].corr(df['laplacian_threshold'])
        
        # Save summary
        summary = {
            'analysis_info': {
                'total_detections': len(df),
                'unique_images': df['image_name'].nunique(),
                'laplacian_thresholds_tested': self.laplacian_thresholds,
                'confidence_range': [float(df['confidence'].min()), float(df['confidence'].max())],
                'confidence_mean': float(df['confidence'].mean()),
                'confidence_std': float(df['confidence'].std())
            },
            'threshold_correlation': {
                'correlation_with_threshold': float(threshold_corr),
                'interpretation': self._interpret_correlation(threshold_corr)
            },
            'laplacian_metric_correlations': correlations,
            'top_correlations': self._get_top_correlations(correlations)
        }
        
        # Save summary
        summary_path = os.path.join(output_dir, 'analysis_summary.yaml')
        with open(summary_path, 'w') as f:
            yaml.dump(summary, f, default_flow_style=False)
        print(f"Analysis summary saved to: {summary_path}")
        
        # Print key findings
        print(f"\nKey Findings:")
        print(f"  Confidence vs Laplacian Threshold correlation: {threshold_corr:.3f}")
        print(f"  Interpretation: {self._interpret_correlation(threshold_corr)}")
        
        top_corrs = self._get_top_correlations(correlations)
        print(f"\nTop correlations with confidence:")
        for metric, corr_info in top_corrs.items():
            print(f"  {metric}: {corr_info['pearson_correlation']:.3f} (p={corr_info['pearson_p_value']:.3e})")
    
    def _interpret_correlation(self, correlation: float) -> str:
        """Interpret correlation strength."""
        abs_corr = abs(correlation)
        direction = "positive" if correlation > 0 else "negative"
        
        if abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate"
        elif abs_corr >= 0.3:
            strength = "weak"
        else:
            strength = "very weak"
        
        return f"{strength} {direction} correlation"
    
    def _get_top_correlations(self, correlations: Dict, top_n: int = 3) -> Dict:
        """Get top N correlations by absolute value."""
        sorted_corrs = sorted(correlations.items(), 
                            key=lambda x: abs(x[1]['pearson_correlation']), 
                            reverse=True)
        return dict(sorted_corrs[:top_n])
    
    def _create_visualizations(self, df: pd.DataFrame, output_dir: str):
        """Create and save visualization plots."""
        print("\nCreating visualizations...")
        
        # 1. Correlation heatmap
        plt.figure(figsize=(12, 8))
        
        # Select numeric columns for correlation
        numeric_cols = ['confidence', 'laplacian_threshold', 'mean_laplacian', 
                       'max_laplacian', 'std_laplacian', 'above_threshold_ratio',
                       'bbox_area', 'bbox_width', 'bbox_height']
        
        # Filter to only existing columns
        available_cols = [col for col in numeric_cols if col in df.columns]
        corr_matrix = df[available_cols].corr()
        
        import seaborn as sns
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title('Correlation Matrix: Confidence vs Laplacian Metrics')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confidence distribution by threshold
        plt.figure(figsize=(15, 10))
        
        # Scatter plot
        plt.subplot(2, 3, 1)
        plt.scatter(df['laplacian_threshold'], df['confidence'], alpha=0.5)
        plt.xlabel('Laplacian Threshold')
        plt.ylabel('Model Confidence')
        plt.title('Confidence vs Laplacian Threshold')
        plt.grid(True, alpha=0.3)
        
        # Box plot
        plt.subplot(2, 3, 2)
        df.boxplot(column='confidence', by='laplacian_threshold', ax=plt.gca())
        plt.title('Confidence Distribution by Threshold')
        plt.suptitle('')  # Remove auto-title
        
        # Mean confidence by threshold
        plt.subplot(2, 3, 3)
        mean_conf = df.groupby('laplacian_threshold')['confidence'].mean()
        plt.plot(mean_conf.index, mean_conf.values, 'o-')
        plt.xlabel('Laplacian Threshold')
        plt.ylabel('Mean Confidence')
        plt.title('Mean Confidence by Threshold')
        plt.grid(True, alpha=0.3)
        
        # Confidence vs mean_laplacian
        plt.subplot(2, 3, 4)
        plt.scatter(df['mean_laplacian'], df['confidence'], alpha=0.5)
        plt.xlabel('Mean Laplacian Response')
        plt.ylabel('Model Confidence')
        plt.title('Confidence vs Mean Laplacian')
        plt.grid(True, alpha=0.3)
        
        # Confidence vs above_threshold_ratio
        plt.subplot(2, 3, 5)
        plt.scatter(df['above_threshold_ratio'], df['confidence'], alpha=0.5)
        plt.xlabel('Above Threshold Ratio')
        plt.ylabel('Model Confidence')
        plt.title('Confidence vs Above Threshold Ratio')
        plt.grid(True, alpha=0.3)
        
        # Confidence histogram
        plt.subplot(2, 3, 6)
        plt.hist(df['confidence'], bins=30, alpha=0.7, edgecolor='black')
        plt.xlabel('Model Confidence')
        plt.ylabel('Frequency')
        plt.title('Confidence Distribution')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confidence_analysis_plots.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    # Example usage
    model_weights_path = "/home/java/hpc-home/cslics_detection/cslics_combined_202508185/weights/best.pt"
    config_path = "config_confidence_laplacian_analysis.yaml"
    image_dir = "/home/java/hpc-home/Data/cslics/2023_2024_combined_dataset/cslics_2024_species_data/cslics_2024_pdae_438d_1_1000_split/test/images"
    output_dir = "/home/java/hpc-home/Data/cslics/confidence_laplacian_results"
    
    analyzer = ConfidenceLaplacianAnalyzer(
        model_weights_path=model_weights_path,
        config_path=config_path,
        confidence_threshold=0.25
    )
    
    results_df = analyzer.analyze_directory(
        image_dir=image_dir,
        output_dir=output_dir,
        max_images=100
    )
    
    print(f"Analysis complete! Results saved to {output_dir}")
