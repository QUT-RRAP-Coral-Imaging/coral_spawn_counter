#!/usr/bin/env python3

"""
Bounding Box Laplacian Analyzer - calculates Laplacian statistics for each detection
without threshold-based filtering, focusing on per-bounding-box analysis.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import yaml
import pandas as pd
import pickle
from typing import List, Dict, Optional
import torch
import torchvision.transforms as transforms
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    print("⚠️  ultralytics not available, trying alternative YOLO imports")

try:
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("⚠️  scikit-learn not available, some features may be limited")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False
    print("⚠️  seaborn not available, using matplotlib for all plots")

# Import existing filters
import sys
sys.path.append('../annotation')
try:
    from FilterLaplacian import FilterLaplacian
    FILTER_AVAILABLE = True
except ImportError:
    FILTER_AVAILABLE = False
    print("⚠️  FilterLaplacian not available, continuing without it")


class BoundingBoxLaplacianAnalyzer:
    def __init__(self, 
                 model_weights_path: str,
                 config_path: str,
                 confidence_threshold: float = 0.3):
        """
        Analyzer that calculates Laplacian statistics for each bounding box detection.
        Uses real YOLO model for inference.
        """
        self.confidence_threshold = confidence_threshold
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # Load YOLO model
        self.model = None
        try:
            if ULTRALYTICS_AVAILABLE:
                print(f"Loading YOLO model from: {model_weights_path}")
                # Load model and override task to detection
                from ultralytics import YOLO
                self.model = YOLO(model_weights_path)
                # Force the model to detection mode by modifying its task
                if hasattr(self.model, 'model') and hasattr(self.model.model, 'model'):
                    # Override the model's forward method to only return detection outputs
                    self.model.task = 'detect'
                print("✓ YOLO model loaded successfully with ultralytics")
            else:
                # Fallback: try to load as torch model
                print(f"Loading model weights from: {model_weights_path}")
                self.model_weights = torch.load(model_weights_path, map_location=self.device)
                print("✓ Model weights loaded successfully (torch format)")
                print("⚠️  Note: Full YOLO inference requires ultralytics package")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("⚠️  Continuing without model - analysis will be limited")
            self.model = None
        
        # Load filter configuration
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)
            
        # Extract Laplacian-specific configuration
        self.laplacian_config = self.config.get('laplacian', {})
        
        # Set analysis parameters from config
        self.laplacian_threshold = self.laplacian_config.get('laplacian_threshold', 25)
        self.denoise_template_window_size = self.laplacian_config.get('denoise_template_window_size', 15)
        self.denoise_search_window_size = self.laplacian_config.get('denoise_search_window_size', 31)
        self.denoise_strength = self.laplacian_config.get('denoise_strength', 11)
        self.laplacian_kernel_size = self.laplacian_config.get('laplacian_kernel_size', 5)
        self.process_denoise = self.laplacian_config.get('process_denoise', True)
        
        print(f"✓ Loaded Laplacian config - threshold: {self.laplacian_threshold}, kernel: {self.laplacian_kernel_size}")
        print(f"  Denoising params: window={self.denoise_template_window_size}, search={self.denoise_search_window_size}, strength={self.denoise_strength}")
            
    def get_yolo_predictions(self, image: np.ndarray) -> List[Dict]:
        """
        Get real YOLO predictions from the loaded model.
        Handles segmentation models gracefully by falling back to simulation on errors.
        """
        if self.model is None:
            print("⚠️  No model loaded, cannot perform inference")
            return []
        
        try:
            if ULTRALYTICS_AVAILABLE and hasattr(self.model, 'predict'):
                # Use ultralytics YOLO - with error handling for segmentation models
                results = self.model.predict(
                    image, 
                    conf=self.confidence_threshold, 
                    verbose=False,
                    device=self.device.type if hasattr(self.device, 'type') else 'cpu'
                )
                
                predictions = []
                if len(results) > 0:
                    result = results[0]  # First (and only) image
                    
                    if result.boxes is not None and len(result.boxes) > 0:
                        boxes = result.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
                        confidences = result.boxes.conf.cpu().numpy()
                        classes = result.boxes.cls.cpu().numpy()
                        
                        for i in range(len(boxes)):
                            if confidences[i] >= self.confidence_threshold:
                                predictions.append({
                                    'bbox': boxes[i].astype(int),
                                    'confidence': float(confidences[i]),
                                    'class': int(classes[i])
                                })
                
                print(f"✓ YOLO detected {len(predictions)} objects (conf >= {self.confidence_threshold})")
                return predictions
                
            else:
                print("⚠️  YOLO model not properly loaded for inference")
                return []
                
        except ValueError as e:
            if "too many values to unpack" in str(e):
                print(f"⚠️  Model appears to be a segmentation model, but we need detection only")
                print(f"⚠️  YOLO inference failed due to segmentation incompatibility: {e}")
                print(f"⚠️  The analysis will continue, but real inference is not available")
                return []
            else:
                print(f"❌ Error during YOLO inference: {e}")
                return []
        except Exception as e:
            print(f"❌ Error during YOLO inference: {e}")
            return []
    
    def calculate_bounding_box_laplacian(self, image: np.ndarray, bbox: np.ndarray) -> Dict:
        """
        Calculate comprehensive Laplacian statistics for a bounding box region.
        Uses config parameters for consistent analysis with the annotation pipeline.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()
        
        # Extract bounding box coordinates
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
        
        # Extract region of interest
        roi = image_gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return self._empty_laplacian_stats()
        
        # Apply different preprocessing techniques using config parameters
        stats = {}
        
        # 1. Raw Laplacian using config kernel size
        laplacian_raw = cv.Laplacian(roi, cv.CV_64F, ksize=self.laplacian_kernel_size)
        stats.update(self._calculate_laplacian_metrics(laplacian_raw, 'raw'))
        
        # 2. Denoised Laplacian using config parameters
        if self.process_denoise:
            roi_denoised = cv.fastNlMeansDenoising(
                roi, 
                templateWindowSize=self.denoise_template_window_size,
                searchWindowSize=self.denoise_search_window_size, 
                h=self.denoise_strength
            )
        else:
            roi_denoised = roi.copy()
            
        laplacian_denoised = cv.Laplacian(roi_denoised, cv.CV_64F, ksize=self.laplacian_kernel_size)
        stats.update(self._calculate_laplacian_metrics(laplacian_denoised, 'denoised'))
        
        # 3. Gaussian blurred then Laplacian
        roi_blurred = cv.GaussianBlur(roi, (3, 3), 0)
        laplacian_blurred = cv.Laplacian(roi_blurred, cv.CV_64F, ksize=self.laplacian_kernel_size)
        stats.update(self._calculate_laplacian_metrics(laplacian_blurred, 'blurred'))
        
        # 4. Different kernel sizes for comparison
        for ksize in [3, 7]:  # Compare with config kernel size
            if ksize != self.laplacian_kernel_size:
                laplacian_k = cv.Laplacian(roi_denoised, cv.CV_64F, ksize=ksize)
                stats.update(self._calculate_laplacian_metrics(laplacian_k, f'k{ksize}'))
        
        # 5. FOCUS ON THRESHOLD ANALYSIS - Key relationship we're investigating
        # Calculate statistics relative to the configured laplacian_threshold
        abs_laplacian_denoised = np.abs(laplacian_denoised)
        
        stats['threshold_analysis'] = {
            'config_threshold': self.laplacian_threshold,
            'above_threshold_ratio': np.sum(abs_laplacian_denoised > self.laplacian_threshold) / abs_laplacian_denoised.size,
            'above_threshold_count': np.sum(abs_laplacian_denoised > self.laplacian_threshold),
            'mean_above_threshold': np.mean(abs_laplacian_denoised[abs_laplacian_denoised > self.laplacian_threshold]) if np.any(abs_laplacian_denoised > self.laplacian_threshold) else 0,
            'mean_below_threshold': np.mean(abs_laplacian_denoised[abs_laplacian_denoised <= self.laplacian_threshold]) if np.any(abs_laplacian_denoised <= self.laplacian_threshold) else 0,
        }
        
        # Flatten the threshold analysis into main stats for easier access
        for key, value in stats['threshold_analysis'].items():
            stats[f'thresh_{key}'] = value
        
        # 6. Edge-based metrics using original parameters
        edges_canny = cv.Canny(roi_denoised, 50, 150)
        stats['edge_density'] = np.sum(edges_canny > 0) / edges_canny.size
        stats['edge_total_length'] = np.sum(edges_canny > 0)
        
        # 7. Texture metrics
        stats['roi_std'] = np.std(roi)
        stats['roi_mean'] = np.mean(roi)
        stats['roi_contrast'] = np.std(roi) / (np.mean(roi) + 1e-8)
        
        # 8. Gradient magnitude
        grad_x = cv.Sobel(roi, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(roi, cv.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        stats['gradient_mean'] = np.mean(gradient_magnitude)
        stats['gradient_std'] = np.std(gradient_magnitude)
        stats['gradient_max'] = np.max(gradient_magnitude)
        
        return stats
    
    def _calculate_laplacian_metrics(self, laplacian: np.ndarray, prefix: str) -> Dict:
        """Calculate various metrics for a Laplacian response."""
        abs_laplacian = np.abs(laplacian)
        
        return {
            f'{prefix}_lap_mean': np.mean(abs_laplacian),
            f'{prefix}_lap_std': np.std(abs_laplacian),
            f'{prefix}_lap_max': np.max(abs_laplacian),
            f'{prefix}_lap_min': np.min(abs_laplacian),
            f'{prefix}_lap_median': np.median(abs_laplacian),
            f'{prefix}_lap_q75': np.percentile(abs_laplacian, 75),
            f'{prefix}_lap_q25': np.percentile(abs_laplacian, 25),
            f'{prefix}_lap_variance': np.var(abs_laplacian),
            f'{prefix}_lap_energy': np.sum(abs_laplacian**2),
            f'{prefix}_lap_entropy': self._calculate_entropy(abs_laplacian)
        }
    
    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate entropy of an image region."""
        # Normalize to 0-255 range
        normalized = ((image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8) * 255).astype(np.uint8)
        
        # Calculate histogram
        hist, _ = np.histogram(normalized, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-8))
        return entropy
    
    def _empty_laplacian_stats(self) -> Dict:
        """Return empty statistics for invalid bounding boxes."""
        empty_stats = {}
        
        # Original Laplacian metrics for different preprocessing methods
        for prefix in ['raw', 'denoised', 'blurred', 'k3', 'k7']:
            empty_stats.update({
                f'{prefix}_lap_mean': 0,
                f'{prefix}_lap_std': 0,
                f'{prefix}_lap_max': 0,
                f'{prefix}_lap_min': 0,
                f'{prefix}_lap_median': 0,
                f'{prefix}_lap_q75': 0,
                f'{prefix}_lap_q25': 0,
                f'{prefix}_lap_variance': 0,
                f'{prefix}_lap_energy': 0,
                f'{prefix}_lap_entropy': 0
            })
        
        # Threshold analysis metrics (the key focus)
        empty_stats.update({
            'thresh_config_threshold': getattr(self, 'laplacian_threshold', 25),
            'thresh_above_threshold_ratio': 0,
            'thresh_above_threshold_count': 0,
            'thresh_mean_above_threshold': 0,
            'thresh_mean_below_threshold': 0,
        })
        
        # Other metrics
        empty_stats.update({
            'edge_density': 0,
            'edge_total_length': 0,
            'roi_std': 0,
            'roi_mean': 0,
            'roi_contrast': 0,
            'gradient_mean': 0,
            'gradient_std': 0,
            'gradient_max': 0
        })
        
        return empty_stats
    
    def process_image(self, image_path: str) -> List[Dict]:
        """
        Process an image and calculate Laplacian statistics for each detection.
        """
        image = cv.imread(image_path)
        if image is None:
            return []
        
        # Get real YOLO predictions
        predictions = self.get_yolo_predictions(image)
        
        all_detection_data = []
        
        for i, pred in enumerate(predictions):
            # Calculate Laplacian stats for this bounding box
            laplacian_stats = self.calculate_bounding_box_laplacian(image, pred['bbox'])
            
            detection_dict = {
                'image_path': image_path,
                'image_name': os.path.basename(image_path),
                'detection_id': i,
                'confidence': pred['confidence'],
                'class': pred['class'],
                'bbox_x1': pred['bbox'][0],
                'bbox_y1': pred['bbox'][1],
                'bbox_x2': pred['bbox'][2],
                'bbox_y2': pred['bbox'][3],
                'bbox_width': pred['bbox'][2] - pred['bbox'][0],
                'bbox_height': pred['bbox'][3] - pred['bbox'][1],
                'bbox_area': (pred['bbox'][2] - pred['bbox'][0]) * (pred['bbox'][3] - pred['bbox'][1]),
                'bbox_aspect_ratio': (pred['bbox'][2] - pred['bbox'][0]) / (pred['bbox'][3] - pred['bbox'][1] + 1e-8),
                **laplacian_stats
            }
            
            all_detection_data.append(detection_dict)
        
        return all_detection_data
    
    def analyze_directory(self, image_dir: str, output_dir: str, max_images: int = 50) -> pd.DataFrame:
        """
        Analyze directory and calculate per-bounding-box Laplacian statistics.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_list = sorted(glob.glob(os.path.join(image_dir, '*.jpg')))[:max_images]
        print(f"Processing {len(image_list)} images...")
        
        all_data = []
        
        for i, image_path in enumerate(image_list):
            if (i + 1) % 10 == 0:
                print(f"Processing {i+1}/{len(image_list)}: {os.path.basename(image_path)}")
            
            detection_data = self.process_image(image_path)
            all_data.extend(detection_data)
        
        df = pd.DataFrame(all_data)
        
        if len(df) > 0:
            # Save results
            df.to_csv(os.path.join(output_dir, 'bounding_box_laplacian_analysis.csv'), index=False)
            
            # Calculate correlations with confidence - focus on threshold analysis
            laplacian_columns = [col for col in df.columns if 'lap_' in col or col in ['edge_density', 'roi_std', 'gradient_mean']]
            threshold_columns = [col for col in df.columns if 'thresh_' in col]
            all_analysis_columns = laplacian_columns + threshold_columns
            
            correlations = {}
            for col in all_analysis_columns:
                correlations[col] = df['confidence'].corr(df[col])
            
            print(f"\nAnalysis Results:")
            print(f"Total detections: {len(df)}")
            print(f"Total images processed: {df['image_name'].nunique()}")
            print(f"Average detections per image: {len(df) / df['image_name'].nunique():.2f}")
            print(f"Config Laplacian Threshold: {self.laplacian_threshold}")
            
            # Show threshold-specific correlations first (key focus)
            print(f"\n🎯 THRESHOLD ANALYSIS (laplacian_threshold={self.laplacian_threshold}):")
            threshold_corr = [(k, v) for k, v in correlations.items() if 'thresh_' in k]
            threshold_corr.sort(key=lambda x: abs(x[1]), reverse=True)
            for metric, corr in threshold_corr:
                print(f"  {metric}: {corr:.3f}")
            
            # Show top overall correlations
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            print(f"\nTop 10 overall correlations with confidence:")
            for metric, corr in sorted_corr[:10]:
                print(f"  {metric}: {corr:.3f}")
            
            # Create comprehensive visualization
            self._create_visualizations(df, correlations, output_dir)
            
            # Save summary with focus on threshold analysis
            threshold_correlations = [(k, v) for k, v in sorted_corr if 'thresh_' in k]
            
            summary = {
                'total_detections': len(df),
                'unique_images': df['image_name'].nunique(),
                'avg_detections_per_image': len(df) / df['image_name'].nunique(),
                'config_parameters': {
                    'laplacian_threshold': self.laplacian_threshold,
                    'laplacian_kernel_size': self.laplacian_kernel_size,
                    'denoise_template_window_size': self.denoise_template_window_size,
                    'denoise_search_window_size': self.denoise_search_window_size,
                    'denoise_strength': self.denoise_strength,
                    'process_denoise': self.process_denoise
                },
                'confidence_stats': {
                    'mean': float(df['confidence'].mean()),
                    'std': float(df['confidence'].std()),
                    'min': float(df['confidence'].min()),
                    'max': float(df['confidence'].max())
                },
                'threshold_analysis': {
                    'mean_above_threshold_ratio': float(df['thresh_above_threshold_ratio'].mean()),
                    'std_above_threshold_ratio': float(df['thresh_above_threshold_ratio'].std()),
                    'correlations_with_confidence': dict(threshold_correlations[:5])
                },
                'top_correlations': dict(sorted_corr[:15]),
                'laplacian_metrics_analyzed': len(all_analysis_columns)
            }
            
            with open(os.path.join(output_dir, 'analysis_summary.yaml'), 'w') as f:
                yaml.dump(summary, f, default_flow_style=False)
        
        return df
    
    def _create_visualizations(self, df: pd.DataFrame, correlations: Dict, output_dir: str):
        """Create comprehensive visualizations of the analysis results."""
        
        # Set up the plotting style
        plt.style.use('default')
        if SEABORN_AVAILABLE:
            sns.set_palette("husl")
        
        # Figure 1: Overview plots with focus on threshold analysis
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Confidence distribution
        axes[0, 0].hist(df['confidence'], bins=30, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Confidence')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Model Confidence')
        axes[0, 0].grid(True, alpha=0.3)
        
        # KEY PLOT: Above threshold ratio vs confidence
        axes[0, 1].scatter(df['thresh_above_threshold_ratio'], df['confidence'], alpha=0.6, color='red')
        axes[0, 1].set_xlabel('Above Threshold Ratio')
        axes[0, 1].set_ylabel('Confidence')
        thresh_corr = correlations.get('thresh_above_threshold_ratio', 0)
        axes[0, 1].set_title(f'Confidence vs Above Threshold Ratio\n(threshold={self.laplacian_threshold}, corr={thresh_corr:.3f})')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Top correlations bar plot with emphasis on threshold metrics
        top_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        metrics, corr_values = zip(*top_corr)
        colors = ['red' if 'thresh_' in m else ('orange' if 'lap_' in m else 'blue') for m in metrics]
        
        axes[0, 2].barh(range(len(metrics)), corr_values, color=colors, alpha=0.7)
        axes[0, 2].set_yticks(range(len(metrics)))
        axes[0, 2].set_yticklabels([m.replace('thresh_', 'T:').replace('_lap_', '\nL:') for m in metrics], fontsize=8)
        axes[0, 2].set_xlabel('Correlation with Confidence')
        axes[0, 2].set_title('Top 10 Correlations\n(Red=Threshold, Orange=Laplacian)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Denoised Laplacian mean vs confidence (using config parameters)
        axes[1, 0].scatter(df['denoised_lap_mean'], df['confidence'], alpha=0.6)
        axes[1, 0].set_xlabel('Denoised Laplacian Mean')
        axes[1, 0].set_ylabel('Confidence')
        axes[1, 0].set_title(f'Confidence vs Denoised Laplacian Mean\n(kernel={self.laplacian_kernel_size}, corr={correlations.get("denoised_lap_mean", 0):.3f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Mean above vs below threshold comparison
        if 'thresh_mean_above_threshold' in df.columns and 'thresh_mean_below_threshold' in df.columns:
            axes[1, 1].scatter(df['thresh_mean_above_threshold'], df['confidence'], alpha=0.6, label='Above threshold', color='red')
            axes[1, 1].scatter(df['thresh_mean_below_threshold'], df['confidence'], alpha=0.6, label='Below threshold', color='blue')
            axes[1, 1].set_xlabel('Mean Laplacian Value')
            axes[1, 1].set_ylabel('Confidence')
            axes[1, 1].set_title(f'Confidence vs Mean Laplacian\n(Above/Below threshold={self.laplacian_threshold})')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Fallback to edge density
            axes[1, 1].scatter(df['edge_density'], df['confidence'], alpha=0.6)
            axes[1, 1].set_xlabel('Edge Density')
            axes[1, 1].set_ylabel('Confidence')
            axes[1, 1].set_title(f'Confidence vs Edge Density\nCorrelation: {correlations.get("edge_density", 0):.3f}')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Threshold count vs confidence
        axes[1, 2].scatter(df['thresh_above_threshold_count'], df['confidence'], alpha=0.6, color='purple')
        axes[1, 2].set_xlabel('Above Threshold Count')
        axes[1, 2].set_ylabel('Confidence')
        count_corr = correlations.get('thresh_above_threshold_count', 0)
        axes[1, 2].set_title(f'Confidence vs Above Threshold Count\n(threshold={self.laplacian_threshold}, corr={count_corr:.3f})')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'overview_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Figure 2: Laplacian comparison across different preprocessing methods
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        preprocessing_methods = ['raw', 'denoised', 'blurred', 'k5']
        
        for i, method in enumerate(preprocessing_methods):
            ax = axes[i//2, i%2]
            metric = f'{method}_lap_mean'
            if metric in df.columns:
                ax.scatter(df[metric], df['confidence'], alpha=0.6)
                ax.set_xlabel(f'{method.title()} Laplacian Mean')
                ax.set_ylabel('Confidence')
                corr = correlations.get(metric, 0)
                ax.set_title(f'{method.title()} Laplacian vs Confidence\nCorrelation: {corr:.3f}')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'laplacian_preprocessing_comparison.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Figure 3: Correlation heatmap
        plt.figure(figsize=(16, 12))
        
        # Select relevant columns for correlation matrix
        correlation_columns = ['confidence'] + [col for col in df.columns if any(x in col for x in ['lap_', 'edge_', 'roi_', 'gradient_', 'bbox_'])]
        correlation_matrix = df[correlation_columns].corr()
        
        # Create heatmap
        if SEABORN_AVAILABLE:
            mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
            sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
        else:
            # Fallback to matplotlib imshow
            plt.imshow(correlation_matrix, cmap='RdBu_r', aspect='auto')
            plt.colorbar()
            # Add correlation values as text
            for i in range(len(correlation_matrix.columns)):
                for j in range(len(correlation_matrix.columns)):
                    if i != j:  # Don't show diagonal values
                        plt.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}', 
                               ha='center', va='center', fontsize=8)
        
        plt.title('Correlation Matrix of Laplacian Metrics and Confidence')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Configuration
    model_weights_path = "/home/java/hpc-home/cslics_detection/cslics_combined_202508185/weights/best.pt"
    config_path = "/home/java/hpc-home/Corals/cslic/coral_spawn_counter/data_yaml_files/annotation_cslics_2024_nov_pdae_tank4_100000001ab0438d.yaml"
    image_dir = "/home/java/hpc-home/Data/cslics/2023_2024_combined_dataset/cslics_2024_species_data/cslics_2024_pdae_438d_1_1000_split/test/images"
    output_dir = "/home/java/hpc-home/Data/cslics/bounding_box_laplacian_results"
    
    # Create analyzer
    analyzer = BoundingBoxLaplacianAnalyzer(
        model_weights_path=model_weights_path,
        config_path=config_path,
        confidence_threshold=0.3
    )
    
    # Run analysis
    results_df = analyzer.analyze_directory(
        image_dir=image_dir,
        output_dir=output_dir,
        max_images=300  
    )
    
    print(f"✓ Analysis complete! Results saved to {output_dir}")
