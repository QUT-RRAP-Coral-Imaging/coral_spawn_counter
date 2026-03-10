#!/usr/bin/env python3

"""
Compare YOLOv8 model predictions between two domains (e.g., lights on vs lights off).
This script runs inference on multiple directories and compares average predictions
to evaluate domain-specific performance.

Usage:
    python compare_domain_predictions.py
"""

import os
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple
import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils import ops


class DomainComparisonPredictor:
    """
    Compare YOLO model predictions across two different domains.
    """
    
    def __init__(self, 
                 weights_path: str,
                 domain1_dirs: List[str],
                 domain2_dirs: List[str],
                 domain1_name: str = "Domain 1",
                 domain2_name: str = "Domain 2",
                 save_dir: str = None,
                 iou_thresh: float = 0.5,
                 conf_thresh: float = 0.25,
                 max_det: int = 1000,
                 save_predictions: bool = True):
        """
        Initialize the domain comparison predictor.
        
        Args:
            weights_path: Path to YOLO model weights
            domain1_dirs: List of directories containing images for domain 1
            domain2_dirs: List of directories containing images for domain 2
            domain1_name: Name/label for domain 1 (e.g., "Lights On")
            domain2_name: Name/label for domain 2 (e.g., "Lights Off")
            save_dir: Directory to save results (default: current directory)
            iou_thresh: IoU threshold for NMS
            conf_thresh: Confidence threshold for detections
            max_det: Maximum number of detections per image
            save_predictions: Whether to save individual prediction files
        """
        self.weights_path = weights_path
        self.domain1_dirs = [Path(d) for d in domain1_dirs]
        self.domain2_dirs = [Path(d) for d in domain2_dirs]
        self.domain1_name = domain1_name
        self.domain2_name = domain2_name
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh
        self.max_det = max_det
        self.save_predictions = save_predictions
        self.current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        # Setup save directory
        if save_dir is None:
            save_dir = f'domain_comparison_{self.current_datetime}'
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        print(f'Loading model: {weights_path}')
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        print(f'Using device: {self.device}')
        self.model = YOLO(weights_path).to(self.device)
        
        # Override to use DetectionPredictor instead of SegmentationPredictor
        self.model.predictor = None
        self.model.task = 'detect'
        print(f'Model loaded - using detection mode (bounding boxes only)')
        
        self.domain1_results = []
        self.domain2_results = []
        
    
    def get_image_list(self, directories: List[Path]) -> List[Path]:
        """
        Get list of all images from multiple directories.
        
        Args:
            directories: List of directory paths
            
        Returns:
            List of image paths
        """
        img_list = []
        for directory in directories:
            if not directory.exists():
                print(f"Warning: Directory does not exist: {directory}")
                continue
            img_list.extend(sorted(directory.glob('*.jpg')))
            img_list.extend(sorted(directory.glob('*.JPG')))
            img_list.extend(sorted(directory.glob('*.png')))
            img_list.extend(sorted(directory.glob('*.PNG')))
        return img_list
    
    
    def predict_image(self, img_path: Path) -> Dict:
        """
        Run prediction on a single image.
        
        Args:
            img_path: Path to image
            
        Returns:
            Dictionary with prediction results
        """
        try:
            results = self.model(
                source=str(img_path),
                conf=self.conf_thresh,
                iou=self.iou_thresh,
                max_det=self.max_det,
                verbose=False,
                stream=False
            )
            
            boxes = results[0].boxes
            predictions = []
            
            if boxes is not None and len(boxes) > 0:
                for b in boxes:
                    if torch.cuda.is_available():
                        xyxyn = b.xyxyn[0].cpu()
                        conf = b.conf.cpu().item()
                        cls = b.cls.cpu().item()
                    else:
                        xyxyn = b.xyxyn[0]
                        conf = b.conf.item()
                        cls = b.cls.item()
                    
                    predictions.append({
                        'bbox': [xyxyn[0].item(), xyxyn[1].item(), 
                                xyxyn[2].item(), xyxyn[3].item()],
                        'confidence': conf,
                        'class': int(cls)
                    })
            
            return {
                'image_path': str(img_path),
                'image_name': img_path.name,
                'num_detections': len(predictions),
                'predictions': predictions,
                'avg_confidence': np.mean([p['confidence'] for p in predictions]) if predictions else 0.0
            }
            
        except Exception as e:
            print(f"  Warning: Could not process {img_path.name}: {str(e)}")
            return {
                'image_path': str(img_path),
                'image_name': img_path.name,
                'num_detections': 0,
                'predictions': [],
                'avg_confidence': 0.0
            }
    
    
    def predict_domain(self, directories: List[Path], domain_name: str) -> List[Dict]:
        """
        Run predictions on all images in a domain.
        
        Args:
            directories: List of directories for this domain
            domain_name: Name of the domain
            
        Returns:
            List of prediction results
        """
        img_list = self.get_image_list(directories)
        print(f'\n{domain_name}:')
        print(f'  Directories: {[str(d) for d in directories]}')
        print(f'  Number of images: {len(img_list)}')
        
        if len(img_list) == 0:
            print(f'  Warning: No images found in {domain_name}!')
            return []
        
        results = []
        start_time = time.time()
        
        for i, img_path in enumerate(img_list):
            if (i + 1) % 50 == 0 or i == 0:
                print(f'  Processing: {i+1}/{len(img_list)}')
            
            result = self.predict_image(img_path)
            results.append(result)
        
        end_time = time.time()
        duration = end_time - start_time
        
        print(f'  Completed in {duration:.2f} sec ({duration/len(img_list):.3f} sec/image)')
        
        return results
    
    
    def save_domain_results(self, results: List[Dict], domain_name: str):
        """
        Save detailed results for a domain to JSON file.
        
        Args:
            results: List of prediction results
            domain_name: Name of the domain
        """
        if not self.save_predictions:
            return
        
        output_file = self.save_dir / f'{domain_name.replace(" ", "_").lower()}_predictions.json'
        
        output_data = {
            'model_path': self.weights_path,
            'domain_name': domain_name,
            'date_run': self.current_datetime,
            'num_images': len(results),
            'results': results
        }
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f'  Saved predictions to: {output_file}')
    
    
    def compute_rolling_average(self, results: List[Dict], window_size: int = 5) -> List[float]:
        """
        Compute rolling average of detections per image.
        
        Args:
            results: List of prediction results
            window_size: Window size for rolling average
            
        Returns:
            List of rolling averages
        """
        if not results or len(results) < window_size:
            return [r['num_detections'] for r in results]
        
        detections = [r['num_detections'] for r in results]
        rolling_avgs = []
        
        for i in range(len(detections)):
            if i < window_size - 1:
                rolling_avgs.append(float(np.mean(detections[:i+1])))
            else:
                rolling_avgs.append(float(np.mean(detections[i-window_size+1:i+1])))
        
        return rolling_avgs
    
    
    def compute_statistics(self, results: List[Dict], window_size: int = 5) -> Dict:
        """
        Compute statistics for a set of results including rolling averages.
        
        Args:
            results: List of prediction results
            window_size: Window size for rolling average computation
            
        Returns:
            Dictionary with statistics
        """
        if not results:
            return {
                'num_images': 0,
                'total_detections': 0,
                'avg_detections_per_image': 0.0,
                'std_detections_per_image': 0.0,
                'median_detections_per_image': 0.0,
                'avg_confidence': 0.0,
                'std_confidence': 0.0,
                'median_confidence': 0.0,
                'min_detections': 0,
                'max_detections': 0,
                'rolling_avg_mean': 0.0,
                'rolling_avg_std': 0.0,
                'rolling_averages': []
            }
        
        detections_per_image = [r['num_detections'] for r in results]
        all_confidences = [p['confidence'] for r in results for p in r['predictions']]
        rolling_avgs = self.compute_rolling_average(results, window_size)
        
        stats = {
            'num_images': int(len(results)),
            'total_detections': int(sum(detections_per_image)),
            'avg_detections_per_image': float(np.mean(detections_per_image)),
            'std_detections_per_image': float(np.std(detections_per_image)),
            'median_detections_per_image': float(np.median(detections_per_image)),
            'min_detections': int(np.min(detections_per_image)),
            'max_detections': int(np.max(detections_per_image)),
            'rolling_avg_mean': float(np.mean(rolling_avgs)),
            'rolling_avg_std': float(np.std(rolling_avgs)),
            'rolling_averages': rolling_avgs
        }
        
        if all_confidences:
            stats.update({
                'avg_confidence': float(np.mean(all_confidences)),
                'std_confidence': float(np.std(all_confidences)),
                'median_confidence': float(np.median(all_confidences)),
                'min_confidence': float(np.min(all_confidences)),
                'max_confidence': float(np.max(all_confidences))
            })
        else:
            stats.update({
                'avg_confidence': 0.0,
                'std_confidence': 0.0,
                'median_confidence': 0.0,
                'min_confidence': 0.0,
                'max_confidence': 0.0
            })
        
        return stats
    
    
    def plot_comparison(self, stats1: Dict, stats2: Dict):
        """
        Create comparison plots between two domains.
        
        Args:
            stats1: Statistics for domain 1
            stats2: Statistics for domain 2
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'Domain Comparison: {self.domain1_name} vs {self.domain2_name}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Average detections per image
        ax = axes[0, 0]
        domains = [self.domain1_name, self.domain2_name]
        avg_dets = [stats1['avg_detections_per_image'], stats2['avg_detections_per_image']]
        std_dets = [stats1['std_detections_per_image'], stats2['std_detections_per_image']]
        
        bars = ax.bar(domains, avg_dets, yerr=std_dets, capsize=10, 
                     color=['#3498db', '#e74c3c'], alpha=0.7)
        ax.set_ylabel('Average Detections per Image')
        ax.set_title('Average Detections per Image')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, avg_dets):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax = axes[0, 1]
        median_dets = [stats1['median_detections_per_image'], stats2['median_detections_per_image']]
        bars = ax.bar(domains, median_dets, color=['#3498db', '#e74c3c'], alpha=0.7)
        ax.set_ylabel('Median Detections per Image')
        ax.set_title('Median Detections per Image')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, median_dets):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.2f}', ha='center', va='bottom', fontweight='bold')
        
        ax = axes[0, 2]
        total_dets = [stats1['total_detections'], stats2['total_detections']]
        bars = ax.bar(domains, total_dets, color=['#3498db', '#e74c3c'], alpha=0.7)
        ax.set_ylabel('Total Detections')
        ax.set_title('Total Detections')
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, total_dets):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val}', ha='center', va='bottom', fontweight='bold')
        
        ax = axes[1, 0]
        avg_conf = [stats1['avg_confidence'], stats2['avg_confidence']]
        std_conf = [stats1['std_confidence'], stats2['std_confidence']]
        bars = ax.bar(domains, avg_conf, yerr=std_conf, capsize=10,
                     color=['#3498db', '#e74c3c'], alpha=0.7)
        ax.set_ylabel('Average Confidence')
        ax.set_title('Average Confidence')
        ax.set_ylim([0, 1.0])
        ax.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, avg_conf):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax = axes[1, 1]
        x_pos = np.arange(len(domains))
        min_dets = [stats1['min_detections'], stats2['min_detections']]
        max_dets = [stats1['max_detections'], stats2['max_detections']]
        
        ax.scatter(x_pos, min_dets, label='Min', s=100, color='green', marker='v', alpha=0.7)
        ax.scatter(x_pos, max_dets, label='Max', s=100, color='red', marker='^', alpha=0.7)
        
        # Draw lines connecting min to max
        for i in range(len(domains)):
            ax.plot([i, i], [min_dets[i], max_dets[i]], 'k-', alpha=0.5, linewidth=2)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(domains)
        ax.set_ylabel('Number of Detections')
        ax.set_title('Detection Range (Min-Max)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 2]
        ax.axis('tight')
        ax.axis('off')
        
        table_data = [
            ['Metric', self.domain1_name, self.domain2_name],
            ['# Images', f"{stats1['num_images']}", f"{stats2['num_images']}"],
            ['Total Detections', f"{stats1['total_detections']}", f"{stats2['total_detections']}"],
            ['Avg Det/Image', f"{stats1['avg_detections_per_image']:.2f}", 
             f"{stats2['avg_detections_per_image']:.2f}"],
            ['Avg Confidence', f"{stats1['avg_confidence']:.3f}", 
             f"{stats2['avg_confidence']:.3f}"],
        ]
        
        table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                        colWidths=[0.35, 0.325, 0.325])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        for i in range(3):
            table[(0, i)].set_facecolor('#34495e')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.tight_layout()
        
        plot_path = self.save_dir / f'domain_comparison_{self.current_datetime}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'\nSaved comparison plot to: {plot_path}')
        
        plt.show()
    
    
    def plot_rolling_average_comparison(self, stats1: Dict, stats2: Dict):
        """
        Create time series plots comparing rolling averages between domains.
        
        Args:
            stats1: Statistics for domain 1
            stats2: Statistics for domain 2
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle(f'Rolling Average Comparison: {self.domain1_name} vs {self.domain2_name}', 
                     fontsize=16, fontweight='bold')
        
        rolling_avgs1 = stats1['rolling_averages']
        rolling_avgs2 = stats2['rolling_averages']
        
        ax = axes[0]
        x1 = np.arange(len(rolling_avgs1))
        x2 = np.arange(len(rolling_avgs2))
        
        ax.plot(x1, rolling_avgs1, color='#3498db', linewidth=2, label=self.domain1_name, alpha=0.8)
        ax.axhline(y=stats1['rolling_avg_mean'], color='#3498db', linestyle='--', 
                   linewidth=1.5, alpha=0.6, label=f'{self.domain1_name} Mean')
        
        ax.plot(x2, rolling_avgs2, color='#e74c3c', linewidth=2, label=self.domain2_name, alpha=0.8)
        ax.axhline(y=stats2['rolling_avg_mean'], color='#e74c3c', linestyle='--', 
                   linewidth=1.5, alpha=0.6, label=f'{self.domain2_name} Mean')
        
        ax.set_xlabel('Image Index')
        ax.set_ylabel('Rolling Average Detections')
        ax.set_title('Rolling Average Detections Over Time (Window=5)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        bins = np.linspace(
            min(min(rolling_avgs1), min(rolling_avgs2)) - 1,
            max(max(rolling_avgs1), max(rolling_avgs2)) + 1,
            30
        )
        
        ax.hist(rolling_avgs1, bins=bins, color='#3498db', alpha=0.6, 
                label=f'{self.domain1_name}\n(Mean: {stats1["rolling_avg_mean"]:.2f} ± {stats1["rolling_avg_std"]:.2f})',
                edgecolor='black')
        ax.hist(rolling_avgs2, bins=bins, color='#e74c3c', alpha=0.6,
                label=f'{self.domain2_name}\n(Mean: {stats2["rolling_avg_mean"]:.2f} ± {stats2["rolling_avg_std"]:.2f})',
                edgecolor='black')
        
        ax.set_xlabel('Rolling Average Detections')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Rolling Average Detections')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.save_dir / f'rolling_average_comparison_{self.current_datetime}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'Saved rolling average plot to: {plot_path}')
        
        plt.show()
    
    
    def plot_histograms(self):
        """
        Create histograms of detections per image for both domains.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Distribution of Detections per Image', fontsize=16, fontweight='bold')
        
        # Domain 1 histogram
        ax = axes[0]
        detections1 = [r['num_detections'] for r in self.domain1_results]
        ax.hist(detections1, bins=30, color='#3498db', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Detections')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.domain1_name}\n(Mean: {np.mean(detections1):.2f}, Median: {np.median(detections1):.2f})')
        ax.grid(True, alpha=0.3)
        
        ax = axes[1]
        detections2 = [r['num_detections'] for r in self.domain2_results]
        ax.hist(detections2, bins=30, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Detections')
        ax.set_ylabel('Frequency')
        ax.set_title(f'{self.domain2_name}\n(Mean: {np.mean(detections2):.2f}, Median: {np.median(detections2):.2f})')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        plot_path = self.save_dir / f'detection_histograms_{self.current_datetime}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f'Saved histogram plot to: {plot_path}')
        
        plt.show()
    
    
    def save_comparison_report(self, stats1: Dict, stats2: Dict):
        """
        Save a detailed comparison report as text and JSON.
        
        Args:
            stats1: Statistics for domain 1
            stats2: Statistics for domain 2
        """
        # Text report
        report_txt = self.save_dir / f'comparison_report_{self.current_datetime}.txt'
        
        with open(report_txt, 'w') as f:
            f.write("="*70 + "\n")
            f.write("DOMAIN COMPARISON REPORT\n")
            f.write("="*70 + "\n\n")
            f.write(f"Model: {self.weights_path}\n")
            f.write(f"Date: {self.current_datetime}\n")
            f.write(f"IoU Threshold: {self.iou_thresh}\n")
            f.write(f"Confidence Threshold: {self.conf_thresh}\n\n")
            
            f.write("-"*70 + "\n")
            f.write(f"DOMAIN 1: {self.domain1_name}\n")
            f.write("-"*70 + "\n")
            f.write(f"Directories:\n")
            for d in self.domain1_dirs:
                f.write(f"  - {d}\n")
            f.write(f"\nNumber of Images: {stats1['num_images']}\n")
            f.write(f"Total Detections: {stats1['total_detections']}\n")
            f.write(f"Average Detections per Image: {stats1['avg_detections_per_image']:.3f} ± {stats1['std_detections_per_image']:.3f}\n")
            f.write(f"Median Detections per Image: {stats1['median_detections_per_image']:.3f}\n")
            f.write(f"Detection Range: [{stats1['min_detections']}, {stats1['max_detections']}]\n")
            f.write(f"Average Confidence: {stats1['avg_confidence']:.3f} ± {stats1['std_confidence']:.3f}\n")
            f.write(f"Median Confidence: {stats1['median_confidence']:.3f}\n")
            f.write(f"Rolling Average (window=5): {stats1['rolling_avg_mean']:.3f} ± {stats1['rolling_avg_std']:.3f}\n\n")
            
            f.write("-"*70 + "\n")
            f.write(f"DOMAIN 2: {self.domain2_name}\n")
            f.write("-"*70 + "\n")
            f.write(f"Directories:\n")
            for d in self.domain2_dirs:
                f.write(f"  - {d}\n")
            f.write(f"\nNumber of Images: {stats2['num_images']}\n")
            f.write(f"Total Detections: {stats2['total_detections']}\n")
            f.write(f"Average Detections per Image: {stats2['avg_detections_per_image']:.3f} ± {stats2['std_detections_per_image']:.3f}\n")
            f.write(f"Median Detections per Image: {stats2['median_detections_per_image']:.3f}\n")
            f.write(f"Detection Range: [{stats2['min_detections']}, {stats2['max_detections']}]\n")
            f.write(f"Average Confidence: {stats2['avg_confidence']:.3f} ± {stats2['std_confidence']:.3f}\n")
            f.write(f"Median Confidence: {stats2['median_confidence']:.3f}\n")
            f.write(f"Rolling Average (window=5): {stats2['rolling_avg_mean']:.3f} ± {stats2['rolling_avg_std']:.3f}\n\n")
            
            f.write("="*70 + "\n")
            f.write("COMPARISON\n")
            f.write("="*70 + "\n")
            
            diff_avg_det = stats2['avg_detections_per_image'] - stats1['avg_detections_per_image']
            pct_change_det = (diff_avg_det / stats1['avg_detections_per_image'] * 100) if stats1['avg_detections_per_image'] > 0 else 0
            
            diff_avg_conf = stats2['avg_confidence'] - stats1['avg_confidence']
            pct_change_conf = (diff_avg_conf / stats1['avg_confidence'] * 100) if stats1['avg_confidence'] > 0 else 0
            
            f.write(f"Average Detections Difference: {diff_avg_det:+.3f} ({pct_change_det:+.2f}%)\n")
            f.write(f"Average Confidence Difference: {diff_avg_conf:+.3f} ({pct_change_conf:+.2f}%)\n\n")
            
            if abs(pct_change_det) > 10:
                f.write(f"⚠ Significant difference (>10%) in detection counts between domains.\n")
            if abs(pct_change_conf) > 5:
                f.write(f"⚠ Significant difference (>5%) in confidence between domains.\n")
        
        print(f'Saved text report to: {report_txt}')
        
        report_json = self.save_dir / f'comparison_report_{self.current_datetime}.json'
        
        stats1_copy = {k: v for k, v in stats1.items() if k != 'rolling_averages'}
        stats2_copy = {k: v for k, v in stats2.items() if k != 'rolling_averages'}
        
        comparison_data = {
            'model_path': self.weights_path,
            'date_run': self.current_datetime,
            'iou_threshold': self.iou_thresh,
            'confidence_threshold': self.conf_thresh,
            'domain1': {
                'name': self.domain1_name,
                'directories': [str(d) for d in self.domain1_dirs],
                'statistics': stats1_copy
            },
            'domain2': {
                'name': self.domain2_name,
                'directories': [str(d) for d in self.domain2_dirs],
                'statistics': stats2_copy
            },
            'comparison': {
                'avg_detections_difference': diff_avg_det,
                'avg_detections_pct_change': pct_change_det,
                'avg_confidence_difference': diff_avg_conf,
                'avg_confidence_pct_change': pct_change_conf
            }
        }
        
        with open(report_json, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f'Saved JSON report to: {report_json}')
    
    
    def run(self):
        """
        Run the complete domain comparison pipeline.
        """
        print("="*70)
        print("DOMAIN COMPARISON - YOLO MODEL EVALUATION")
        print("="*70)
        print(f"Model: {self.weights_path}")
        print(f"Save Directory: {self.save_dir}")
        print("="*70)
        
        print("\n" + "="*70)
        print("RUNNING PREDICTIONS ON DOMAIN 1")
        print("="*70)
        self.domain1_results = self.predict_domain(self.domain1_dirs, self.domain1_name)
        self.save_domain_results(self.domain1_results, self.domain1_name)
        
        print("\n" + "="*70)
        print("RUNNING PREDICTIONS ON DOMAIN 2")
        print("="*70)
        self.domain2_results = self.predict_domain(self.domain2_dirs, self.domain2_name)
        self.save_domain_results(self.domain2_results, self.domain2_name)
        
        print("\n" + "="*70)
        print("COMPUTING STATISTICS")
        print("="*70)
        stats1 = self.compute_statistics(self.domain1_results)
        stats2 = self.compute_statistics(self.domain2_results)
        
        # Print summary
        print(f"\n{self.domain1_name}:")
        print(f"  Images: {stats1['num_images']}")
        print(f"  Avg Detections/Image: {stats1['avg_detections_per_image']:.3f} ± {stats1['std_detections_per_image']:.3f}")
        print(f"  Avg Confidence: {stats1['avg_confidence']:.3f} ± {stats1['std_confidence']:.3f}")
        print(f"  Rolling Avg: {stats1['rolling_avg_mean']:.3f} ± {stats1['rolling_avg_std']:.3f}")
        
        print(f"\n{self.domain2_name}:")
        print(f"  Images: {stats2['num_images']}")
        print(f"  Avg Detections/Image: {stats2['avg_detections_per_image']:.3f} ± {stats2['std_detections_per_image']:.3f}")
        print(f"  Avg Confidence: {stats2['avg_confidence']:.3f} ± {stats2['std_confidence']:.3f}")
        print(f"  Rolling Avg: {stats2['rolling_avg_mean']:.3f} ± {stats2['rolling_avg_std']:.3f}")
        
        print("\n" + "="*70)
        print("GENERATING REPORTS")
        print("="*70)
        self.save_comparison_report(stats1, stats2)
        
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        self.plot_comparison(stats1, stats2)
        self.plot_rolling_average_comparison(stats1, stats2)
        self.plot_histograms()
        
        print("\n" + "="*70)
        print("COMPARISON COMPLETE")
        print("="*70)


def main():
    """
    Main function with example usage.
    """
    parser = argparse.ArgumentParser(
        description='Compare YOLOv8 predictions between two domains',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python compare_domain_predictions.py \\
    --weights /path/to/model.pt \\
    --domain1-dirs /path/to/lights_on_1 /path/to/lights_on_2 \\
    --domain2-dirs /path/to/lights_off_1 /path/to/lights_off_2 \\
    --domain1-name "Lights On" \\
    --domain2-name "Lights Off" \\
    --save-dir ./results
        """
    )
    
    parser.add_argument('--weights', type=str, required=True,
                       help='Path to YOLO model weights (.pt file)')
    parser.add_argument('--domain1-dirs', type=str, nargs='+', required=True,
                       help='List of directories for domain 1')
    parser.add_argument('--domain2-dirs', type=str, nargs='+', required=True,
                       help='List of directories for domain 2')
    parser.add_argument('--domain1-name', type=str, default='Domain 1',
                       help='Name for domain 1 (default: Domain 1)')
    parser.add_argument('--domain2-name', type=str, default='Domain 2',
                       help='Name for domain 2 (default: Domain 2)')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save results (default: domain_comparison_<timestamp>)')
    parser.add_argument('--iou', type=float, default=0.5,
                       help='IoU threshold for NMS (default: 0.5)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (default: 0.25)')
    parser.add_argument('--max-det', type=int, default=1000,
                       help='Maximum detections per image (default: 1000)')
    parser.add_argument('--no-save-predictions', action='store_true',
                       help='Do not save individual prediction JSON files')
    
    args = parser.parse_args()
    
    predictor = DomainComparisonPredictor(
        weights_path=args.weights,
        domain1_dirs=args.domain1_dirs,
        domain2_dirs=args.domain2_dirs,
        domain1_name=args.domain1_name,
        domain2_name=args.domain2_name,
        save_dir=args.save_dir,
        iou_thresh=args.iou,
        conf_thresh=args.conf,
        max_det=args.max_det,
        save_predictions=not args.no_save_predictions
    )
    
    predictor.run()


if __name__ == "__main__":
    USE_COMMAND_LINE = False
    
    if USE_COMMAND_LINE:
        main()
    else:
        weights_path = '/home/java/hpc-home/cslics_detection/cslics_2025_combined_20262424_HSV-0.5/weights/best.pt'
        
        domain1_dirs = [
            '/home/java/hpc-home/Data/cslics/2025_dec_spawn/1421024251440/1440_Testset/Lights_on',
        ]
        
        domain2_dirs = [
            '/home/java/hpc-home/Data/cslics/2025_dec_spawn/1421024251440/1440_Testset/Lights_off',
        ]
        
        predictor = DomainComparisonPredictor(
            weights_path=weights_path,
            domain1_dirs=domain1_dirs,
            domain2_dirs=domain2_dirs,
            domain1_name="Lights On",
            domain2_name="Lights Off",
            save_dir="./domain_comparison_results",
            iou_thresh=0.5,
            conf_thresh=0.25,
            max_det=1000
        )
        
        predictor.run()
