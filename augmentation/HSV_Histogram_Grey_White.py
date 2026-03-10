#!/usr/bin/env python3++
import os
import json
import numpy as np
import matplotlib.pyplot as plt

# Output directory where JSON files are cached (same as in RGB_HSV_converter.py)
output_dir = "Data/cslics/2025_dec_spawn/hsv_cache"

# Tank definitions matching the grey and white directories
grey_tanks = {
    "Tank_1421024219711": "Data/cslics/2025_dec_spawn/hsv_cache/grey/1421024219711",
    "Tank_1421024251440": "Data/cslics/2025_dec_spawn/hsv_cache/grey/1421024251440"
}

white_tanks = {
    "Tank_1421024220172": "Data/cslics/2025_dec_spawn/hsv_cache/white/1421024220172",
    "Tank_1422724372929": "Data/cslics/2025_dec_spawn/hsv_cache/white/1422724372929"
}

def load_data_from_json_dir(json_dir):
    """
    Load all JSON files from a directory and return HSV data.
    
    Args:
        json_dir: Directory containing JSON files with HSV data
    
    Returns:
        Dictionary mapping filenames to avg_hsv values
    """
    avg_HSV_data = {}
    
    if not os.path.exists(json_dir):
        print(f"Warning: Directory not found: {json_dir}")
        return avg_HSV_data
    
    for root, dirs, files in os.walk(json_dir):
        for file in files:
            if file.endswith('.json'):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        avg_HSV_data[file] = data['avg_hsv']
                except Exception as e:
                    print(f"Error loading {file_path}: {e}")
    
    return avg_HSV_data


def plot_tank_histograms(tank_data_dict, category_name, colors=None):
    """
    Create histograms for H, S, V values for multiple tanks.
    
    Args:
        tank_data_dict: Dictionary of {tank_name: hsv_data_dict}
        category_name: Name for the category (e.g., 'Grey Tanks', 'White Tanks')
        colors: List of colors for each tank
    """
    if colors is None:
        colors = ['r', 'b', 'g', 'orange', 'purple', 'cyan']
    
    # Filter out empty tanks
    tank_data_dict = {k: v for k, v in tank_data_dict.items() if len(v) > 0}
    
    if len(tank_data_dict) == 0:
        print(f"No data found for {category_name}")
        return
    
    # Print statistics for each tank
    print(f"\n{'='*80}")
    print(f"{category_name} - Statistics")
    print(f"{'='*80}")
    for tank_name, tank_data in tank_data_dict.items():
        print(f"\n{tank_name} ({len(tank_data)} images):")
        print(f"  Lowest Hue:       {min(tank_data.values(), key=lambda x: x[0])[0]:.3f}")
        print(f"  Highest Hue:      {max(tank_data.values(), key=lambda x: x[0])[0]:.3f}")
        print(f"  Lowest Saturation: {min(tank_data.values(), key=lambda x: x[1])[1]:.3f}")
        print(f"  Highest Saturation: {max(tank_data.values(), key=lambda x: x[1])[1]:.3f}")
        print(f"  Lowest Value:     {min(tank_data.values(), key=lambda x: x[2])[2]:.3f}")
        print(f"  Highest Value:    {max(tank_data.values(), key=lambda x: x[2])[2]:.3f}")
    
    # Create figure with 3 subplots for H, S, V
    plt.figure(figsize=(20, 6))
    
    # Plot Hue Values
    plt.subplot(1, 3, 1)
    all_hue_values = [value[0] for data in tank_data_dict.values() for value in data.values()]
    bin_edges = np.histogram_bin_edges(all_hue_values, bins=30)
    
    for i, (tank_name, tank_data) in enumerate(tank_data_dict.items()):
        hue_values = [value[0] for value in tank_data.values()]
        plt.hist(hue_values, bins=bin_edges, alpha=0.5, 
                color=colors[i % len(colors)], label=tank_name)
    
    plt.xlabel('Hue Value')
    plt.ylabel('Frequency')
    plt.title(f'{category_name} - Histogram of Hue Values')
    plt.legend()
    
    # Plot Saturation Values
    plt.subplot(1, 3, 2)
    all_sat_values = [value[1] for data in tank_data_dict.values() for value in data.values()]
    bin_edges = np.histogram_bin_edges(all_sat_values, bins=30)
    
    for i, (tank_name, tank_data) in enumerate(tank_data_dict.items()):
        sat_values = [value[1] for value in tank_data.values()]
        plt.hist(sat_values, bins=bin_edges, alpha=0.5, 
                color=colors[i % len(colors)], label=tank_name)
    
    plt.xlabel('Saturation Value')
    plt.ylabel('Frequency')
    plt.title(f'{category_name} - Histogram of Saturation Values')
    plt.legend()
    
    # Plot Value Values
    plt.subplot(1, 3, 3)
    all_val_values = [value[2] for data in tank_data_dict.values() for value in data.values()]
    bin_edges = np.histogram_bin_edges(all_val_values, bins=30)
    
    for i, (tank_name, tank_data) in enumerate(tank_data_dict.items()):
        val_values = [value[2] for value in tank_data.values()]
        plt.hist(val_values, bins=bin_edges, alpha=0.5, 
                color=colors[i % len(colors)], label=tank_name)
    
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(f'{category_name} - Histogram of Value Values')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()


def plot_combined_grey_white_histograms(grey_data_dict, white_data_dict):
    """
    Create combined histograms comparing all grey tanks vs all white tanks.
    
    Args:
        grey_data_dict: Dictionary of {tank_name: hsv_data_dict} for grey tanks
        white_data_dict: Dictionary of {tank_name: hsv_data_dict} for white tanks
    """
    # Combine all grey and white data
    all_grey_data = []
    all_white_data = []
    
    for tank_data in grey_data_dict.values():
        all_grey_data.extend(tank_data.values())
    
    for tank_data in white_data_dict.values():
        all_white_data.extend(tank_data.values())
    
    if len(all_grey_data) == 0 or len(all_white_data) == 0:
        print("Insufficient data for combined comparison")
        return
    
    # Create figure with 3 subplots for H, S, V
    plt.figure(figsize=(20, 6))
    
    # Plot Hue Values
    plt.subplot(1, 3, 1)
    all_values = [v[0] for v in all_grey_data] + [v[0] for v in all_white_data]
    bin_edges = np.histogram_bin_edges(all_values, bins=30)
    
    plt.hist([v[0] for v in all_grey_data], bins=bin_edges, alpha=0.6, 
            color='gray', label=f'Grey Tanks (n={len(all_grey_data)})')
    plt.hist([v[0] for v in all_white_data], bins=bin_edges, alpha=0.6, 
            color='white', edgecolor='black', label=f'White Tanks (n={len(all_white_data)})')
    plt.xlabel('Hue Value')
    plt.ylabel('Frequency')
    plt.title('Grey vs White - Histogram of Hue Values')
    plt.legend()
    
    # Plot Saturation Values
    plt.subplot(1, 3, 2)
    all_values = [v[1] for v in all_grey_data] + [v[1] for v in all_white_data]
    bin_edges = np.histogram_bin_edges(all_values, bins=30)
    
    plt.hist([v[1] for v in all_grey_data], bins=bin_edges, alpha=0.6, 
            color='gray', label=f'Grey Tanks (n={len(all_grey_data)})')
    plt.hist([v[1] for v in all_white_data], bins=bin_edges, alpha=0.6, 
            color='white', edgecolor='black', label=f'White Tanks (n={len(all_white_data)})')
    plt.xlabel('Saturation Value')
    plt.ylabel('Frequency')
    plt.title('Grey vs White - Histogram of Saturation Values')
    plt.legend()
    
    # Plot Value Values
    plt.subplot(1, 3, 3)
    all_values = [v[2] for v in all_grey_data] + [v[2] for v in all_white_data]
    bin_edges = np.histogram_bin_edges(all_values, bins=30)
    
    plt.hist([v[2] for v in all_grey_data], bins=bin_edges, alpha=0.6, 
            color='gray', label=f'Grey Tanks (n={len(all_grey_data)})')
    plt.hist([v[2] for v in all_white_data], bins=bin_edges, alpha=0.6, 
            color='white', edgecolor='black', label=f'White Tanks (n={len(all_white_data)})')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Grey vs White - Histogram of Value Values')
    plt.legend()
    
    plt.tight_layout()


if __name__ == '__main__':
    print("="*80)
    print("HSV Histogram Analysis - Grey vs White Tanks")
    print("="*80)
    
    # Load data for grey tanks
    print("\nLoading grey tank data...")
    grey_data_dict = {}
    for tank_name, json_dir in grey_tanks.items():
        data = load_data_from_json_dir(json_dir)
        grey_data_dict[tank_name] = data
        print(f"  {tank_name}: {len(data)} images")
    
    # Load data for white tanks
    print("\nLoading white tank data...")
    white_data_dict = {}
    for tank_name, json_dir in white_tanks.items():
        data = load_data_from_json_dir(json_dir)
        white_data_dict[tank_name] = data
        print(f"  {tank_name}: {len(data)} images")
    
    # Plot individual tank histograms
    if len(grey_data_dict) > 0:
        plot_tank_histograms(grey_data_dict, "Grey Tanks", colors=['darkgray', 'dimgray'])
    
    if len(white_data_dict) > 0:
        plot_tank_histograms(white_data_dict, "White Tanks", colors=['lightcoral', 'lightblue'])
    
    # Plot combined comparison
    plot_combined_grey_white_histograms(grey_data_dict, white_data_dict)
    
    # Show all plots
    plt.show()
