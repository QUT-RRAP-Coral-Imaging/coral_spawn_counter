#!/usr/bin/env python3++
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from RGB_HSV_converter import HSVInfo
from concurrent.futures import ThreadPoolExecutor
import time
import json

def load_data(json_dirs):
    avg_HSV_data = {}
    for json_dir in json_dirs:
        for root, dirs, files in os.walk(json_dir):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as json_file:
                        data = json.load(json_file)
                        avg_HSV_data[file] = data['avg_hsv']
    return avg_HSV_data

def load_image(file_path):
    img = cv2.imread(file_path)
    if img is not None:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        avg_hsv = np.mean(hsv_img, axis=(0, 1))
        return avg_hsv.tolist() 
    return None

def save_data(data_dir, save_dir, max_images=10000000):
    end_dir = os.path.basename(os.path.normpath(data_dir))
    save_json_dir = os.path.join(save_dir, end_dir)
    os.makedirs(save_json_dir, exist_ok=True)
    
    #avg_hsv_data = {}
    image_counter = 0
    file_paths = []
    
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('clean.jpg'):
                file_paths.append(os.path.join(root, file))

    start_time = time.time()  # Start the stopwatch    
    with ThreadPoolExecutor() as executor:
        for file_path, avg_hsv in zip(file_paths, executor.map(load_image, file_paths)):
            if avg_hsv is not None:
                file_name = os.path.splitext(os.path.basename(file_path))[0] + '.json'
                json_path = os.path.join(save_json_dir, file_name)
                
                with open(json_path, 'w') as json_file:
                    json.dump({"avg_hsv": avg_hsv}, json_file, indent=4)
                image_counter += 1
                
                elapsed_time = time.time() - start_time  # Calculate elapsed time
                #print(f'Images processed: {image_counter}, Time elapsed: {elapsed_time:.2f} seconds', end='\r')
                
                if image_counter >= max_images:
                    print("\nReached the maximum limit of images.")
                    break
    
    print()  # Move to the next line after the loop completes
    #return avg_hsv_data

if __name__ == '__main__':
    #data_dirs = '/media/java/CSLICSNov24/cslics_data/2024_november_spawning, /media/java/CSLICSOct24/cslics_october_2024/20241023_spawning'
    data_dir = '/Data/cslics/2025_dec_spawn/'
    #save_data(data_dir,'/mnt/hpccs01/home/wardlewo/Data/cslics/tank_data')
    
    # Load data from JSON cache directories (note: absolute paths)
    print("Loading tank data from JSON cache...")
    Tank_G_9711 = load_data(['/home/java/hpc-home/Data/cslics/2025_dec_spawn/hsv_cache/grey/1421024219711'])
    print(f"  Tank_G_9711: {len(Tank_G_9711)} images")
    Tank_G_1440 = load_data(['/home/java/hpc-home/Data/cslics/2025_dec_spawn/hsv_cache/grey/1421024251440'])
    print(f"  Tank_G_1440: {len(Tank_G_1440)} images")
    Tank_W_0172 = load_data(['/home/java/hpc-home/Data/cslics/2025_dec_spawn/hsv_cache/white/1421024220172'])
    print(f"  Tank_W_0172: {len(Tank_W_0172)} images")
    Tank_W_2929 = load_data(['/home/java/hpc-home/Data/cslics/2025_dec_spawn/hsv_cache/white/1422724372929'])
    print(f"  Tank_W_2929: {len(Tank_W_2929)} images")
    print("Loading data complete\n")
    
    # Check if data was loaded
    if len(Tank_G_9711) == 0:
        print("ERROR: No data found for Tank_G_9711")
    if len(Tank_G_1440) == 0:
        print("ERROR: No data found for Tank_G_1440")
    if len(Tank_W_0172) == 0:
        print("ERROR: No data found for Tank_W_0172")
    if len(Tank_W_2929) == 0:
        print("ERROR: No data found for Tank_W_2929")
    
    for tank_name, tank_data in [("Tank_G_9711", Tank_G_9711), ("Tank_G_1440", Tank_G_1440), ("Tank_W_0172", Tank_W_0172), ("Tank_W_2929", Tank_W_2929)]:
        if len(tank_data) == 0:
            print(f"Skipping {tank_name} - no data found")
            continue
            
        print(f"{tank_name} - Lowest Hue")
        print(min(tank_data, key=lambda x: tank_data[x][0]))
        print(f"{tank_name} - Highest Hue")
        print(max(tank_data, key=lambda x: tank_data[x][0]))
        
        print(f"{tank_name} - Lowest Saturation")
        print(min(tank_data, key=lambda x: tank_data[x][1]))
        print(f"{tank_name} - Highest Saturation")
        print(max(tank_data, key=lambda x: tank_data[x][1]))
        
        print(f"{tank_name} - Lowest Value")
        print(min(tank_data, key=lambda x: tank_data[x][2]))
        print(f"{tank_name} - Highest Value")
        print(max(tank_data, key=lambda x: tank_data[x][2]))


    legend = ['Tank_G_9711', 'Tank_G_1440', 'Tank_W_0172', 'Tank_W_2929']
    
    ## Create separate figures for hue, saturation, and value
    
    # Figure 1: Hue Values
    plt.figure(figsize=(12, 8))
    all_values = (
        [value[0] for value in Tank_G_9711.values()] +
        [value[0] for value in Tank_G_1440.values()] +
        [value[0] for value in Tank_W_0172.values()] +
        [value[0] for value in Tank_W_2929.values()]
    )
    bin_edges = np.histogram_bin_edges(all_values, bins=30)
    plt.hist([value[0] for value in Tank_G_9711.values()], bins=bin_edges, alpha=0.6, color='r', label=legend[0])
    plt.hist([value[0] for value in Tank_G_1440.values()], bins=bin_edges, alpha=0.6, color='b', label=legend[1])
    plt.hist([value[0] for value in Tank_W_0172.values()], bins=bin_edges, alpha=0.6, color='g', label=legend[2])
    plt.hist([value[0] for value in Tank_W_2929.values()], bins=bin_edges, alpha=0.6, color='orange', label=legend[3])
    plt.xlabel('Hue Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Hue Values - All Tanks', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Figure 2: Saturation Values
    plt.figure(figsize=(12, 8))
    all_values = (
        [value[1] for value in Tank_G_9711.values()] +
        [value[1] for value in Tank_G_1440.values()] +
        [value[1] for value in Tank_W_0172.values()] +
        [value[1] for value in Tank_W_2929.values()]
    )
    bin_edges = np.histogram_bin_edges(all_values, bins=30)
    plt.hist([value[1] for value in Tank_G_9711.values()], bins=bin_edges, alpha=0.6, color='r', label=legend[0])
    plt.hist([value[1] for value in Tank_G_1440.values()], bins=bin_edges, alpha=0.6, color='b', label=legend[1])
    plt.hist([value[1] for value in Tank_W_0172.values()], bins=bin_edges, alpha=0.6, color='g', label=legend[2])
    plt.hist([value[1] for value in Tank_W_2929.values()], bins=bin_edges, alpha=0.6, color='orange', label=legend[3])
    plt.xlabel('Saturation Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Saturation Values - All Tanks', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Figure 3: Value Values
    plt.figure(figsize=(12, 8))
    all_values = (
        [value[2] for value in Tank_G_9711.values()] +
        [value[2] for value in Tank_G_1440.values()] +
        [value[2] for value in Tank_W_0172.values()] +
        [value[2] for value in Tank_W_2929.values()]
    )
    bin_edges = np.histogram_bin_edges(all_values, bins=30)
    plt.hist([value[2] for value in Tank_G_9711.values()], bins=bin_edges, alpha=0.6, color='r', label=legend[0])
    plt.hist([value[2] for value in Tank_G_1440.values()], bins=bin_edges, alpha=0.6, color='b', label=legend[1])
    plt.hist([value[2] for value in Tank_W_0172.values()], bins=bin_edges, alpha=0.6, color='g', label=legend[2])
    plt.hist([value[2] for value in Tank_W_2929.values()], bins=bin_edges, alpha=0.6, color='orange', label=legend[3])
    plt.xlabel('Value', fontsize=14)
    plt.ylabel('Frequency', fontsize=14)
    plt.title('Histogram of Value Values - All Tanks', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.show()

