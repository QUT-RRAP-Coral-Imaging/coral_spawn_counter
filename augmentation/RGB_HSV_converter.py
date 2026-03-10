#!/usr/bin/env python3++
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# Specify the directories for grey and white images
grey_dirs = ["Data/cslics/2025_dec_spawn/1421024219711", "Data/cslics/2025_dec_spawn/1421024251440"]
white_dirs = ["Data/cslics/2025_dec_spawn/1421024220172", "Data/cslics/2025_dec_spawn/1422724372929"]

# Specify output directory for JSON files
output_dir = "Data/cslics/2025_dec_spawn/hsv_cache"

# Number of parallel processes
NUM_PROCESSES = 8

def channel_shift_stitch(image_name: str, channel: str, intensity: int):
    """
    Create a stitched image with the same image shifted by negative intensity, original, and positive intensity 
    for a specified channel (H, S, or V) in HSV color space.
    
    Parameters:
        image_name (str): Path to the input image.
        channel (str): The channel to modify ('H', 'S', or 'V').
        intensity (int): The intensity of the shift.
    
    Returns:
        stitched_image (np.ndarray): The stitched image with applied channel shifts.
    """
    # Map the channel to index (0: H, 1: S, 2: V)
    channel_map = {'H': 0, 'S': 1, 'V': 2}
    if channel not in channel_map:
        raise ValueError("Channel must be 'H', 'S', or 'V'.")
    
    channel_index = channel_map[channel]
    
    # Read the image and convert to HSV
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"Image '{image_name}' not found.")
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Function to apply channel shift
    def shift_channel(hsv_img, channel_idx, shift_val):
        shifted_img = hsv_img.copy()
        shifted_img[:, :, channel_idx] = np.clip(shifted_img[:, :, channel_idx] + shift_val, 0, 255)
        return shifted_img
    
    # Generate the three versions of the image
    left_image = shift_channel(hsv_image, channel_index, -intensity)
    right_image = shift_channel(hsv_image, channel_index, intensity)
    
    # Convert the shifted images back to BGR
    left_image_bgr = cv2.cvtColor(left_image, cv2.COLOR_HSV2BGR)
    center_image_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    right_image_bgr = cv2.cvtColor(right_image, cv2.COLOR_HSV2BGR)
    
    # Stitch the images together
    stitched_image = np.hstack((left_image_bgr, center_image_bgr, right_image_bgr))
    
    return stitched_image

def multi_channel_shift_stitch(image_name: str, h_shift: int, s_shift: int, v_shift: int):
    """
    Create a stitched image with the same image shifted by specified intensities for H, S, and V channels in HSV color space.
    
    Parameters:
        image_name (str): Path to the input image.
        h_shift (int): The intensity of the shift for the H channel.
        s_shift (int): The intensity of the shift for the S channel.
        v_shift (int): The intensity of the shift for the V channel.
    
    Returns:
        stitched_image (np.ndarray): The stitched image with applied channel shifts.
    """
    # Read the image and convert to HSV
    image = cv2.imread(image_name)
    if image is None:
        raise FileNotFoundError(f"Image '{image_name}' not found.")
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    # Function to apply channel shift
    def shift_channel(hsv_img, h_shift, s_shift, v_shift):
        shifted_img = hsv_img.copy()
        shifted_img[:, :, 0] = np.clip(shifted_img[:, :, 0] + h_shift, 0, 255)
        shifted_img[:, :, 1] = np.clip(shifted_img[:, :, 1] + s_shift, 0, 255)
        shifted_img[:, :, 2] = np.clip(shifted_img[:, :, 2] + v_shift, 0, 255)
        return shifted_img
    
    # Generate the three versions of the image
    left_image = shift_channel(hsv_image, -h_shift, -s_shift, -v_shift)
    right_image = shift_channel(hsv_image, h_shift, s_shift, v_shift)
    
    # Convert the shifted images back to BGR
    left_image_bgr = cv2.cvtColor(left_image, cv2.COLOR_HSV2RGB)
    center_image_bgr = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    right_image_bgr = cv2.cvtColor(right_image, cv2.COLOR_HSV2RGB)
    
    # Stitch the images together
    stitched_image = np.hstack((left_image_bgr, center_image_bgr, right_image_bgr))
    
    return stitched_image

class HSVInfo:
    """
    Class to store the name of an image and the value of a channel.
    """
    def __init__(self, name, value):
        self.name = name
        self.value = value

    def find_min_values(dictionary):
        """
        Find the image and the value with the lowest value in the dictionary for each channel.
        Returns:
            hue_min (HSVInfo): lowest hue value.
            saturation_min (HSVInfo): lowest saturation value.
            value_min (HSVInfo):  lowest value value.
        """
        hue_min        = HSVInfo(min(dictionary, key=lambda x: dictionary[x][0]), dictionary[min(dictionary, key=lambda x: dictionary[x][0])][0])
        saturation_min = HSVInfo(min(dictionary, key=lambda x: dictionary[x][1]), dictionary[min(dictionary, key=lambda x: dictionary[x][1])][1])
        value_min      = HSVInfo(min(dictionary, key=lambda x: dictionary[x][2]), dictionary[min(dictionary, key=lambda x: dictionary[x][2])][2])
    
        return hue_min, saturation_min, value_min
    
    def find_max_values(dictionary):
        """
        Find the image and the value with the peak value in the dictionary for each channel.
        Returns:
            hue_max (HSVInfo): peak hue value.
            saturation_max (HSVInfo): peak saturation value.
            value_max (HSVInfo):  peak value value.
        """
        hue_max        = HSVInfo(max(dictionary, key=lambda x: dictionary[x][0]), dictionary[(max(dictionary, key=lambda x: dictionary[x][0]))][0])
        saturation_max = HSVInfo(max(dictionary, key=lambda x: dictionary[x][1]), dictionary[(max(dictionary, key=lambda x: dictionary[x][1]))][1])
        value_max      = HSVInfo(max(dictionary, key=lambda x: dictionary[x][2]), dictionary[(max(dictionary, key=lambda x: dictionary[x][2]))][2])

        return hue_max, saturation_max, value_max
    
    def calculate_average_values(dictionary):
        """
        Calculate the average of the average values of the channels
        Rounded to 3 decimal places
        Returns:
            hue_avg (float): The average of the hue values.
            saturation_avg (float): The average of the saturation values.
            value_avg (float): The average of the value values.
        """
        hue_avg = round(sum(value[0] for value in dictionary.values()) / len(dictionary),3)
        saturation_avg = round(sum(value[1] for value in dictionary.values()) / len(dictionary),3)
        value_avg = round(sum(value[2] for value in dictionary.values()) / len(dictionary),3)
        return hue_avg, saturation_avg, value_avg
    def calculate_std_deviation(dictionary):
        """
        Calculate the standard deviation of the values in the dictionary for each channel.
        
        Returns:
            hue_std (float): The standard deviation of the hue values.
            saturation_std (float): The standard deviation of the saturation values.
            value_std (float): The standard deviation of the value values.
        """
        hue_std = np.std([value[0] for value in dictionary.values()])
        saturation_std = np.std([value[1] for value in dictionary.values()])
        value_std = np.std([value[2] for value in dictionary.values()])
        
        return hue_std, saturation_std, value_std


def process_single_image(args):
    """Process a single image and return its path and average HSV values."""
    image_path, json_path = args
    
    # Check if JSON already exists
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                return image_path, data['avg_hsv'], False  # False = loaded from cache
        except:
            pass  # If JSON is corrupted, recalculate
    
    # Process the image
    img = cv2.imread(image_path)
    if img is not None:
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        avg_hsv = tuple(cv2.mean(hsv_img)[:3])  # Get H, S, V averages
        
        # Save to JSON
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, 'w') as f:
            json.dump({"avg_hsv": avg_hsv}, f, indent=4)
        
        return image_path, avg_hsv, True  # True = newly processed
    
    return None, None, False


def load_or_process_images(directories, category_name, output_base_dir, num_processes=8):
    """
    Load images from directories, using cached JSON if available or processing in parallel.
    
    Args:
        directories: List of directory paths to search
        category_name: Name for this category (e.g., 'grey' or 'white')
        output_base_dir: Base directory for JSON cache files
        num_processes: Number of parallel processes to use
    
    Returns:
        Dictionary mapping image paths to average HSV tuples
    """
    # Collect all image files
    image_files = []
    for directory in directories:
        if not os.path.exists(directory):
            print(f"Warning: Directory not found: {directory}")
            continue
        
        for root, dirs, files in os.walk(directory):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                    image_path = os.path.join(root, file)
                    
                    # Create corresponding JSON path
                    rel_path = os.path.relpath(image_path, directory)
                    json_filename = os.path.splitext(rel_path)[0] + '.json'
                    json_path = os.path.join(output_base_dir, category_name, 
                                            os.path.basename(directory), json_filename)
                    
                    image_files.append((image_path, json_path))
    
    print(f"Found {len(image_files)} image files for {category_name}")
    
    if len(image_files) == 0:
        return {}
    
    # Process images in parallel
    results = {}
    processed_count = 0
    cached_count = 0
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        futures = {executor.submit(process_single_image, args): args for args in image_files}
        
        for i, future in enumerate(as_completed(futures), 1):
            image_path, avg_hsv, was_processed = future.result()
            
            if image_path is not None:
                results[image_path] = avg_hsv
                if was_processed:
                    processed_count += 1
                else:
                    cached_count += 1
            
            # Progress update
            elapsed = time.time() - start_time
            print(f'\r{category_name}: {i}/{len(image_files)} images '
                  f'(processed: {processed_count}, cached: {cached_count}, '
                  f'time: {elapsed:.1f}s)', end='', flush=True)
    
    print()  # New line after progress
    return results


if __name__ == '__main__':
    print("="*80)
    print("HSV Analysis with JSON Caching and Parallel Processing")
    print(f"Using {NUM_PROCESSES} parallel processes")
    print(f"Output directory: {output_dir}")
    print("="*80)
    print()
    
    # Load or process grey images
    print("Processing grey images...")
    grey_hsv_avg = load_or_process_images(grey_dirs, 'grey', output_dir, NUM_PROCESSES)
    
    # Load or process white images
    print("\nProcessing white images...")
    white_hsv_avg = load_or_process_images(white_dirs, 'white', output_dir, NUM_PROCESSES)
    
    # Check if we have images to process
    if len(grey_hsv_avg) == 0:
        print("ERROR: No grey images found! Please check the grey directory paths.")
        exit(1)
    if len(white_hsv_avg) == 0:
        print("ERROR: No white images found! Please check the white directory paths.")
        exit(1)
    
    print(f"\nTotal: {len(grey_hsv_avg)} grey images and {len(white_hsv_avg)} white images")
    print()

    # Find the image and the value with the peak and lowest value in the grey dictionary for each channel
    grey_hue_max, grey_saturation_max, grey_value_max          =   HSVInfo.find_max_values(grey_hsv_avg)
    white_hue_max, white_saturation_max, white_value_max =   HSVInfo.find_max_values(white_hsv_avg)
    grey_hue_min, grey_saturation_min, grey_value_min          =   HSVInfo.find_min_values(grey_hsv_avg)
    white_hue_min, white_saturation_min, white_value_min =   HSVInfo.find_min_values(white_hsv_avg)

    print()
    # Print the peak and lowest values for each channel with flipped bolding
    print("Channel     | Lowest Value | Peak Value    | % Difference | Absolute HSV Difference")
    print("------------|--------------|---------------|--------------|-------------------------")
    print(f"\033[1mGrey   H\033[0m | \033[1m{round(grey_hue_min.value, 3):<12}\033[0m | \033[1m{round(grey_hue_max.value, 3):<13}\033[0m | \033[1m{round(((grey_hue_max.value - grey_hue_min.value) / grey_hue_max.value) * 100, 3):<12}\033[0m | \033[1m{round(grey_hue_max.value - grey_hue_min.value, 3):<12}\033[0m")
    print(f"White H | {round(white_hue_min.value, 3):<12} | {round(white_hue_max.value, 3):<13} | {round(((white_hue_max.value - white_hue_min.value) / white_hue_max.value) * 100, 3):<12} | {round(white_hue_max.value - white_hue_min.value, 3):<12}")
    print(f"\033[1mGrey   S\033[0m | \033[1m{round(grey_saturation_min.value, 3):<12}\033[0m | \033[1m{round(grey_saturation_max.value, 3):<13}\033[0m | \033[1m{round(((grey_saturation_max.value - grey_saturation_min.value) / grey_saturation_max.value) * 100, 3):<12}\033[0m | \033[1m{round(grey_saturation_max.value - grey_saturation_min.value, 3):<12}\033[0m")
    print(f"White S | {round(white_saturation_min.value, 3):<12} | {round(white_saturation_max.value, 3):<13} | {round(((white_saturation_max.value - white_saturation_min.value) / white_saturation_max.value) * 100, 3):<12} | {round(white_saturation_max.value - white_saturation_min.value, 3):<12}")
    print(f"\033[1mGrey   V\033[0m | \033[1m{round(grey_value_min.value, 3):<12}\033[0m | \033[1m{round(grey_value_max.value, 3):<13}\033[0m | \033[1m{round(((grey_value_max.value - grey_value_min.value) / grey_value_max.value) * 100, 3):<12}\033[0m | \033[1m{round(grey_value_max.value - grey_value_min.value, 3):<12}\033[0m")
    print(f"White V | {round(white_value_min.value, 3):<12} | {round(white_value_max.value, 3):<13} | {round(((white_value_max.value - white_value_min.value) / white_value_max.value) * 100, 3):<12} | {round(white_value_max.value - white_value_min.value, 3):<12}")
    print()
    #print(grey_hue_max.name)
    #print(grey_value_min.name,grey_value_max.name)


    # Calculate the average values for grey and white dictionaries
    grey_hue_avg, grey_saturation_avg, grey_value_avg          = HSVInfo.calculate_average_values(grey_hsv_avg)
    white_hue_avg, white_saturation_avg, white_value_avg = HSVInfo.calculate_average_values(white_hsv_avg)
    print()

    # Calculate the standard deviation values for grey and white dictionaries
    grey_hue_std, grey_saturation_std, grey_value_std = HSVInfo.calculate_std_deviation(grey_hsv_avg)
    white_hue_std, white_saturation_std, white_value_std = HSVInfo.calculate_std_deviation(white_hsv_avg)

    # Print the standard deviation values
    print("Channel     | White Avg      | White Std      | Grey Avg      | Grey Std      | % Difference | Absolute HSV Difference")
    print("------------|----------------|----------------|---------------|---------------|--------------|------------------------")
    print(f"    H       | {white_hue_avg:<14} | {round(white_hue_std,3):<14} | {grey_hue_avg:<13} | {round(grey_hue_std,3):<13} | {round(abs(grey_hue_avg - white_hue_avg) / white_hue_avg * 100, 2):<12} | {round(abs(grey_hue_avg - white_hue_avg), 2):<12}")
    print(f"    S       | {white_saturation_avg:<14} | {round(white_saturation_std,3):<14} | {(grey_saturation_avg):<13} | {round(grey_saturation_std,3):<13} | {round(abs(grey_saturation_avg - white_saturation_avg) / white_saturation_avg * 100, 2):<12} | {round(abs(grey_saturation_avg - white_saturation_avg), 2):<12}")
    print(f"    V       | {white_value_avg:<14} | {round(white_value_std,3):<14} | {grey_value_avg:<13} | {round(grey_value_std,3):<13} | {round(abs(grey_value_avg - white_value_avg) / white_value_avg * 100, 2):<12} | {round(abs(grey_value_avg - white_value_avg), 2):<12}")
    print()


    #testImage = "/media/java/cslics_ssd/2024_cslics_light_dark_banding/2024_october_spawning/100000009c23b5af/grey/2024-10-24_16-02-38_clean.jpg"
    #testImage = "/media/java/cslics_ssd/2024_cslics_light_dark_banding/2024_november_spawning/100000000029da9b/grey/2024-11-23_17-01-05_clean.jpg"
    testImage = "/media/java/cslics_ssd/2024_cslics_light_dark_banding/2024_october_spawning/100000009c23b5af/white/2024-10-26_03-00-18_clean.jpg"
    result = multi_channel_shift_stitch(testImage, grey_hue_std, grey_saturation_std, grey_value_std)
    cv2.namedWindow("HSV_Adjusted_Image", cv2.WINDOW_NORMAL) 
    cv2.resizeWindow("HSV_Adjusted_Image", 1920, 1080) 
    cv2.imshow("HSV_Adjusted_Image", result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

