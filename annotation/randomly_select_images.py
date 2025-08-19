#! /usr/bin/env python3

# Author: Dorian Tsai
# Date: 2025 Jan 30

# quick script to grab a desired number of target random images from source to target directory
# read from a source folder, then grab a random assortment of files and copy those into a different target folder
# the intended behaviour is to glob all the photos in the sub-directories of the specified cslics uuid folder, and then amalgamate them into a single folder containing X images

# NOTE images that are unsuitable for training can be manually removed. The number of subtracted images can then be determined manually and then editted
# and then use the append option, and iterate until the original intended target number of images is reached
import os
import shutil
import random
# from glob import glob
from pathlib import Path
import sys


def copy_images(imgs_list, target_dir, num_subfolders=1):
    # copy images (list of path objects) to target_dir, split equally into subfolders
    if num_subfolders == 1:
        # Original behavior - copy to single directory
        for i, img_path in enumerate(imgs_list):
            print(f'img {i+1}/{len(imgs_list)}')
            shutil.copy(img_path, 
                        os.path.join(target_dir, img_path.name))
    else:
        # Split images equally into subfolders
        images_per_folder = len(imgs_list) // num_subfolders
        remainder = len(imgs_list) % num_subfolders
        
        # Create subfolders
        for i in range(num_subfolders):
            subfolder_path = os.path.join(target_dir, f'subfolder_{i+1:02d}')
            os.makedirs(subfolder_path, exist_ok=True)
        
        # Distribute images
        img_index = 0
        for folder_idx in range(num_subfolders):
            subfolder_path = os.path.join(target_dir, f'subfolder_{folder_idx+1:02d}')
            
            # Calculate how many images for this folder (distribute remainder across first folders)
            images_for_this_folder = images_per_folder + (1 if folder_idx < remainder else 0)
            
            for j in range(images_for_this_folder):
                img_path = imgs_list[img_index]
                print(f'img {img_index+1}/{len(imgs_list)} -> subfolder_{folder_idx+1:02d}')
                shutil.copy(img_path, os.path.join(subfolder_path, img_path.name))
                img_index += 1


def get_existing_image_names(exclude_dir):
    """
    Get a set of image filenames that already exist in the exclude directory.
    This helps avoid selecting images that are already in test/validation sets.
    """
    if not exclude_dir or not os.path.exists(exclude_dir):
        return set()
    
    print(f'Checking for existing images in exclude directory: {exclude_dir}')
    existing_images = set()
    
    # Get all image files in the exclude directory (including subdirectories)
    exclude_path = Path(exclude_dir)
    for img_path in exclude_path.rglob('*.jpg'):
        existing_images.add(img_path.name)
    
    print(f'Found {len(existing_images)} existing images to exclude')
    return existing_images


def filter_available_images(img_list, exclude_dir=None):
    """
    Filter the image list to exclude images that already exist in the exclude directory.
    Returns a filtered list of available images.
    """
    if not exclude_dir:
        return img_list
    
    existing_names = get_existing_image_names(exclude_dir)
    
    # Filter out images that already exist in the exclude directory
    available_images = [img for img in img_list if img.name not in existing_names]
    
    excluded_count = len(img_list) - len(available_images)
    print(f'Excluded {excluded_count} images that already exist in {exclude_dir}')
    print(f'Available images for selection: {len(available_images)}')
    
    return available_images


# source folder - currently, target based on cslics uuid folder
# source_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/100000000029da9b/image_test'
# source_dir = '/media/dtsai/CSLICSOct24/cslics_october_2024/20241023_spawning/100000001ab0438d'
# source_dir = '/media/dtsai/CSLICSOct24/cslics_october_2024/20241023_spawning/10000000f620da42'
source_dir = '/home/java/hpc-home/Data/cslics/2024_spawn_tanks_data/oct/100000001ab0438d'


# target folder
# target_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/100000000029da9b/output_test'
# target_dir = '/home/dtsai/Data/cslics_datasets/cslics_2024_october_subsurface_dataset/10000000f620da42/images'
target_dir = '/home/java/hpc-home/Data/cslics/2024_spawn_tanks_data/maeq/maeq_oct_438d/'

# exclude directory - images in this directory will not be selected
# Set to None or empty string to disable exclusion
exclude_dir = '/home/java/hpc-home/Data/cslics/2024_spawn_tanks_data/oct/100000001ab0438d/images'  # Example: '/path/to/test_set' or '/path/to/validation_set'

# target number of images
target_images = 700

# number of subfolders to split images into (set to 1 for original behavior)
num_subfolders = 3

# check to make sure assume target images is greater than number of images in the folder
print(f'Gathering list of images in all sub-directories of source directory: {source_dir}')
img_list = sorted(Path(source_dir).rglob('*_clean.jpg'))
n_img = len(img_list)

# Filter out images that already exist in exclude directory
available_img_list = filter_available_images(img_list, exclude_dir)

if target_images > len(available_img_list):
    print(f'Number of target images: {target_images}')
    print(f'Number of available images (after exclusions): {len(available_img_list)}')
    print(f'ERROR: number of target images is greater than number of available images')
    sys.exit(1)

# randomly sample from available images
print(f'randomly sample {target_images} images out of {len(available_img_list)} available images')
print(f'Percent sampled from available images: {target_images / len(available_img_list) * 100}%')
print(f'Images will be split into {num_subfolders} subfolders')
imgs_rng = random.sample(available_img_list, target_images)

# take those image names and then move them into a new folder:
# be sure to clear folder ahead of time - or ask user
if not os.path.exists(target_dir):
    print(f'Target_dir does not yet exist, making new directory')
    os.makedirs(target_dir)
    copy_images(imgs_rng, target_dir, num_subfolders)
else:
    # if directory exists, wait for user input
    i = 0
    print(f'Target_dir already exists.')
    while True and i < 10:
        user_input = input(' Do you want to (d) Delete and recreate the directory, (a) append to the directory---possibly overwriting files, or (e) exit the operation? ')
        if user_input == "d": 
            print('Removing existing folder, creating new target_dir, copying to new target_dir')
            shutil.rmtree(target_dir)
            os.makedirs(target_dir)
            copy_images(imgs_rng, target_dir, num_subfolders)
            break
        elif user_input == "a":
            print('Copying over existing target_dir')
            copy_images(imgs_rng, target_dir, num_subfolders)
            break
        elif user_input == "e":
            print('Exiting operation')
            sys.exit(1)

        else:
            print('Invalid user input. Please enter ''d'', ''a'', or ''e''')

        i+=1

    # yes: delete and overwrite
    # no: append
    # exit: stop





print('done')

# import code
# code.interact(local=dict(globals(), **locals()))
