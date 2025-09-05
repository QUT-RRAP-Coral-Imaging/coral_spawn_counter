#! /usr/bin/env python3

"""
use the results from SubSurface_detector and Surface detector pixkle files and plot them
assume that you have run the Surface and SubSurface Detectors on the relevant data already...
"""
#test
t = 1

import os
import cv2 as cv
import numpy as np
import glob
import random as rng
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import seaborn.objects as so
import pickle
import pandas as pd
from datetime import datetime
import time


from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.read_manual_counts import read_manual_counts
# from coral_spawn_counter.Detector_helper import convert_to_decimal_days

from coral_spawn_counter.Surface_Detector import Surface_Detector
from coral_spawn_counter.SubSurface_Detector import SubSurface_Detector


# Consts (more below as well)
window_size = 20 # for rolling means, etc

# TODO: replace this process/solve for nimage_to_tank_surface via initial counts? Should compare

# estimated tank surface area
rad_tank = 100.0/2 # cm^2 # actually measured the tanks this time
area_tank = np.pi * rad_tank**2 
# NOTE: cslics surface area counts differ for different cslics!!
# area_cslics = 2.3**2*(3/4) # cm^2 for cslics03 @ 15cm distance - had micro1 lens
# area_cslics = 2.35**2*(3/4) # cm2 for cslics01 @ 15.5 cm distance with micro2 lens
area_cslics = 1.2**2*(3/4) # cm^2 prboably closer to this @ 10cm distance, cslics04
nimage_to_tank_surface = area_tank / area_cslics
capture_time = []
n = 1 # how many std deviations to show
mpercent = 0.1 # range for manual counts

# File locations
root_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04'
img_dir = os.path.join(root_dir, 'images_jpg')
save_dir = os.path.join(root_dir, 'combined_detections')
manual_counts_file = os.path.join(root_dir, 'metadata/20221113_ManualCounts_AMaggieTenuis_Tank4-Sheet1.csv')

# img_dir = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04/images_jpg"
# save_dir = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04/combined_detections"
# manual_counts_file = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04/metadata/20221113_ManualCounts_AMaggieTenuis_Tank4-Sheet1.csv"
# root_dir = "/home/java/Java/data/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04"
object_names_file = 'metadata/obj.names'
subsurface_det_file = 'subsurface_det_testing.pkl'
surface_pkl_file = 'detection_results1.pkl'
subsurface_det_path = os.path.join(save_dir, subsurface_det_file)
save_plot_dir = os.path.join(save_dir, 'plots')
save_img_dir = os.path.join(save_dir, 'images')

# File setup
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
os.makedirs(save_dir, exist_ok=True)
os.makedirs(save_plot_dir, exist_ok=True)
os.makedirs(save_img_dir, exist_ok=True)

# load classes
with open(os.path.join(root_dir, 'metadata','obj.names'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]
        
##################################### surface counts
# load results
with open(os.path.join(root_dir, surface_pkl_file), 'rb') as f:
    results = pickle.load(f)
# get counts as arrays:
print('getting counts from Surface')
count_eggs = []
count_first = []
count_two = []
count_four = []
count_adv = []
count_dmg = []
capture_time_str = []

for res in results:
    # create a list of strings of all the detections for the given image
    counted_classes = [det[6] for det in res.detections]

    # do list comprehensions on counted_classes  # TODO consider putting this as a function into CoralImage/detections
    # could I replace this with iterating over the classes dictionary?
    count_eggs.append(float(counted_classes.count('Egg')))
    count_first.append(counted_classes.count('FirstCleavage')) # TODO fix class names to be all one continuous string (no spaces)
    count_two.append(counted_classes.count('TwoCell'))
    count_four.append(counted_classes.count('FourEightCell'))
    count_adv.append(counted_classes.count('Advanced'))
    count_dmg.append(counted_classes.count('Damaged'))
    capture_time_str.append(res.metadata['capture_time'])

# parse capture_time into datetime objects so we can sort them
surface_capture_times = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time_str]

surface_counts = [count_eggs[i] + count_first[i] + count_two[i] + count_four[i] + count_adv[i] for i in range(len(count_eggs))]
# apply rolling means

def plot_surface_counts(surface_capture_times, count_eggs, count_first, count_two, count_four,
                        count_adv, count_dmg, surface_counts):
    plotdatadict = {'capture times': surface_capture_times,
                    'eggs': count_eggs,
                    'first': count_first,
                    'two': count_two,
                    'four': count_four,
                    'adv': count_adv,
                    'dmg': count_dmg,
                    'total': surface_counts}

    df = pd.DataFrame(plotdatadict)

    count_eggs_mean = df['eggs'].rolling(window_size).mean()
    count_eggs_std = df['eggs'].rolling(window_size).std()

    count_first_mean = df['first'].rolling(window_size).mean()
    count_first_std = df['first'].rolling(window_size).std()

    count_two_mean = df['two'].rolling(window_size).mean()
    count_two_std = df['two'].rolling(window_size).std()

    count_four_mean = df['four'].rolling(window_size).mean()
    count_four_std = df['four'].rolling(window_size).std()

    count_adv_mean = df['adv'].rolling(window_size).mean()
    count_adv_std = df['adv'].rolling(window_size).std()

    count_dmg_mean = df['dmg'].rolling(window_size).mean()
    count_dmg_std = df['dmg'].rolling(window_size).std()

    count_total_mean = df['total'].rolling(window_size).mean()
    count_total_std = df['total'].rolling(window_size).std()

    return count_total_mean, count_total_std

count_total_mean, count_total_std = plot_surface_counts(surface_capture_times, count_eggs, count_first, count_two, count_four,
                        count_adv, count_dmg, surface_counts)

# sum everything
# countperimage_total = count_eggs_mean + count_first_mean + count_two_mean + count_four_mean + count_adv # not counting damaged
surface_decimal_days = convert_to_decimal_days(surface_capture_times)
counttank_total = count_total_mean * nimage_to_tank_surface

#######################################################################

# Subsurface load pixle data
# load pickle file for blobs_list and blobs_count

with open(subsurface_det_path, 'rb') as f:
    save_data = pickle.load(f)
    
blobs_list = save_data['blobs_list']
blobs_count = save_data['blobs_count']
image_index = save_data['image_index']
capture_time = save_data['capture_time']

# convert blobs_count into actual count, not interior list of indices
image_count = [len(blobs_index) for blobs_index in blobs_count]
image_count = np.array(image_count)

capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time]
decimal_days = convert_to_decimal_days(capture_time_dt)

########################################################
# read manual counts file

dt, mc, tw = read_manual_counts(manual_counts_file)
zero_time = capture_time_dt[0]
manual_decimal_days = convert_to_decimal_days(dt, zero_time)

########################################################

# Consts

# counts per image to density counts: need volume:
# calculated by hand to be approximately 0.1 mL 
# 2.23 cm x 1.675 cm x 0.267 cm
image_volume = 0.10 # mL
# density_count = [c / image_volume for c in count]
density_count = image_count * image_volume
# overall tank count: 
tank_volume = 500 * 1000 # 500 L * 1000 mL/L
tank_count = density_count * tank_volume

mpl.use('Agg')
print("Above line just for Java while testing")

##############################################################################

# show averages to apply rolling means
plotdatadict = {
    'index': image_index,
    'capture_time_days': decimal_days,
    'image_count': image_count,
    'density_count': density_count,
    'tank_count': tank_count
}
df = pd.DataFrame(plotdatadict)


image_count_mean = df['image_count'].rolling(window_size).mean()
image_count_std = df['image_count'].rolling(window_size).std()

density_count_mean = df['density_count'].rolling(window_size).mean()
density_count_std = df['density_count'].rolling(window_size).std()

tank_count_mean = df['tank_count'].rolling(window_size).mean()
tank_count_std = df['tank_count'].rolling(window_size).std()


# TODO showcase counts over time?
sns.set_theme(style='whitegrid')


fig1, ax1 = plt.subplots()
plt.plot(decimal_days, image_count_mean, label='count')
plt.fill_between(decimal_days, 
                 image_count_mean - n*image_count_std,
                 image_count_mean + n*image_count_std,
                 alpha=0.2)
plt.xlabel('days since stocking')
plt.ylabel('image count')
plt.title('Subsurface Counts vs Image Index - Prelim')
plt.savefig(os.path.join(save_plot_dir, 'subsubfacecounts.png'))


fig2, ax2 = plt.subplots()
plt.plot(decimal_days, density_count_mean, label='density [count/mL]')
plt.fill_between(decimal_days, 
                 density_count_mean - n*density_count_std,
                 density_count_mean + n*density_count_std,
                 alpha=0.2)
plt.xlabel('days since stocking')
plt.ylabel('density')
plt.title('Subsurface Density count/mL vs Image Index - Prelim')
plt.savefig(os.path.join(save_plot_dir, 'subsubface_densitycounts.png'))


def wholistic_tank_count_n_plot(surface_decimal_days, counttank_total, nimage_to_tank_surface, count_total_std, n,
                            decimal_days, tank_count_mean, tank_count_std,
                            manual_decimal_days, mc, mpercent,
                            save_plot_dir):
    # wholistic tank count
    fig3, ax3 = plt.subplots()

    # surface counts
    plt.plot(surface_decimal_days, counttank_total, label='surface count', color='orange')
    plt.fill_between(surface_decimal_days,
                    counttank_total - n * nimage_to_tank_surface * count_total_std,
                    counttank_total + n * nimage_to_tank_surface * count_total_std,
                    alpha=0.2,
                    color='orange')

    # subsurface counts
    plt.plot(decimal_days, tank_count_mean, label='subsurface count')
    plt.fill_between(decimal_days, 
                    tank_count_mean - n*tank_count_std,
                    tank_count_mean + n*tank_count_std,
                    alpha=0.2)

    # manual counts
    plt.plot(manual_decimal_days, mc, label='manual count', color='green', marker='o')
    plt.fill_between(manual_decimal_days, 
                    mc - mpercent * mc,
                    mc + mpercent * mc,
                    alpha=0.1,
                    color='green')
    
    #labeling
    plt.xlabel('days since stocking')
    plt.ylabel('tank count')
    plt.title('Overall tank count vs Time')
    plt.legend()
    plt.savefig(os.path.join(save_plot_dir, 'subsubface_tankcounts.png'))

wholistic_tank_count_n_plot(surface_decimal_days, counttank_total, nimage_to_tank_surface, count_total_std, n,
                            decimal_days, tank_count_mean, tank_count_std,
                            manual_decimal_days, mc, mpercent,
                            save_plot_dir)

print('results ploted')


# TODO need to ascertain the validity of the blobs - not all edges are ideal - chat with Andrew
# TODO save blob counts to txt file for reading/later usage?
# TODO write to xml file for uploading blob annotations    
import code
code.interact(local=dict(globals(), **locals()))