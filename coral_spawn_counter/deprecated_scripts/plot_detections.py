#! /usr/bin/env python3
import os
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib as mpl
import seaborn.objects as so

from coral_spawn_counter.CoralImage import CoralImage

# root_dir = '/home/agkelpie/Code/cslics_ws/src/datasets/20221114_amtenuis_cslics01'
root_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04'
basename = os.path.basename(root_dir)

# TODO experimental pickle support for saving interactive matplotlib figures
# https://fredborg-braedstrup.dk/blog/2014/10/10/saving-mpl-figures-using-pickle/

# load variables from file
with open(os.path.join(root_dir, 'detection_results.pkl'), 'rb') as f:
    results = pickle.load(f)

with open(os.path.join(root_dir, 'metadata','obj.names'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# TODO put this into specific file, similar to agklepie project
# define class-specific colours
sns_colors = sns.color_palette()
orange = sns_colors[3] # four-eight cell stage
blue = sns_colors[1] # first cleavage
purple = sns_colors[2] # two-cell stage
yellow = sns_colors[4] # advanced
brown = sns_colors[5] # damaged
green = sns_colors[0] # egg
class_colours = {classes[0]: orange,
                 classes[1]: blue,
                 classes[2]: purple,
                 classes[3]: yellow,
                 classes[4]: brown,
                 classes[5]: green}

# get counts as arrays:
print('getting counts from detections')
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
capture_times = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time_str]

# apply rolling means
plotdatadict = {'capture times': capture_times,
                'eggs': count_eggs,
                'first': count_first,
                'two': count_two,
                'four': count_four,
                'adv': count_adv,
                'dmg': count_dmg}
df = pd.DataFrame(plotdatadict)
# plotdata = pd.Series(count_eggs)

window_size = 20
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

# TODO fert ratio is just first cleavage to eggs, or everything else to eggs?
countperimage_total = count_eggs_mean + count_first_mean + count_two_mean + count_four_mean + count_adv # not counting damaged

fert_ratio = (count_first_mean + count_two_mean + count_four_mean + count_adv)/ countperimage_total

# ===========================================================================
sns.set_theme(style='whitegrid')

# plt.plot(capture_times, count_eggs, label='Egg')
n = 1
fig1, ax1 = plt.subplots()

plt.plot(capture_times, count_eggs_mean, label='Egg', color=class_colours['Egg'])
plt.fill_between(capture_times, count_eggs_mean - n*count_eggs_std, count_eggs_mean + n*count_eggs_std, color=class_colours['Egg'], alpha=0.2)

plt.plot(capture_times, count_first_mean, label='First Cleavage', color=class_colours['FirstCleavage'])
plt.fill_between(capture_times, count_first_mean - count_first_std*n, count_first_mean + n*count_first_std, color=class_colours['FirstCleavage'], alpha=0.2)

# plt.plot(capture_times, count_first, label='First Cleavage')
# plt.plot(capture_times, count_two, label='Two Cell Stage')
# plt.plot(capture_times, count_four, label='Four-Eight Cell Stage')
# plt.plot(capture_times, count_adv, label='Advanced')
# plt.plot(capture_times, count_dmg, label='Damaged')

# TODO calculate the standard deviation
# then up/low bounds by adding dev to mean
# plot this both

plt.xlabel('Date')
plt.ylabel('Count')
plt.title(f'{basename}: Cell Counts over Time')
plt.legend()
plt.savefig(os.path.join(root_dir,'detections','CellCountsEggsFirst.png'))


# ===========================================================================
fig1a, ax1a = plt.subplots()
plt.plot(capture_times, count_eggs_mean, label='Egg', color=class_colours['Egg'])
plt.fill_between(capture_times, count_eggs_mean - n*count_eggs_std, count_eggs_mean + n*count_eggs_std, color=class_colours['Egg'], alpha=0.2)

plt.plot(capture_times, count_first_mean, label='First Cleavage', color=class_colours['FirstCleavage'])
plt.fill_between(capture_times, count_first_mean - count_first_std*n, count_first_mean + n*count_first_std, color=class_colours['FirstCleavage'], alpha=0.2)

plt.plot(capture_times, count_two_mean, label='Two-Cell Stage', color=class_colours['TwoCell'])
plt.fill_between(capture_times, count_two_mean - count_two_std*n, count_two_mean + n*count_two_std, color=class_colours['TwoCell'], alpha=0.2)

plt.plot(capture_times, count_four_mean, label='Four-Eight-Cell Stage', color=class_colours['FourEightCell'])
plt.fill_between(capture_times, count_four_mean - count_four_std*n, count_four_mean + n*count_four_std, color=class_colours['FourEightCell'], alpha=0.2)

plt.plot(capture_times, count_adv_mean, label='Advanced', color=class_colours['Advanced'])
plt.fill_between(capture_times, count_adv_mean - count_adv_std*n, count_adv_mean + n*count_adv_std, color=class_colours['Advanced'], alpha=0.2)

plt.plot(capture_times, count_dmg_mean, label='Damaged', color=class_colours['Damaged'])
plt.fill_between(capture_times, count_dmg_mean - count_dmg_std*n, count_dmg_mean + n*count_dmg_std, color=class_colours['Damaged'], alpha=0.2)

plt.xlabel('Date')
plt.ylabel('Count')
plt.title(f'{basename}: Cell Counts over Time')
plt.legend()
plt.savefig(os.path.join(root_dir,'detections','CellCountsAll.png'))


# ===========================================================================
fig2, ax2 = plt.subplots()
sns.set_theme(style='whitegrid')
plt.plot(capture_times, fert_ratio, label='Fertilisation Ratio')
plt.xlabel('Date')
plt.ylabel('fert ratio')
plt.title(f'{basename}: Fertilisation Ratio over Time')
plt.legend()
plt.savefig(os.path.join(root_dir, 'detections','FertRatio.png'))


# ===========================================================================
fig3, ax3 = plt.subplots()
sns.set_theme(style='darkgrid')
plt.plot(capture_times, countperimage_total, label='Total Count/Image')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title(f'{basename}: Total Coral Spawn Count/Image over Time')
plt.legend()
plt.savefig(os.path.join(root_dir, 'detections', 'TotalCoralSpawnCountPerImage.png'))

# # plot time deltas:
# from datetime import timedelta
# # time_delta = [cap[i+1] - cap[i] for i, cap in enumerate(capture_times) if i < len(capture_times)]
# time_delta = []
# for i in range(len(capture_times) - 1):
#     time_delta.append(capture_times[i+1] - capture_times[i])

# time_delta_sec = [td.seconds for td in time_delta]

# plt.plot(time_delta_sec, label='time between image capture times', marker='o')
# plt.title('Capture Time Deltas (seconds)')
# plt.xlabel('index')
# plt.ylabel('seconds')
# plt.savefig(os.path.join(root_dir, 'detections','CaptureTimeDeltas100.png'))

# TODO actual surface counts
# estimated tank surface area
rad_tank = 100.0/2 # cm^2 # actually measured the tanks this time
area_tank = np.pi * rad_tank**2
# note: cslics surface area counts differ for different cslics!!
# area_cslics = 2.3**2*(3/4) # cm^2 for cslics03 @ 15cm distance - had micro1 lens
# area_cslics = 2.35**2*(3/4) # cm2 for cslics01 @ 15.5 cm distance with micro2 lens
area_cslics = 1.2**2*(3/4) # cm^2 prboably closer to this @ 10cm distance, cslics04
nimage_to_tank_surface = area_tank / area_cslics

counttank_total = countperimage_total * nimage_to_tank_surface

# ===========================================================================
fig4, ax4 = plt.subplots()
sns.set_theme(style='darkgrid')
plt.plot(capture_times, counttank_total, label='Total Count')
plt.xlabel('Date')
plt.ylabel('Count')
plt.title(f'{basename}: Total Coral Spawn Count for Tank Surface over Time')
plt.legend()
plt.savefig(os.path.join(root_dir, 'detections', 'TotalCoralSpawnCountPerTankSurface.png'))

plt.show()

import code
code.interact(local=dict(globals(), **locals()))