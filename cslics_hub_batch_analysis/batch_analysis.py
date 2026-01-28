#! /usr/bin/env python3

import os

# Dorian Tsai
# 2025-11-26
# batch analysis script for processing .csv files from CSLICS Hub export
# and comparing/plotting to manual counts

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns   
from scipy import stats
import numpy as np
import glob
from matplotlib.dates import HourLocator, DateFormatter

# location of batch export directory 
# data_dir = '/home/dtsai/Data/cslics_datasets/cslics_batch_export_2025/2025nov/lar12_1421024219729_aken_export_2025-11-22_16-00'
data_dir = '/home/dtsai/Data/cslics_datasets/cslics_batch_export_2025/2025nov/lar11_1421124265569_aken_keppel_export_2025-11-22_16-00'

# data file:
data_filename = 'sample_data.csv'

# read sample data csv file
data_file = os.path.join(data_dir, data_filename)
data_cslics = pd.read_csv(data_file)

# extract sample times
time_cslics_datetime = pd.to_datetime(data_cslics.iloc[:,0])

# extract tank estimates
tank_est = data_cslics.iloc[:,5]

# choose the range of data to plot and analyse
# choose the range of data to plot and analyse
# Set time limits for x-axis (modify these as needed)
# times for LAR11 run
time_min = '2025-11-21 13:00:00'  # Example start time
time_max = '2025-11-22 18:00:00'  # Example end time

# times for LAR12 run
# time_min = '2025-11-19 19:00:00'  # Example start time
# time_max = '2025-11-22 18:00:00'  # Example end time

# Convert to datetime objects
time_min_dt = pd.to_datetime(time_min)
time_max_dt = pd.to_datetime(time_max)

#######################
# location of manual counts file
manual_counts_file = 'CSLICS_harvesting_test(manual_counts_LAR11).csv'
# manual_counts_file = 'CSLICS_harvesting_test(manual_counts_LAR12).csv'

# read from manual counts file
data_manual = pd.read_csv(os.path.join(data_dir, manual_counts_file))
time_manual_datetime = pd.to_datetime(data_manual['Date'] + ' ' + data_manual['Time'],format='%d/%m/%Y %H:%M')

tank_manual_counts = data_manual['Manual']
std_dev_mc_counts = data_manual['count1'] 

num_rows = data_manual.shape[0]
std_dev_mc_counts = np.zeros(num_rows)
for i in range(1,num_rows):
    std_dev_mc_counts[i] = np.std(data_manual.iloc[i,4:9])

scale_std_dev = 1.0/50.0*1000.0*440.0  # scale to tank volume and from liters to milliliters
std_dev_mc = std_dev_mc_counts * scale_std_dev

#######################

# try scaling the CSLICS estimates to the manual counts
# scale the last manual count to the average of the last cslics count (last 10 points)
# last_avg = tank_est[-10:].mean()
# print('last avg = ', last_avg)
# scale_factor = tank_manual_counts.iloc[1] / last_avg
# print('scale factor = ', scale_factor)
scale_factor = 0.7
tank_est_scaled = tank_est * scale_factor


#######################

# plot coral counts vs time
plt.figure(figsize=(12, 8))
plt.plot(time_cslics_datetime, tank_est, marker='o', linewidth=2, markersize=4, label='CSLICS calibrated based on stocking density', alpha=0.5)
plt.plot(time_cslics_datetime, tank_est_scaled, marker='^', linewidth=2, markersize=4, label='CSLICS scaled to pre-harvest count', color='purple')
# Plot manual counts with different colors based on Notes column
pre_mask = data_manual['Notes'] == 'pre'
post_mask = data_manual['Notes'] == 'post'
plt.plot(time_manual_datetime[pre_mask], tank_manual_counts[pre_mask], 'o', 
         color='orange', markersize=12, linewidth=2, label='Manual Count (pre-harvest)')
plt.plot(time_manual_datetime[post_mask], tank_manual_counts[post_mask], 'o', 
         color='red', markersize=12, linewidth=2,label='Manual Count (post-harvest)')
plt.errorbar(time_manual_datetime[pre_mask], tank_manual_counts[pre_mask], 
             yerr=std_dev_mc[pre_mask], fmt='.', 
             color='orange', markersize=12, linewidth=2, capsize=5,
             label='Manual Count Error Bars (1 std) (pre-harvest)')

plt.errorbar(time_manual_datetime[post_mask], tank_manual_counts[post_mask], 
             yerr=std_dev_mc[post_mask], fmt='.', 
             color='red', markersize=12, linewidth=2, capsize=5,
             label='Manual Count Error Bars (1 std) (post-harvest)')
plt.xlabel('Time')
plt.ylabel('Coral Count')
plt.title('LAR 11 Tank Estimate vs Time During Simulated Harvest')
plt.grid(True, alpha=0.3)

plt.legend(loc='upper right', frameon=True, shadow=True, fontsize=10)

# Set x-axis ticks every X hours
ax = plt.gca()
ax.xaxis.set_major_locator(HourLocator(interval=6))
ax.xaxis.set_major_formatter(DateFormatter('%m-%d %H:%M'))

# Set x-axis limits
plt.xlim(time_min_dt, time_max_dt)
plt.ylim(0, 700000)
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(os.path.join(data_dir, 'tank_estimate_vs_time.png'), dpi=300)
plt.show()

# plot some statistics - NOTE: need to determine variance during specified time period when tank counts are constant
# print(f"Data shape: {data_cslics.shape}")
# print(f"Mean coral count: {data_cslics['coral'].mean():.2f}")

# DEBUG
import code
code.interact(local=dict(globals(), **locals()))