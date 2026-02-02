#! /usr/bin/env python3

# script to plot all cslics sample_data.csv files in their respective folders & plots

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import glob

# location of batch export directories
data_dir = '/home/dtsai/Data/cslics_datasets/cslics_batch_export_2025/2025dec'

# grab all directories in data_dir that start with 'LAR'
batch_dirs = [d for d in glob.glob(os.path.join(data_dir, 'LAR*')) if os.path.isdir(d)]
print(f'Found batch directories:')
for b in batch_dirs:
    print(f' - {b}')

# name of sample data file
data_filename = 'sample_data.csv'

for bdir in batch_dirs:
    print(f'\nProcessing batch directory: {bdir}')
    # read sample data csv file
    df = pd.read_csv(os.path.join(bdir, data_filename))
    
    # extract sample times
    t_sample = pd.to_datetime(df["timestamp"])
    # extract tank estimates
    
    # TODO: adjust coral count to tank estimate using scale factor (recorded elsewhere)
    # for now, just read in the tank estimate column directly
    tank_est = df["Coral (Tank ~)"]
    
    # plot tank estimates over time
    plt.figure(figsize=(10, 6))
    plt.plot(t_sample, tank_est, marker='o', linestyle='-', color='b')
    # TODO extract CSLICS ID from metadata, and tank ID from folder name, add to title
    plt.title(f'CSLICS Tank Estimates Over Time\nBatch: {os.path.basename(bdir)}')
    plt.xlabel('Sample Time')
    plt.ylabel('Tank Estimate (Coral)')
    plt.grid(True)
    plt.tight_layout()
    # plt.show()  
    plt.savefig(os.path.join(bdir, 'tank_estimates_plot.png'))
    plt.close()

import code
code.interact(local=dict(globals(), **locals()))

