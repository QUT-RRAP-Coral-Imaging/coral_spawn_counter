#! /usr/bin/env python3

# Dorian Tsai
# 2026/01/30
# Manual counts formatting for CSLICS Hub
# just scripting up the function to take the LAR tank counts based on LAR##

import pandas as pd
import numpy as np

manual_dir = '/home/dtsai/Data/cslics_datasets/cslics_batch_export_2025/2025dec'
manual_counts_file = 'Dec2025_CSLICS_Larval Cultures(Culture Density).csv'

mc = pd.read_csv(f'{manual_dir}/{manual_counts_file}')


# for each row, we want standard deviation
mc['Std'] = np.nanstd(mc[['subcount1', 'subcount2', 'subcount3', 'subcount4', 'subcount5']], axis=1)
# adjust std deviation based on method:
scale_5x50ml = 1.0 / 50.0 * 1000.0 * 440.0  # scale to tank volume and from liters to milliliters
scale_3x1000ml = 1.0 / 1000.0 * 1000.0 * 440.0  # scale to tank volume and from liters to milliliters
mc['Std_scaled'] = mc.apply(
    lambda row: row['Std'] * scale_5x50ml if row['Method'] == '5x50ml' 
    else row['Std'] * scale_3x1000ml if row['Method'] == '3x1000ml' 
    else row['Std'], 
    axis=1
)


# compute difference between manual count and CSLICS estimate
# Convert CSLICS column from string to float64
mc['Manual'] = pd.to_numeric(mc['Manual'], errors='coerce')
mc['CSLICS'] = pd.to_numeric(mc['CSLICS'], errors='coerce')
mc['diff'] = abs(mc['Manual'] - mc['CSLICS'])
mc['percent_diff'] = mc['diff'] / mc['Manual'] * 100.0

# for each tank, I want to print every relevant row (count) for the manual count, std, and CSLICS estimate
for tank, group in mc.groupby('Tank'):
    print(f'Tank: {tank}')
    for idx, row in group.iterrows():
        date = row['Date']
        time = row['Time']
        species = row['Species']
        manual_count = row['Manual']
        std_dev = row['Std_scaled']
        meth = row['Method']
        cslics_estimate = row['CSLICS']
        diff = row['diff']
        pdiff = row['percent_diff']
        print(f'Date: {date} Time: {time}\tSpecies: {species}\t'
              f'Manual: {manual_count:06.0f}\tStd Dev: {std_dev:05.0f}\t'
              f'CSLICS Estimate: {cslics_estimate:06.0f}\tDiff: {diff:06.0f}\tPercent Diff: {pdiff:02.2f}%')
    print('')
import code
code.interact(local=dict(globals(), **locals()))