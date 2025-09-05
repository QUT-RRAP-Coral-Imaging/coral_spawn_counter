#! /usr/bin/env python3

"""
use the results from SubSurface_detector and Surface detector pixkle files and plot them
Can be  run with or without the Surface and SubSurface Detectors on the relevant data already 
ith subsurface detector being a blob counter.
"""
import os
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn.objects as so
import pickle
import pandas as pd
from datetime import datetime
import sys
import bisect
from sklearn.metrics import mean_squared_error
sys.path.insert(0, '')

from coral_spawn_counter.CoralImage import CoralImage
from coral_spawn_counter.read_manual_counts import read_manual_counts

from coral_spawn_counter.Surface_Detector import Surface_Detector
from coral_spawn_counter.SubSurface_Detector import SubSurface_Detector

## Varibles for running the script
Counts_avalible = True #if True, means pkl files avalible, else run the surface detectors
scale_detection_results = True #if True, scale the detection results to the tank size
dataset = '20231205_alor_tank4_cslics01' #name of the dataset

# File locations
save_plot_dir = '/home/java/Java/data/'+dataset
manual_counts_file = '/home/java/Java/data/cslics_ManualCounts/2023-12/C-SLIC culture density data sheet.xlsx'
sheet_name = 'Dec-A.lor Tank 4'
img_dir = '/home/java/Java/data/'+dataset+'/images'
object_names_file = '/home/java/Java/cslics/metadata/obj.names'
result_plot_name = 'tankcounts_with_scaling_10.png'
plot_title = 'Cslics06 '+sheet_name+' alor_aten_2000'
if Counts_avalible==True: #if false will have to set up the paths for the detectors
    subsurface_det_path = '/home/java/Java/data/'+dataset+'/detections_subsurface/subsurface_detections.pkl'  # path to subsurface detections
    surface_det_path = '/home/java/Java/data/'+dataset+'/detect_surface/surface_detections.pkl' # path to surface detections

## Constant Definitions
window_size = 100 # for rolling means, etc
n = 1 # how many std deviations to show
mpercent = 0.1 # range for manual counts
image_volume = 0.10 # mL
tank_volume = 500 * 1000 # 500 L * 1000 mL/L
image_span = 100 # how many images to averge out for the time history comparison
idx_surface_manual_count = 1
# estimated tank specs area
rad_tank = 100.0/2 # cm^2 # actually measured the tanks this time
area_tank = np.pi * rad_tank**2 
area_cslics = 1.2**2*(3/4) # cm^2 prboably closer to this @ 10cm distance, cslics04
volume_image = 35 # Ml # VERY MUCH AN APPROXIMATION - TODO FIGURE OUT THE MORE PRECISE METHOD
volume_tank = 500 * 1000 # 500 L = 500000 ml
if scale_detection_results==False:
    nimage_to_tank_surface = area_tank / area_cslics ### replaced latter with modifier based on manual counts
    nimage_to_tank_volume = volume_tank / volume_image # thus, how many cslics images will fill the whole volume of the tank
capture_time = []


if Counts_avalible==False:
    MAX_IMG = 10e10
    skip_img = 10
    subsurface_pkl_name = 'subsurface_detections.pkl'
    surface_pkl_name = 'surface_detections.pkl'
    save_dir_subsurface = '/home/java/Java/data/'+dataset+'/alor_atem_2000_subsurface_detections'
    save_dir_surface = '/home/java/Java/data/'+dataset+'/alor_atem_2000_surface_detections'
    meta_dir = '/home/java/Java/cslics' 
    weights = '/home/java/Java/ultralytics/runs/detect/train - aten_alor_2000/weights/best.pt'
    object_names_file = '/home/java/Java/cslics/metadata/obj.names'
    Coral_Detector = Surface_Detector(weights_file=weights, meta_dir = meta_dir, img_dir=img_dir, save_dir=save_dir_surface, 
                                      output_file=surface_pkl_name, max_img=MAX_IMG, skip_img=skip_img)
    Coral_Detector.run()
    Coral_Detector = SubSurface_Detector(meta_dir = meta_dir, img_dir = img_dir, save_dir=save_dir_subsurface, 
                                         detection_file=subsurface_pkl_name, max_img = MAX_IMG, skip_img=skip_img)
    Coral_Detector.run() 
    subsurface_det_path = os.path.join(save_dir_subsurface, subsurface_pkl_name) 
    surface_det_path = os.path.join(save_dir_surface, surface_pkl_name)

# File setup
img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
os.makedirs(save_plot_dir, exist_ok=True)

# load classes
with open(object_names_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

def convert_to_decimal_days(dates_list, time_zero=None):
    """from a list of datetime objects, convert to decimal days since time_zero"""
    if time_zero is None:
        time_zero = dates_list[0]  # Time zero is the first element date in the list
    else:
        time_zero = time_zero
        
    decimal_days_list = []

    for date in dates_list:
        time_difference = date - time_zero
        decimal_days = time_difference.total_seconds() / (60 * 60 * 24)
        decimal_days_list.append(decimal_days)

    return decimal_days_list

def read_scale_times(dt, file, sheet_name):
    """read the scale times from the excel file and return the index of the closest datetime object"""
    df = pd.read_excel(os.path.join(file), sheet_name=sheet_name, engine='openpyxl', header=2) 
    date_column = df['Date'].iloc[:]
    notes_column = df['Notes'].iloc[:]
    time_column = df['Time Collected'].iloc[:]

    combined_datetime = np.nan
    closest_index = None
    min_difference = None

    for index, note in enumerate(notes_column):
        if note == 'Time placed into water' and not pd.isnull(combined_datetime):
            break
        elif not pd.isnull(date_column[index]):
            combined_datetime = datetime.combine(date_column[index], time_column[index])

    for index, dt_item in enumerate(dt):
        difference = abs(combined_datetime - dt_item)
        if min_difference is None or difference < min_difference:
            min_difference = difference
            closest_index = index

    return closest_index, combined_datetime

def new_read_manual_counts(file, sheet_name):
    """read the manual counts from the excel file and return the datetime objects and counts"""
    df = pd.read_excel(os.path.join(file), sheet_name=sheet_name, engine='openpyxl', header=2) 
    count_coloum = df['Mean, last 2 digits rounded\nSD, last 2 digits rounded'].iloc[:]
    date_column = df['Date'].iloc[:]
    time_column = df['Time Collected'].iloc[:]

    combined_datetime_objects = []
    man_counts = []
    std = []
    for i, value in enumerate(count_coloum):
        if i >= 1 and i % 6 == 0: #would normally just be i>0
            combined_datetime_obj = datetime.combine(date_column[i], time_column[i])
        if i>6 and (i - 4) % 6 == 0 and not np.isnan(value): #would normally be i>4
            man_counts.append(value)
            combined_datetime_objects.append(combined_datetime_obj)
        if i>6 and (i - 5) % 6 == 0 and not np.isnan(value):
            std.append(value)
    return combined_datetime_objects, man_counts, std

    """create a time history got coral counts"""
    averaged_counts = []
    ts_of_averaged_counts = []
    stds_of_averaged_counts = []
    for i, manual_count in enumerate(manual_counts):
        time_of_man = manual_decimal_days[i]
        idx = bisect.bisect_right(coral_counts_decimal, time_of_man)
        lower_bound = int(max(0, idx - 0.5*image_span))
        upper_bound = int(min(len(scaled_coral_counts), idx + 0.5*image_span))
        range_of_cont = scaled_coral_counts[lower_bound:upper_bound]
        valid_values = range_of_cont[~np.isnan(range_of_cont)]
        
        if len(valid_values) > 0:
            averge_count = np.mean(valid_values)
            averaged_counts.append(averge_count)
            ts_of_averaged_counts.append(coral_counts_decimal[idx])
    stds_of_averaged_counts = np.std(averaged_counts)
    return averaged_counts, ts_of_averaged_counts, stds_of_averaged_counts

########################################################
# read manual counts file
########################################################
dt, mc, manual_std = new_read_manual_counts(manual_counts_file, sheet_name)
submersion_idx, submersion_time = read_scale_times(dt, manual_counts_file, 'CSLICS_'+sheet_name)
zero_time = dt[0]
plot_title = plot_title + ' ' + dt[0].strftime("%Y-%m-%d %H:%M:%S")
manual_decimal_days = convert_to_decimal_days(dt, zero_time)
submersion_time = convert_to_decimal_days([submersion_time], zero_time)[0]

#######################################################################
# Subsurface load pixle data
#######################################################################
with open(subsurface_det_path, 'rb') as f:
    save_data = pickle.load(f)
    
blobs_list = save_data['blobs_list']
blobs_count = save_data['blobs_count']
image_index = save_data['image_index']
capture_time = save_data['capture_time']

# convert blobs_count into actual count, not interior list of indices
subsurface_imge_count = [len(blobs_index) for blobs_index in blobs_count]
subsurface_imge_count = np.array(subsurface_imge_count)

#capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S_%f') for d in capture_time]
capture_time_dt = [datetime.strptime(d, '%Y%m%d_%H%M%S') for d in capture_time]
decimal_days = convert_to_decimal_days(capture_time_dt)

#subsurface counts given manual count
if scale_detection_results==True:
    manual_count = mc[submersion_idx+1]
    manual_scale_factor = manual_count / np.mean(subsurface_imge_count)
    volume_image = volume_tank / manual_scale_factor
    nimage_to_tank_volume = volume_tank / volume_image
    print(f'cslics spawn subsurface scale factor: {nimage_to_tank_volume}')

subsurface_image_count_total = subsurface_imge_count * nimage_to_tank_volume 

subsurface_mean = pd.Series(subsurface_image_count_total).rolling(window_size, center=True, min_periods=1).mean()
subsurface_std = pd.Series(subsurface_image_count_total).rolling(window_size, center=True, min_periods=1).std()

#######################################################################
# interpolate manual counts to frequency of subsurface counts, calcualte RMSE and correlation coefficient
idx_subsurface_manual_count_stop_time = bisect.bisect_right(capture_time_dt, dt[-1]) - 1
mc_interpolated = np.interp(decimal_days[:idx_subsurface_manual_count_stop_time], manual_decimal_days, mc)

rmse_not_scaled = np.sqrt(mean_squared_error(mc_interpolated, subsurface_imge_count[:idx_subsurface_manual_count_stop_time]))
correlation_coefficient_not_scaled = np.corrcoef(mc_interpolated, subsurface_imge_count[:idx_subsurface_manual_count_stop_time])[0, 1]
rmse_scaled = np.sqrt(mean_squared_error(mc_interpolated, subsurface_image_count_total[:idx_subsurface_manual_count_stop_time]))
correlation_coefficient_scaled = np.corrcoef(mc_interpolated, subsurface_image_count_total[:idx_subsurface_manual_count_stop_time])[0, 1]

print(f'After scaling subsurface: RMSE {rmse_scaled / nimage_to_tank_volume}, correlation coefficient {correlation_coefficient_scaled}')
print(f'Before scaling subsurface: RMSE {rmse_not_scaled}, correlation coefficient {correlation_coefficient_not_scaled}')

##################################### surface counts ########################################
def load_surface_counts(surface_det_path):
    #with open(os.path.join(root_dir, surface_pkl_file), 'rb') as f:
    with open(surface_det_path, 'rb') as f:
        results = pickle.load(f)
    # get counts as arrays:
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
    return surface_capture_times, surface_counts, count_eggs, count_first, count_two, count_four, count_adv, count_dmg

def get_surface_mean_n_std(surface_capture_times, count_eggs, count_first, count_two, count_four,
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

    count_eggs_mean = df['eggs'].rolling(window_size, center=True, min_periods=1).mean()
    count_eggs_std = df['eggs'].rolling(window_size, center=True, min_periods=1).std()

    count_first_mean = df['first'].rolling(window_size, center=True, min_periods=1).mean()
    count_first_std = df['first'].rolling(window_size, center=True, min_periods=1).std()

    count_two_mean = df['two'].rolling(window_size, center=True, min_periods=1).mean()
    count_two_std = df['two'].rolling(window_size, center=True, min_periods=1).std()

    count_four_mean = df['four'].rolling(window_size, center=True, min_periods=1).mean()
    count_four_std = df['four'].rolling(window_size, center=True, min_periods=1).std()

    count_adv_mean = df['adv'].rolling(window_size, center=True, min_periods=1).mean()
    count_adv_std = df['adv'].rolling(window_size, center=True, min_periods=1).std()

    count_dmg_mean = df['dmg'].rolling(window_size, center=True, min_periods=1).mean()
    count_dmg_std = df['dmg'].rolling(window_size, center=True, min_periods=1).std()

    count_total_mean = df['total'].rolling(window_size, center=True, min_periods=1).mean()
    count_total_std = df['total'].rolling(window_size, center=True, min_periods=1).std()

    return count_total_mean, count_total_std

surface_capture_times, surface_counts, count_eggs, count_first, count_two, count_four, count_adv, count_dmg = load_surface_counts(surface_det_path)

surface_count_total_mean, surface_count_total_std = get_surface_mean_n_std(surface_capture_times, count_eggs, count_first, count_two, count_four,
                        count_adv, count_dmg, surface_counts)

#Surface Count given manual count
if scale_detection_results==True:
    manual_count = mc[0]
    first_non_nan_index = surface_count_total_mean.index[surface_count_total_mean.notna() & (surface_count_total_mean != 0)][0]
    cslics_fov_est = (area_tank / manual_count)*surface_count_total_mean[first_non_nan_index] 
    nimage_to_tank_surface = area_tank / (cslics_fov_est)
    print(f'cslics surface count using FOV from manual count = {nimage_to_tank_surface}')

# countperimage_total = count_eggs_mean + count_first_mean + count_two_mean + count_four_mean + count_adv # not counting damaged
surface_decimal_days = convert_to_decimal_days(surface_capture_times)
surface_counttank_total = surface_count_total_mean * nimage_to_tank_surface 
surface_counttank_total_std = surface_count_total_std * nimage_to_tank_surface

##############################################################################
## Plots
##############################################################################

#### subsurface plot
def subsurface_plot(plot_subsurface_days, plot_subsurface_mean, plot_subsurface_std,
                    manual_decimal_days, mc, plot_manual_std,
                    idx_subsuface_manual, dt_idx_subsurface_manual):
    fig1, ax1 = plt.subplots()
    # subsurface counts
    plt.plot(plot_subsurface_days, plot_subsurface_mean, label='subsurface count')
    plt.errorbar(dt_idx_subsurface_manual, plot_subsurface_mean[idx_subsuface_manual], 
                 yerr=plot_subsurface_std[idx_subsuface_manual], fmt='o', color='blue', alpha=0.5)

    # manual counts
    plt.plot(manual_decimal_days, mc, label='manual count', color='green', marker='o', linestyle='--')
    plt.errorbar(manual_decimal_days, mc, yerr=plot_manual_std, fmt='o', color='green', alpha=0.5)
    
    #labeling
    plt.xlabel('days since stocking')
    plt.ylabel('tank count')
    plt.title('Subsurface counts')
    plt.suptitle(plot_title)
    plt.legend()
    plt.savefig(os.path.join(save_plot_dir, "subsurface_counts.png"))

#get location to start and stop the plot
subsurface_start_idx = bisect.bisect_right(decimal_days,  manual_decimal_days[submersion_idx]) - 1
subsurface_stop_idx = bisect.bisect_right(decimal_days, manual_decimal_days[-1])

plot_subsurface_days = decimal_days[subsurface_start_idx:subsurface_stop_idx]
plot_subsurface_mean = subsurface_mean[subsurface_start_idx:subsurface_stop_idx]
plot_subsurface_std = subsurface_std[subsurface_start_idx:subsurface_stop_idx]

#get the error for the subsurface counts at the same time of manual counts
idx_subsuface_manual = []
dt_idx_subsurface_manual = []
for i in manual_decimal_days[submersion_idx:]:
    idx = bisect.bisect_right(plot_subsurface_days, i)
    idx_subsuface_manual.append(idx+subsurface_start_idx-1)
    dt_idx_subsurface_manual.append(plot_subsurface_days[idx-1])

subsurface_plot(plot_subsurface_days, plot_subsurface_mean, plot_subsurface_std,
                manual_decimal_days[submersion_idx:], 
                mc[submersion_idx:], 
                manual_std[submersion_idx:],
                idx_subsuface_manual, dt_idx_subsurface_manual)


#### surface plot
def surface_plot(plot_surface_days, plot_surface_mean, plot_surface_std,
                 manual_decimal_days, mc, plot_manual_std,
                 idx_suface_manual, dt_idx_surface_manual):
    fig2, ax2 = plt.subplots()
    # surface counts
    plt.plot(plot_surface_days, plot_surface_mean, label='surface count', color='orange')
    plt.errorbar(dt_idx_surface_manual, plot_surface_mean[idx_suface_manual], 
                 yerr=plot_surface_std[idx_suface_manual], fmt='o', color='orange', alpha=0.5)

    # manual counts
    plt.plot(manual_decimal_days, mc, label='manual count', color='green', marker='o', linestyle='--')
    plt.errorbar(manual_decimal_days, mc, yerr=plot_manual_std, fmt='o', color='green', alpha=0.5)
    
    
    #labeling
    plt.xlabel('days since stocking')
    plt.ylabel('tank count')
    plt.title('Surface counts')
    plt.suptitle(plot_title)
    plt.legend()
    plt.savefig(os.path.join(save_plot_dir, "surface_counts.png"))

#get location to start and stop the plot
surface_start_idx = bisect.bisect_right(surface_decimal_days,  manual_decimal_days[0]) - 1
surface_stop_idx = bisect.bisect_right(surface_decimal_days, manual_decimal_days[submersion_idx])+1

plot_surface_days = surface_decimal_days[surface_start_idx:surface_stop_idx]
plot_surface_mean = surface_counttank_total[surface_start_idx:surface_stop_idx]
plot_surface_std = surface_counttank_total_std[surface_start_idx:surface_stop_idx]

#get the error for the surface counts at the same time of manual counts
idx_suface_manual = []
dt_idx_surface_manual = []
for i in manual_decimal_days[:submersion_idx+1]:
    idx = bisect.bisect_right(plot_surface_days, i)
    idx_suface_manual.append(idx)
    dt_idx_surface_manual.append(plot_surface_days[idx-1])

surface_plot(plot_surface_days, plot_surface_mean, plot_surface_std,
             manual_decimal_days[:submersion_idx+1], 
             mc[:submersion_idx+1], 
             manual_std[:submersion_idx+1],
             idx_suface_manual, dt_idx_surface_manual)

print('results ploted')
 
import code
code.interact(local=dict(globals(), **locals()))