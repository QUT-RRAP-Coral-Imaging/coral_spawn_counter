#! /usr/bin/env python3

""" run cslics on Nov 2022 spawning data """

# import surface and subsurface detectors
# setup list of directories
# compute fert ratio, total counts
# calibrate wrt initial count?

import os
import sys

from coral_spawn_counter.Surface_Detector import Surface_Detector
from coral_spawn_counter.SubSurface_Detector import SubSurface_Detector


########### File locations #########
# root_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics04'
# root_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics01'
# root_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221113_amtenuis_cslics03'
# root_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221114_amtenuis_cslics01'
# root_dir = '/home/dorian/Data/cslics_2022_datasets/AIMS_2022_Nov_Spawning/20221114_amtenuis_cslics03'
root_dir = sys.argv[1] # run with commandline/bash script fto do multiple folders at once

img_dir = os.path.join(root_dir, 'images/*')
save_dir = os.path.join(root_dir, 'combined_detections')
# manual_counts_file = os.path.join(root_dir, 'metadata/20221113_ManualCounts_AMaggieTenuis_Tank4-Sheet1.csv')

########### Parameters #########

object_names_file = 'metadata/obj.names'
# weights_file = '/home/dorian/Code/cslics_ws/src/coral_spawn_counter/weights/cslics_20230905_yolov8m_640p_amtenuis1000.pt'
weights_file = os.path.join( '/home/dorian/Code/cslics_ws/src/ultralytics_cslics/weights','cslics_20240117_yolov8x_640p_amt_alor2000.pt')

# for output:
# subsurface_det_file = 'subsurface_det_testing.pkl'
# surface_pkl_file = 'detection_results1.pkl'
# subsurface_det_path = os.path.join(save_dir, subsurface_det_file)
# save_plot_dir = os.path.join(save_dir, 'plots')
# save_img_dir = os.path.join(save_dir, 'images')

# img_list = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
# os.makedirs(save_dir, exist_ok=True)
# os.makedirs(save_plot_dir, exist_ok=True)
# os.makedirs(save_img_dir, exist_ok=True)

max_img = 100
# SD = Surface_Detector(meta_dir=root_dir,
#                       img_dir=img_dir,
#                       save_dir=os.path.join(save_dir, 'surface'),
#                       max_img=max_img, 
#                       weights_file=weights_file)
# SD.run()

SSD = SubSurface_Detector(meta_dir=root_dir,
                          img_dir=img_dir,
                          save_dir=os.path.join(save_dir,'subsurface'),
                          max_img=max_img)
SSD.run()

# import code
# code.interact(local=dict(globals(), **locals()))