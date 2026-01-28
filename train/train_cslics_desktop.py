#!/usr/bin/env python3

# script to train cslics desktop models
# used for embryogenesis stages as well as larvae stages (two separate models due to visual ambiguity)



from ultralytics import YOLO
import torch
import os
import glob
import cv2 as cv



# specify location of data file
# data_file = 'cslics_desktop_embryogenesis_2023.yaml'
# data_file = '../data_yaml_files/cslics_desktop_embryogenesis_2025.yaml'
data_file = '../data_yaml_files/cslics_desktop_larvae_2025.yaml'

# load pretrained model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model = YOLO('weights/yolov8n.pt').to(device=device)
model = YOLO('/home/dtsai/Data/models/yolov8n.pt')

# train the model
model.train(data=data_file,
            epochs=400,
            imgsz=1280,
            patience=50,
            workers=4,
            cache=True,
            amp=False,
            batch=4, # for m-model, use batch=2
            device=device)

print('training complete')

# TODO: validate model
# model.val()


# also, run it on all detections from test set

