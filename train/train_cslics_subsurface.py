from ultralytics import YOLO
import torch

# load pretrained model
model = YOLO('/home/dtsai/Data/models/yolov8m.pt')
# The `yolov8m` in this code snippet is referring to a
# pretrained YOLO (You Only Look Once) model named `yolov8m.pt`.
# This model is being loaded using the Ultralytics library in
# Python. YOLO is a popular object detection algorithm that can
# detect objects in images or videos. The `yolov8m` model is
# being used for training on a specific dataset specified in the
# `data_file` variable. The training process includes setting
# various parameters such as the number of epochs, image size,
# number of workers, and batch size. The `model.train()`
# function is used to start the training process for the YOLO
# model.


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

data_file = '/home/dtsai/Code/cslics/coral_spawn_counter/data_yaml_files/cslics_2023_2024_2025_subsurface_M12_ICAM540.yaml'

# train the model
model.train(data=data_file, 
            epochs=500,
            patience=50, 
            imgsz=640,
            workers=4,
            cache=True,
            amp=False,
            batch=4, # was 4
            )

print('done')