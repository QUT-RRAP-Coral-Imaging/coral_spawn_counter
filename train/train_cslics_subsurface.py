from ultralytics import YOLO
import torch

# load pretrained model
model = YOLO('/home/dtsai/Data/models/yolov8m.pt')

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