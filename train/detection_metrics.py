# script to output binary detection metrics from yolo ultralytics model

from ultralytics import YOLO

model_filepath = "/home/dtsai/Data/cslics_datasets/models/20240523_cslics_subsurface_640p_yolov8x.pt"
model = YOLO(model_filepath)

results = model.val(
    data='icra_data.yaml', # path to data.yaml
    split='test', # evaluate on test set
    imgsz=640, # image size
    conf=0.3, # confidence threshold
    iou=0.5, # IoU threshold
    save_json=True, # save results to JSON file
    save=True, # save results to text file
    max_det=2000,
    single_cls=True,
    agnostic_nms=True
)
print(results)

# print("Precision:", results.box.maps[0])  # per-class AP
# print("mAP@0.5:", results.box.map50)      # mean AP at IoU=0.5
# print("mAP@0.5:0.95:", results.box.map)   # mean AP across IoU 0.5:0.95